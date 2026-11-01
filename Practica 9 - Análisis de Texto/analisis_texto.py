import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import os
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols

# Configuración estética global
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'figure.titlesize': 14,
    'figure.dpi': 150
})

# ==============================================================================
# 1. CARGA DE DATOS Y CONSERVACIÓN DEL TEXTO ORIGINAL
# ==============================================================================

# Carga el dataset de muscle cars limpios y estructura los segmentos de condición.
# Conserva la columna 'description' original intacta para la búsqueda de frases exactas.
def cargar_datos(ruta_csv: str) -> pd.DataFrame:
    if not os.path.exists(ruta_csv):
        raise FileNotFoundError(f"No se encontró el dataset en: {ruta_csv}")
    
    df = pd.read_csv(ruta_csv)
    
    map_condicion = {
        'new': 'Excelente',
        'like new': 'Excelente',
        'excellent': 'Excelente',
        'good': 'Buena',
        'fair': 'Regular / Proyecto',
        'salvage': 'Regular / Proyecto'
    }
    df['segmento'] = df['condition'].map(map_condicion).fillna('Desconocido')
    df['desc_raw'] = df['description'].fillna('')
    df['desc_raw_lower'] = df['desc_raw'].str.lower()
    
    return df

# ==============================================================================
# 2. LIMPIEZA, PREPROCESAMIENTO Y LEMATIZACIÓN
# ==============================================================================
# Conjunto de números de especificaciones mecánicas clásicas reconocidas (Opción B)
CILINDRADAS_MECANICAS = {
    '289', '302', '305', '307', '318', '326', '327', '340', '350', '351', '360', 
    '383', '389', '390', '396', '400', '402', '409', '426', '427', '428', '429', 
    '440', '454', '455', '460', '502'
}

STOPWORDS_DOMINIO = {
    'call', 'text', 'phone', 'email', 'contact', 'ask', 'asking', 'obo', 'sale', 
    'sell', 'price', 'cash', 'title', 'miles', 'mileage', 'year', 'car', 'vehicle', 'auto'
}

# Aplica limpieza modular y lematización:
# 1. Remoción de URLs, residuos HTML, correos electrónicos y teléfonos.
# 2. Conversión a minúsculas y normalización de términos compuestos (ej. '4-speed' -> '4speed').
# 3. Filtrado de números: se preservan únicamente especificaciones mecánicas conocidas (Opción B).
# 4. Eliminación de stopwords estándar en inglés + stopwords de dominio aprobadas.
# 5. Lematización canónica mediante WordNetLemmatizer.
def preprocesar_texto(df: pd.DataFrame) -> tuple:
    stop_en = set(stopwords.words('english'))
    todas_stopwords = stop_en.union(STOPWORDS_DOMINIO)
    lemmatizer = WordNetLemmatizer()
    
    html_pat = re.compile(r'<[^>]+>|&(?:nbsp|amp|gt|lt|quot);')
    url_pat = re.compile(r'https?://\S+|www\.\S+')
    email_pat = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
    phone_pat = re.compile(r'(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}|\b\d{3}[-.\s]\d{3}[-.\s]\d{4}\b|\b\d{10}\b')
    
    def limpiar_individual(texto: str) -> str:
        t = texto.lower()
        t = html_pat.sub(' ', t)
        t = url_pat.sub(' ', t)
        t = email_pat.sub(' ', t)
        t = phone_pat.sub(' ', t)
        
        # Unir términos mecánicos con guion antes de separar signos (ej. '4-speed' -> '4speed')
        t = re.sub(r'(\d+)\s*-\s*(speed|spd|bbl|barrel)', r'\1\2', t)
        
        # Tokenización básica conservando palabras y alfanuméricos
        tokens = re.findall(r'\b[a-zA-Z0-9]+\b', t)
        
        tokens_filtrados = []
        for tok in tokens:
            # Si es puramente numérico
            if tok.isdigit():
                if tok in CILINDRADAS_MECANICAS:
                    tokens_filtrados.append(tok)
                continue
                
            if tok in todas_stopwords or len(tok) <= 2:
                continue
                
            # Lematización como sustantivo y verbo
            lema = lemmatizer.lemmatize(tok, pos='v')
            lema = lemmatizer.lemmatize(lema, pos='n')
            
            if lema not in todas_stopwords and len(lema) > 2:
                tokens_filtrados.append(lema)
                
        return ' '.join(tokens_filtrados)
        
    df['desc_clean'] = df['desc_raw'].apply(limpiar_individual)
    return df, todas_stopwords

# ==============================================================================
# 3. NUBES DE PALABRAS Y FRECUENCIA LÉXICA
# ==============================================================================

# Genera nubes de palabras con estilo visual idéntico para el corpus global y
# por cada segmento de condición. Extrae las 20 palabras más frecuentes.
def generar_nubes_y_frecuencias(df: pd.DataFrame, carpeta_salida: str) -> dict:
    os.makedirs(carpeta_salida, exist_ok=True)
    segmentos = ['Global', 'Regular / Proyecto', 'Buena', 'Excelente', 'Desconocido']
    paletas = {
        'Global': 'viridis',
        'Regular / Proyecto': 'Oranges_r',
        'Buena': 'Purples_r',
        'Excelente': 'Greens_r',
        'Desconocido': 'magma'
    }
    
    tablas_top20 = {}
    
    # 1. Nube Global
    texto_global = ' '.join(df['desc_clean'])
    wc_global = WordCloud(width=1000, height=550, background_color='white', 
                          colormap=paletas['Global'], max_words=100, random_state=42).generate(texto_global)
    
    plt.figure(figsize=(11, 6))
    plt.imshow(wc_global, interpolation='bilinear')
    plt.axis('off')
    plt.title("Nube de Palabras Global: Corpus de Descripciones de Muscle Cars (N = 5,703)", pad=12, fontweight='bold', fontsize=13)
    plt.tight_layout()
    ruta_wc_glob = os.path.join(carpeta_salida, "grafica_2_nube_palabras_global.png")
    plt.savefig(ruta_wc_glob, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Conteo top 20 global
    vec_glob = CountVectorizer(max_features=20)
    X_glob = vec_glob.fit_transform(df['desc_clean'])
    tablas_top20['Global'] = pd.DataFrame({
        'Palabra': vec_glob.get_feature_names_out(),
        'Frecuencia': np.asarray(X_glob.sum(axis=0)).flatten()
    }).sort_values('Frecuencia', ascending=False).reset_index(drop=True)
    
    # 2. Nubes por Segmento (Panel Combinado 2x2 para comparación directa)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    seg_coords = [
        ('Regular / Proyecto', 0, 0),
        ('Buena', 0, 1),
        ('Excelente', 1, 0),
        ('Desconocido', 1, 1)
    ]
    
    for seg, r, c in seg_coords:
        ax = axes[r, c]
        sub = df[df['segmento'] == seg]
        texto_seg = ' '.join(sub['desc_clean'])
        
        wc_seg = WordCloud(width=600, height=380, background_color='white', 
                           colormap=paletas[seg], max_words=80, random_state=42).generate(texto_seg)
        ax.imshow(wc_seg, interpolation='bilinear')
        ax.axis('off')
        ax.set_title(f"Segmento: {seg} (n = {len(sub):,})", pad=8, fontweight='bold', fontsize=12)
        
        vec_seg = CountVectorizer(max_features=20)
        X_seg = vec_seg.fit_transform(sub['desc_clean'])
        tablas_top20[seg] = pd.DataFrame({
            'Palabra': vec_seg.get_feature_names_out(),
            'Frecuencia': np.asarray(X_seg.sum(axis=0)).flatten()
        }).sort_values('Frecuencia', ascending=False).reset_index(drop=True)
        
    fig.suptitle("Nubes de Palabras Comparativas por Segmento de Condición", fontsize=15, y=0.98, fontweight='bold')
    plt.tight_layout()
    ruta_wc_seg = os.path.join(carpeta_salida, "grafica_3_nubes_palabras_por_segmento.png")
    plt.savefig(ruta_wc_seg, dpi=300, bbox_inches='tight')
    plt.close()
    
    return tablas_top20

# ==============================================================================
# 4. ANÁLISIS DE N-GRAMAS Y TÉRMINOS DISTINTIVOS (TF-IDF)
# ==============================================================================

# Calcula:
# 1. Frecuencia de Bigramas y Trigramas más comunes a nivel global y por segmento.
# 2. Puntuación TF-IDF para identificar términos distintivos de cada segmento.
# Fundamento Matemático de TF-IDF:
# TF-IDF(t, d, D) = TF(t, d) * IDF(t, D)
# - TF(t, d): Frecuencia relativa del término t en el documento o segmento d.
# - IDF(t, D) = ln((1 + |D|) / (1 + DF(t))) + 1: Ponderación inversa de documentos.
# Penaliza palabras omnipresentes en todos los segmentos y maximiza el peso de términos
# especializados de un segmento específico (ej. '426 hemi' en 'Excelente').
def analizar_ngrams_y_tfidf(df: pd.DataFrame, carpeta_salida: str) -> tuple:
    # 1. Bigramas Globales
    vec_bi = CountVectorizer(ngram_range=(2, 2), max_features=15)
    X_bi = vec_bi.fit_transform(df['desc_clean'])
    df_bi = pd.DataFrame({
        'Bigrama': vec_bi.get_feature_names_out(),
        'Frecuencia': np.asarray(X_bi.sum(axis=0)).flatten()
    }).sort_values('Frecuencia', ascending=False).reset_index(drop=True)
    
    # 2. Trigramas Globales
    vec_tri = CountVectorizer(ngram_range=(3, 3), max_features=15)
    X_tri = vec_tri.fit_transform(df['desc_clean'])
    df_tri = pd.DataFrame({
        'Trigrama': vec_tri.get_feature_names_out(),
        'Frecuencia': np.asarray(X_tri.sum(axis=0)).flatten()
    }).sort_values('Frecuencia', ascending=False).reset_index(drop=True)
    
    # Gráfica 4: Palabras y Bigramas más Frecuentes
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    vec_words = CountVectorizer(max_features=12)
    X_words = vec_words.fit_transform(df['desc_clean'])
    df_words = pd.DataFrame({
        'Palabra': vec_words.get_feature_names_out(),
        'Frecuencia': np.asarray(X_words.sum(axis=0)).flatten()
    }).sort_values('Frecuencia', ascending=True)
    
    ax1.barh(df_words['Palabra'], df_words['Frecuencia'], color='steelblue', edgecolor='black', alpha=0.85)
    ax1.set_title("Top 12 Palabras Lematizadas Globales", pad=10, fontweight='bold')
    ax1.set_xlabel("Frecuencia Absoluta")
    ax1.grid(True, linestyle='--', alpha=0.6)
    
    df_bi_plot = df_bi.head(12).sort_values('Frecuencia', ascending=True)
    ax2.barh(df_bi_plot['Bigrama'], df_bi_plot['Frecuencia'], color='coral', edgecolor='black', alpha=0.85)
    ax2.set_title("Top 12 Bigramas Más Frecuentes", pad=10, fontweight='bold')
    ax2.set_xlabel("Frecuencia Absoluta")
    ax2.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    ruta_g4 = os.path.join(carpeta_salida, "grafica_4_frecuencia_palabras_y_bigramas.png")
    plt.savefig(ruta_g4, dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. TF-IDF por Segmento (Tratando el texto agregado de cada segmento como un documento)
    segmentos = ['Regular / Proyecto', 'Buena', 'Excelente', 'Desconocido']
    docs_segmento = [ ' '.join(df[df['segmento'] == seg]['desc_clean']) for seg in segmentos ]
    
    tfidf_vec = TfidfVectorizer(max_features=1500, ngram_range=(1, 2))
    tfidf_mat = tfidf_vec.fit_transform(docs_segmento).toarray()
    nombres_features = np.array(tfidf_vec.get_feature_names_out())
    
    top_tfidf_por_segmento = {}
    for i, seg in enumerate(segmentos):
        scores = tfidf_mat[i]
        top_indices = np.argsort(scores)[::-1][:15]
        top_tfidf_por_segmento[seg] = pd.DataFrame({
            'Término / Frase': nombres_features[top_indices],
            'Puntuación TF-IDF': scores[top_indices]
        })
        
    return df_bi, df_tri, top_tfidf_por_segmento

# ==============================================================================
# 5. PRUEBA DE LAS DOS HIPÓTESIS CON LISTAS DE FRASES
# ==============================================================================

# Evalúa empíricamente las dos hipótesis sobre el texto crudo en minúsculas:
# Hipótesis 1: Lenguaje 'Highballer' / Precio Firme.
# Comprueba si los vendedores del segmento 'Regular / Proyecto' utilizan con mayor
# frecuencia frases de intransigencia en el precio en comparación con los demás segmentos.
# Hipótesis 2: Variables Omitidas y Calidad Oculta.
# Crea indicadores binarios de atributos cualitativos clave ('numbers matching', 'restored', 
# 'rust free', etc.), evalúa su impacto en precio con la Prueba t de Welch y calcula la mejora
# en R² de un modelo de regresión lineal múltiple al incluir estas variables.
def evaluar_hipotesis_textuales(df: pd.DataFrame, carpeta_salida: str) -> tuple:
    desc_lower = df['desc_raw_lower']
    
    # --------------------------------------------------------------------------
    # Hipótesis 1: Frases Highballer / Precio Firme
    # --------------------------------------------------------------------------
    patrones_highballer = [
        r'no\s+low\s*ballers?',
        r'low\s*ballers?\s+(?:will be|ignored)',
        r'serious\s+(?:buyers?|inquiries|inquiry)\s+only',
        r'price\s+is\s+firm',
        r'firm\s+price',
        r'firm\s+on\s+price',
        r'i\s+know\s+what\s+i\s+have',
        r'not\s+giving\s+it\s+away',
        r'don\'?t\s+waste\s+my\s+time',
        r'no\s+tire\s*kickers?'
    ]
    regex_highballer = re.compile('|'.join(patrones_highballer))
    df['is_highballer'] = desc_lower.apply(lambda s: bool(regex_highballer.search(s)))
    
    tabla_hb = []
    for seg in ['Regular / Proyecto', 'Buena', 'Excelente', 'Desconocido']:
        sub = df[df['segmento'] == seg]
        n_tot = len(sub)
        n_hb = sub['is_highballer'].sum()
        pct = (n_hb / n_tot) * 100
        tabla_hb.append({
            'Segmento': seg,
            'Anuncios_Totales': n_tot,
            'Con_Frase_Highballer': n_hb,
            'Porcentaje (%)': round(pct, 2)
        })
    df_hb = pd.DataFrame(tabla_hb)
    
    # Gráfica 5: Prevalencia de Lenguaje Highballer por Segmento
    plt.figure(figsize=(9, 5.5))
    barras = plt.bar(df_hb['Segmento'], df_hb['Porcentaje (%)'], 
                     color=['#d95f02', '#7570b3', '#1b9e77', '#666666'], edgecolor='black', alpha=0.85)
    plt.title("Prevalencia de Lenguaje 'Highballer' / Precio Firme por Segmento de Condición", pad=12, fontweight='bold')
    plt.ylabel("Porcentaje de Anuncios con al menos una Frase (%)")
    plt.ylim(0, 10)
    for b in barras:
        h = b.get_height()
        plt.annotate(f"{h:.2f}%\n(n={int(b.get_height()*len(df)/100)})" if False else f"{h:.2f}%", 
                     xy=(b.get_x() + b.get_width() / 2, h),
                     xytext=(0, 4), textcoords="offset points", ha='center', va='bottom', fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    ruta_g5 = os.path.join(carpeta_salida, "grafica_5_prevalencia_lenguaje_highballer.png")
    plt.savefig(ruta_g5, dpi=300, bbox_inches='tight')
    plt.close()
    
    # --------------------------------------------------------------------------
    # Hipótesis 2: Variables Omitidas y Prueba t de Welch
    # --------------------------------------------------------------------------
    dicc_omitidas = {
        'Numbers Matching': [r'numbers?\s+matching', r'matching\s+numbers?'],
        'All Original': [r'\ball\s+original\b', r'\b100%\s+original\b'],
        'Survivor': [r'\bsurvivor\b'],
        'Restored / Frame-off': [r'\brestored\b', r'\brestoration\b', r'\bframe\s*off\b', r'\brotisserie\b'],
        'Rust Free / No Rust': [r'rust\s*free', r'no\s+rust'],
        'Barn Find': [r'barn\s*find'],
        'Documented / Build Sheet': [r'\bdocumented\b', r'build\s*sheet', r'protecto\s*plate', r'window\s*sticker']
    }
    
    tabla_omitidas = []
    for etiqueta, patrones in dicc_omitidas.items():
        rx = re.compile('|'.join(patrones))
        col_bin = f"has_{etiqueta.lower().replace(' ', '_').replace('/', '_').replace('-', '_')}"
        df[col_bin] = desc_lower.apply(lambda s: bool(rx.search(s)))
        
        con = df[df[col_bin] == True]['price']
        sin = df[df[col_bin] == False]['price']
        
        # Test t de Welch (varianzas desiguales)
        t_stat, p_val = stats.ttest_ind(con, sin, equal_var=False)
        diff_media = con.mean() - sin.mean()
        diff_pct = (diff_media / sin.mean()) * 100
        
        # Tamaño del efecto d de Cohen
        n1, n2 = len(con), len(sin)
        s1, s2 = con.std(), sin.std()
        s_pool = np.sqrt(((n1 - 1)*s1**2 + (n2 - 1)*s2**2) / (n1 + n2 - 2))
        cohen_d = diff_media / s_pool
        
        tabla_omitidas.append({
            'Frase / Variable Omitida': etiqueta,
            'Con Frase (n)': n1,
            'Media Con ($)': round(con.mean(), 2),
            'Mediana Con ($)': con.median(),
            'Sin Frase (n)': n2,
            'Media Sin ($)': round(sin.mean(), 2),
            'Diferencia ($)': round(diff_media, 2),
            'Diferencia (%)': round(diff_pct, 2),
            't de Welch': round(t_stat, 2),
            'p-value': f"{p_val:.2e}",
            'd de Cohen': round(cohen_d, 2)
        })
    df_omitidas = pd.DataFrame(tabla_omitidas)
    
    # Gráfica 6: Comparativa de Precio con y sin Frases Clave
    plt.figure(figsize=(11, 6))
    x_indices = np.arange(len(df_omitidas))
    ancho = 0.35
    
    plt.bar(x_indices - ancho/2, df_omitidas['Media Con ($)'], ancho, label='Con la Frase en Descripción', color='#1b9e77', edgecolor='black', alpha=0.85)
    plt.bar(x_indices + ancho/2, df_omitidas['Media Sin ($)'], ancho, label='Sin la Frase', color='#7570b3', edgecolor='black', alpha=0.85)
    
    plt.title("Impacto de Frases Clave de Valor en el Precio Medio Pedido (USD)", pad=12, fontweight='bold')
    plt.xlabel("Variable Textual / Frase de Calidad")
    plt.ylabel("Precio Promedio Ofertado (USD)")
    plt.gca().yaxis.set_major_formatter('${x:,.0f}')
    plt.xticks(x_indices, df_omitidas['Frase / Variable Omitida'], rotation=25, ha='right')
    plt.legend(loc='upper right', frameon=True)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    ruta_g6 = os.path.join(carpeta_salida, "grafica_6_impacto_precio_variables_omitidas.png")
    plt.savefig(ruta_g6, dpi=300, bbox_inches='tight')
    plt.close()
    
    # --------------------------------------------------------------------------
    # Aumento del Modelo D de la Práctica 5 con Variables de Texto
    # --------------------------------------------------------------------------
    df_reg = df.copy()
    df_reg['is_convertible'] = (df_reg['type'] == 'convertible').astype(int)
    df_reg['is_clean_title'] = (df_reg['title_status'] == 'clean').astype(int)
    df_reg['is_good_condition'] = df_reg['condition'].isin(['good', 'excellent', 'like new', 'new']).astype(int)
    df_reg['cylinders_num'] = df_reg['cylinders'].str.extract(r'(\d+)').astype(float).fillna(8)
    
    df_reg['txt_numbers_matching'] = df_reg['has_numbers_matching'].astype(int)
    df_reg['txt_restored'] = df_reg['has_restored___frame_off'].astype(int)
    df_reg['txt_documented'] = df_reg['has_documented___build_sheet'].astype(int)
    df_reg['txt_rust_free'] = df_reg['has_rust_free___no_rust'].astype(int)
    
    # Modelo D Original
    mod_d_orig = ols('price ~ year + I(year**2) + odometer + is_convertible + is_clean_title + is_good_condition + cylinders_num', data=df_reg).fit()
    
    # Modelo D Aumentado con Texto
    mod_d_text = ols('price ~ year + I(year**2) + odometer + is_convertible + is_clean_title + is_good_condition + cylinders_num + txt_numbers_matching + txt_restored + txt_documented + txt_rust_free', data=df_reg).fit()
    
    tabla_reg = pd.DataFrame([
        {
            'Modelo': 'Modelo D Original (Práctica 5)',
            'Variables': 'Variables numéricas + dummies tabulares',
            'R2': round(mod_d_orig.rsquared, 4),
            'R2_Ajustado': round(mod_d_orig.rsquared_adj, 4),
            'AIC': round(mod_d_orig.aic, 1),
            'BIC': round(mod_d_orig.bic, 1),
            'RMSE ($)': round(np.sqrt(mod_d_orig.mse_resid), 2)
        },
        {
            'Modelo': 'Modelo D Aumentado con Texto (Práctica 9)',
            'Variables': 'Modelo D + 4 variables binarias textuales',
            'R2': round(mod_d_text.rsquared, 4),
            'R2_Ajustado': round(mod_d_text.rsquared_adj, 4),
            'AIC': round(mod_d_text.aic, 1),
            'BIC': round(mod_d_text.bic, 1),
            'RMSE ($)': round(np.sqrt(mod_d_text.mse_resid), 2)
        }
    ])
    
    coef_text = pd.DataFrame({
        'Variable Textual': ['txt_numbers_matching', 'txt_restored', 'txt_documented', 'txt_rust_free'],
        'Coeficiente ($)': [mod_d_text.params[v] for v in ['txt_numbers_matching', 'txt_restored', 'txt_documented', 'txt_rust_free']],
        'p-value': [mod_d_text.pvalues[v] for v in ['txt_numbers_matching', 'txt_restored', 'txt_documented', 'txt_rust_free']]
    })
    
    return df_hb, df_omitidas, tabla_reg, coef_text

if __name__ == "__main__":
    carpeta_practica = os.path.dirname(os.path.abspath(__file__))
    carpeta_graficas = os.path.join(carpeta_practica, "graficas")
    os.makedirs(carpeta_graficas, exist_ok=True)
    ruta_csv = os.path.join(carpeta_practica, "..", "Practica 1 - Limpieza de Datos", "muscle_cars_clean.csv")
    
    print("=" * 80)
    print("EJECUTANDO ANÁLISIS DE TEXTO Y PRUEBA DE HIPÓTESIS (PRÁCTICA 9)")
    print("=" * 80)
    
    df = cargar_datos(ruta_csv)
    print(f"Observaciones cargadas: {len(df):,} registros.")
    
    # 1. Limpieza y Lematización
    print("\n>>> [1/4] Preprocesando texto (URLs, HTML, teléfonos, lematización WordNet y especificaciones mecánicas)...")
    df, stopwords_usadas = preprocesar_texto(df)
    print(f"Texto procesado. Vocabulario de stopwords aplicado: {len(stopwords_usadas)} términos.")
    
    # 2. Nubes de palabras y tablas de frecuencia
    print("\n>>> [2/4] Generando nubes de palabras y tablas de frecuencia léxica...")
    tablas_top20 = generar_nubes_y_frecuencias(df, carpeta_graficas)
    print("\nTop 10 palabras más frecuentes en el corpus global:")
    print(tablas_top20['Global'].head(10).to_string(index=False))
    
    # 3. N-gramas y TF-IDF
    print("\n>>> [3/4] Calculando N-gramas (bigramas, trigramas) y términos distintivos por TF-IDF...")
    df_bi, df_tri, top_tfidf = analizar_ngrams_y_tfidf(df, carpeta_graficas)
    print("\nTop 5 Bigramas Globales:")
    print(df_bi.head(5).to_string(index=False))
    
    # 4. Evaluación de Hipótesis y Aumento de Modelo
    print("\n>>> [4/4] Evaluando Hipótesis 1 (Highballers) e Hipótesis 2 (Variables Omitidas)...")
    df_hb, df_omitidas, tabla_reg, coef_text = evaluar_hipotesis_textuales(df, carpeta_graficas)
    
    print("\n--- PREVALENCIA DE FRASES HIGHBALLER POR SEGMENTO ---")
    print(df_hb.to_string(index=False))
    
    print("\n--- VARIABLES OMITIDAS Y TEST T DE WELCH ---")
    print(df_omitidas[['Frase / Variable Omitida', 'Con Frase (n)', 'Media Con ($)', 'Media Sin ($)', 'Diferencia (%)', 't de Welch', 'p-value', 'd de Cohen']].to_string(index=False))
    
    print("\n--- COMPARATIVA MODELO D ORIGINAL vs. MODELO D AUMENTADO CON TEXTO ---")
    print(tabla_reg.to_string(index=False))
    
    print("\nCoeficientes de las variables de texto en el Modelo D Aumentado:")
    for _, r in coef_text.iterrows():
        print(f"  {r['Variable Textual']:24s}: Coef = +${r['Coeficiente ($)']:8.2f} USD | p-value = {r['p-value']:.2e}")
        
    print("\n" + "=" * 80)
    print("EJECUCIÓN COMPLETADA CON ÉXITO. TODAS LAS FIGURAS EXPORTADAS A 300 DPI EN 'graficas/'.")
    print("=" * 80)
