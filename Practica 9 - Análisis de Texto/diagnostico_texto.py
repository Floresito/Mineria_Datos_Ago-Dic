import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import os
import nltk
from nltk.corpus import stopwords

# Configuración estética global
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'figure.titlesize': 14,
    'figure.dpi': 150
})

# Carga el dataset y asigna los segmentos de condición definidos en la Práctica 8.
def cargar_y_estructurar_datos(ruta_csv: str) -> pd.DataFrame:
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
    df['char_len'] = df['desc_raw'].str.len()
    df['word_len'] = df['desc_raw'].apply(lambda s: len(s.split()))
    
    return df

# Calcula las métricas de diagnóstico sobre el texto crudo sin alterar.
def diagnosticar_metricas_texto(df: pd.DataFrame) -> dict:
    # 1. Vacíos y nulos
    nulos = df['description'].isna().sum()
    vacios = (df['desc_raw'].str.strip() == '').sum()
    
    # 2. Residuos técnicos (HTML, URL, Email, Teléfono)
    html_pat = re.compile(r'<[^>]+>|&(?:nbsp|amp|gt|lt|quot);')
    url_pat = re.compile(r'https?://\S+|www\.\S+')
    email_pat = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
    phone_pat = re.compile(r'(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}|\b\d{3}[-.\s]\d{3}[-.\s]\d{4}\b|\b\d{10}\b')
    
    has_html = df['desc_raw'].apply(lambda s: bool(html_pat.search(s)))
    has_url = df['desc_raw'].apply(lambda s: bool(url_pat.search(s)))
    has_email = df['desc_raw'].apply(lambda s: bool(email_pat.search(s)))
    has_phone = df['desc_raw'].apply(lambda s: bool(phone_pat.search(s)))
    
    # 3. Boilerplate / texto repetido
    conteos_desc = df['desc_raw'].value_counts()
    unicas = len(conteos_desc)
    repetidas = (conteos_desc > 1).sum()
    filas_repetidas = conteos_desc[conteos_desc > 1].sum()
    
    # 4. Idioma
    stop_en = set(stopwords.words('english'))
    stop_es = set(stopwords.words('spanish'))
    
    def detectar(s):
        w = set(re.findall(r'[a-zA-ZáéíóúñÁÉÍÓÚÑ]+', s.lower()))
        if not w: return 'vacio'
        en_c = len(w.intersection(stop_en))
        es_c = len(w.intersection(stop_es))
        if en_c == 0 and es_c == 0: return 'indeterminado'
        return 'espanol' if es_c > en_c else 'ingles'
    
    df['idioma'] = df['desc_raw'].apply(detectar)
    idioma_dist = df['idioma'].value_counts()
    
    res = {
        'total_filas': len(df),
        'nulos': nulos,
        'vacios': vacios,
        'has_html': has_html.sum(),
        'has_url': has_url.sum(),
        'has_email': has_email.sum(),
        'has_phone': has_phone.sum(),
        'desc_unicas': unicas,
        'desc_repetidas_tipos': repetidas,
        'filas_repetidas': filas_repetidas,
        'idioma_dist': idioma_dist
    }
    return res

# Genera gráfica de distribución de longitud en palabras (Boxplot por segmento e Histograma global).
def generar_grafica_diagnostico_longitud(df: pd.DataFrame, carpeta_salida: str) -> None:
    os.makedirs(carpeta_salida, exist_ok=True)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    orden_seg = ['Regular / Proyecto', 'Buena', 'Excelente', 'Desconocido']
    paleta = ['#d95f02', '#7570b3', '#1b9e77', '#666666']
    
    # Panel A: Boxplot por segmento (acotado al percentil 95 para visualización limpia)
    p95 = df['word_len'].quantile(0.95)
    sns.boxplot(data=df, x='segmento', y='word_len', hue='segmento', order=orden_seg, palette=paleta, ax=ax1, showfliers=False, legend=False)
    ax1.set_title("A. Distribución de Longitud de Palabras por Segmento\n(Sin valores atípicos extremos para escala)", pad=10, fontweight='bold')
    ax1.set_xlabel("Segmento de Condición")
    ax1.set_ylabel("Cantidad de Palabras por Anuncio")
    ax1.grid(True, linestyle='--', alpha=0.6)
    
    # Anotar medianas
    medianas = df.groupby('segmento')['word_len'].median()
    for i, seg in enumerate(orden_seg):
        med = medianas[seg]
        ax1.text(i, med + 4, f"Med: {med:.0f}", ha='center', va='bottom', fontsize=9, fontweight='bold',
                 bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="black", alpha=0.8))
        
    # Panel B: Histograma y KDE global
    sns.histplot(df['word_len'], bins=40, binrange=(0, p95), kde=True, color='navy', ax=ax2)
    ax2.axvline(df['word_len'].median(), color='crimson', linestyle='--', linewidth=2, label=f"Mediana Global = {df['word_len'].median():.0f} palabras")
    ax2.axvline(df['word_len'].mean(), color='forestgreen', linestyle=':', linewidth=2, label=f"Media Global = {df['word_len'].mean():.1f} palabras")
    ax2.set_title("B. Distribución de Frecuencia de Longitud (Palabras)\n(Histograma y Densidad hasta Percentil 95)", pad=10, fontweight='bold')
    ax2.set_xlabel("Cantidad de Palabras por Anuncio")
    ax2.set_ylabel("Frecuencia (Cantidad de Anuncios)")
    ax2.legend(loc='upper right', frameon=True)
    ax2.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    ruta_g1 = os.path.join(carpeta_salida, "grafica_1_diagnostico_longitud_descripciones.png")
    plt.savefig(ruta_g1, dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    carpeta_practica = os.path.dirname(os.path.abspath(__file__))
    carpeta_graficas = os.path.join(carpeta_practica, "graficas")
    ruta_csv = os.path.join(carpeta_practica, "..", "Practica 1 - Limpieza de Datos", "muscle_cars_clean.csv")
    
    print("=" * 80)
    print("EJECUTANDO DIAGNÓSTICO EXPLORATORIO DE TEXTO (PRÁCTICA 9 - PUNTO 1)")
    print("=" * 80)
    
    df = cargar_y_estructurar_datos(ruta_csv)
    res = diagnosticar_metricas_texto(df)
    
    print(f"Total observaciones: {res['total_filas']:,} registros.")
    print(f"Descripciones nulas: {res['nulos']} | Vacías (espacios): {res['vacios']}")
    print(f"Descripciones únicas: {res['desc_unicas']:,} ({res['desc_unicas']/res['total_filas']*100:.2f}%)")
    print(f"Anuncios con descripciones repetidas (boilerplate): {res['filas_repetidas']:,} ({res['filas_repetidas']/res['total_filas']*100:.2f}%)")
    print(f"Anuncios con URLs: {res['has_url']:,} ({res['has_url']/res['total_filas']*100:.2f}%)")
    print(f"Anuncios con Números Telefónicos: {res['has_phone']:,} ({res['has_phone']/res['total_filas']*100:.2f}%)")
    print(f"Anuncios con Correos Electrónicos: {res['has_email']} ({res['has_email']/res['total_filas']*100:.2f}%)")
    print(f"Anuncios con Residuos HTML: {res['has_html']} ({res['has_html']/res['total_filas']*100:.2f}%)")
    
    generar_grafica_diagnostico_longitud(df, carpeta_graficas)
    print(f"\nGráfica exportada a 300 DPI:")
    print(f" - {os.path.join(carpeta_graficas, 'grafica_1_diagnostico_longitud_descripciones.png')}")
    print("=" * 80)
