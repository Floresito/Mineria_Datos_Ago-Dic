import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.decomposition import PCA
import os

# Configuración estética global
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'figure.titlesize': 14,
    'figure.dpi': 150
})

# Carga el dataset y prepara los datos
# price, odometer - Continuas
# year, cylinders - Discretas
def cargar_y_preparar_datos(ruta_csv: str) -> tuple:
    if not os.path.exists(ruta_csv):
        raise FileNotFoundError(f"No se encontró el dataset en: {ruta_csv}")
    
    df = pd.read_csv(ruta_csv)
    
    # Extracción numérica de cilindrada ('8 cylinders' -> 8.0)
    df['cylinders_num'] = df['cylinders'].str.extract(r'(\d+)').astype(float).fillna(8)
    
    features = ['price', 'odometer', 'year', 'cylinders_num']
    X = df[features].copy()
    
    return df, X, features

# Crea el directorio para guardar las gráficas
def crear_directorio_graficas(carpeta_salida: str) -> None:
    os.makedirs(carpeta_salida, exist_ok=True)

# Estandarización de características (StandardScaler)
def estandarizar_caracteristicas(X: pd.DataFrame) -> tuple:
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler

# Evaluación del hiperparámetro K: Método del Codo y Silhouette Score
def evaluar_rango_k(X_scaled: np.ndarray, k_min: int = 2, k_max: int = 10, 
                    n_init: int = 30, random_state: int = 42, carpeta_salida: str = "") -> pd.DataFrame:
    # Calcula la curva de inercia (WCSS) y el Silhouette Score promedio para K entre 2 y 10.
    k_valores = list(range(k_min, k_max + 1))
    inercias = []
    silhouettes = []
    
    for k in k_valores:
        km = KMeans(n_clusters=k, init='k-means++', n_init=n_init, random_state=random_state)
        etiquetas = km.fit_predict(X_scaled)
        
        inercias.append(km.inertia_)
        sil_score = silhouette_score(X_scaled, etiquetas)
        silhouettes.append(sil_score)
        
    df_eval = pd.DataFrame({
        'K': k_valores,
        'Inercia': inercias,
        'Silhouette_Score': silhouettes
    })
    
    # Gráfica 1: Método del Codo (Inercia vs K)
    plt.figure(figsize=(9, 5.5))
    plt.plot(k_valores, inercias, marker='o', linestyle='-', color='navy', linewidth=2, markersize=7)
    plt.axvline(4, color='crimson', linestyle='--', linewidth=1.8, label='K = 4 Seleccionado')
    plt.title("Método del Codo: Inercia (WCSS) vs. Número de Clústers (K)", pad=12)
    plt.xlabel("Número de Clústers (K)")
    plt.ylabel("Inercia (Suma Cuadrática Intra-Clúster)")
    plt.xticks(k_valores)
    plt.legend(loc='upper right')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    ruta_codo = os.path.join(carpeta_salida, "grafica_1_metodo_codo.png")
    plt.savefig(ruta_codo, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Gráfica 2: Silhouette Score Promedio vs K
    plt.figure(figsize=(9, 5.5))
    plt.plot(k_valores, silhouettes, marker='s', linestyle='-', color='crimson', linewidth=2, markersize=7)
    plt.axvline(4, color='navy', linestyle='--', linewidth=1.8, label='K = 4 (Silhouette = 0.3508)')
    plt.scatter([4], [silhouettes[2]], color='navy', s=120, zorder=5)
    plt.title("Coeficiente de Silueta Promedio vs. Número de Clústers (K)", pad=12)
    plt.xlabel("Número de Clústers (K)")
    plt.ylabel("Silhouette Score Promedio")
    plt.xticks(k_valores)
    plt.legend(loc='upper right')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    ruta_sil = os.path.join(carpeta_salida, "grafica_2_silhouette_score.png")
    plt.savefig(ruta_sil, dpi=300, bbox_inches='tight')
    plt.close()
    
    return df_eval

# Entrenamiento del modelo final y extracción de centroides 
def entrenar_modelo_final(X_scaled: np.ndarray, scaler: StandardScaler, df_original: pd.DataFrame, 
                          features: list, k_elegido: int = 4, n_init: int = 30, random_state: int = 42) -> tuple:
    km_final = KMeans(n_clusters=k_elegido, init='k-means++', n_init=n_init, random_state=random_state)
    cluster_labels = km_final.fit_predict(X_scaled)
    
    df_resultado = df_original.copy()
    df_resultado['cluster'] = cluster_labels
    
    # Métricas del modelo final
    inercia_final = km_final.inertia_
    silhouette_global = silhouette_score(X_scaled, cluster_labels)
    
    # Des-estandarización de centroides para interpretación cuantitativa
    centroides_escalados = km_final.cluster_centers_
    centroides_originales = scaler.inverse_transform(centroides_escalados)
    
    df_centroides = pd.DataFrame(
        centroides_originales,
        columns=features,
        index=[f"Clúster {i}" for i in range(k_elegido)]
    )
    
    # Métricas descriptivas completas por clúster
    conteo_clusters = df_resultado['cluster'].value_counts().sort_index()
    df_centroides['n_observaciones'] = conteo_clusters.values
    df_centroides['porcentaje'] = (conteo_clusters.values / len(df_resultado)) * 100
    
    resumen_media = df_resultado.groupby('cluster')[features].mean()
    resumen_mediana = df_resultado.groupby('cluster')[features].median()
    resumen_std = df_resultado.groupby('cluster')[features].std()
    
    resumen_media.index = [f"Clúster {i}" for i in resumen_media.index]
    resumen_mediana.index = [f"Clúster {i}" for i in resumen_mediana.index]
    resumen_std.index = [f"Clúster {i}" for i in resumen_std.index]
    
    return km_final, df_resultado, df_centroides, resumen_media, resumen_mediana, resumen_std, inercia_final, silhouette_global

# Visualizaciones de clusters: 2D scatter y PCA
def generar_graficas_clusters(df_resultado: pd.DataFrame, df_centroides: pd.DataFrame, 
                              X_scaled: np.ndarray, cluster_labels: np.ndarray, 
                              k_elegido: int, carpeta_salida: str) -> None:
    paleta = sns.color_palette("tab10", k_elegido)
    
    # Gráfica 3: Proyección 2D (Price vs Year)
    plt.figure(figsize=(10, 6.5))
    for i in range(k_elegido):
        subset = df_resultado[df_resultado['cluster'] == i]
        plt.scatter(subset['year'], subset['price'], label=f"Clúster {i} (n={len(subset):,})", 
                    color=paleta[i], alpha=0.45, s=25, edgecolors='none')
        
    plt.scatter(df_centroides['year'], df_centroides['price'], color='black', marker='X', s=180, 
                linewidths=2, edgecolors='white', zorder=10, label='Centroides de Clúster')
    
    for i, row in df_centroides.iterrows():
        plt.annotate(i, (row['year'], row['price']), textcoords="offset points", 
                     xytext=(0, 10), ha='center', fontweight='bold', fontsize=10,
                     bbox=dict(boxstyle="round,pad=0.2", fc="yellow", alpha=0.75, ec="black"))
        
    plt.title(f"Agrupamiento K-Means (K={k_elegido}): Proyección Precio vs. Año de Fabricación", pad=12)
    plt.xlabel("Año de Fabricación (year)")
    plt.ylabel("Precio de Venta en USD (price)")
    plt.legend(loc='upper right', frameon=True)
    plt.tight_layout()
    ruta_scatter = os.path.join(carpeta_salida, "grafica_3_clusters_price_vs_year.png")
    plt.savefig(ruta_scatter, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Gráfica 4: Proyección PCA 2D
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    var_exp = pca.explained_variance_ratio_
    
    plt.figure(figsize=(10, 6.5))
    for i in range(k_elegido):
        idx = np.where(cluster_labels == i)[0]
        plt.scatter(X_pca[idx, 0], X_pca[idx, 1], label=f"Clúster {i} (n={len(idx):,})", 
                    color=paleta[i], alpha=0.45, s=25, edgecolors='none')
        
    plt.title(f"Proyección PCA 2D de Clústers K-Means (K={k_elegido})\n"
              f"Varianza Acumulada Explicada: {sum(var_exp)*100:.1f}% (PC1: {var_exp[0]*100:.1f}%, PC2: {var_exp[1]*100:.1f}%)", pad=12)
    plt.xlabel(f"Componente Principal 1 ({var_exp[0]*100:.1f}% Varianza)")
    plt.ylabel(f"Componente Principal 2 ({var_exp[1]*100:.1f}% Varianza)")
    plt.legend(loc='upper right', frameon=True)
    plt.tight_layout()
    ruta_pca = os.path.join(carpeta_salida, "grafica_4_clusters_pca_2d.png")
    plt.savefig(ruta_pca, dpi=300, bbox_inches='tight')
    plt.close()

# Validación cruzada contra variables no incluidas en el modelo
def generar_tablas_cruzadas(df_resultado: pd.DataFrame) -> tuple:
    ct_condition_abs = pd.crosstab(df_resultado['cluster'], df_resultado['condition'], margins=True)
    ct_condition_pct = pd.crosstab(df_resultado['cluster'], df_resultado['condition'], normalize='index') * 100
    
    ct_manufacturer_abs = pd.crosstab(df_resultado['cluster'], df_resultado['manufacturer'], margins=True)
    ct_manufacturer_pct = pd.crosstab(df_resultado['cluster'], df_resultado['manufacturer'], normalize='index') * 100
    
    return ct_condition_abs, ct_condition_pct, ct_manufacturer_abs, ct_manufacturer_pct

# Evaluación adicional: correlación en subpoblación de 8 cilindros (V8)
def evaluar_correlacion_v8_odometro_precio(df: pd.DataFrame, carpeta_salida: str) -> dict:
    # Dataset completo
    r_full, p_full = stats.pearsonr(df['odometer'], df['price'])
    rho_full, p_rho_full = stats.spearmanr(df['odometer'], df['price'])
    
    # Subconjunto V8 (cylinders_num == 8.0)
    df_v8 = df[df['cylinders_num'] == 8.0].copy()
    r_v8, p_v8 = stats.pearsonr(df_v8['odometer'], df_v8['price'])
    rho_v8, p_rho_v8 = stats.spearmanr(df_v8['odometer'], df_v8['price'])
    
    # Ajuste lineal en subconjunto V8: price = slope * odometer + intercept
    slope_v8, intercept_v8, r_val, p_val, std_err = stats.linregress(df_v8['odometer'], df_v8['price'])
    
    # Gráfica 5: Scatter Price vs Odometer en V8 con Línea de Tendencia
    plt.figure(figsize=(10, 6.5))
    plt.scatter(df_v8['odometer'], df_v8['price'], color='steelblue', alpha=0.35, s=22, edgecolors='none', label=f'Autos V8 (n={len(df_v8):,})')
    
    x_vals = np.linspace(df_v8['odometer'].min(), df_v8['odometer'].max(), 300)
    y_vals = slope_v8 * x_vals + intercept_v8
    plt.plot(x_vals, y_vals, color='crimson', linewidth=2.5, 
             label=f'Regresión Lineal: Price = {slope_v8:.4f}·Odo + ${intercept_v8:,.0f}\nPearson r = {r_v8:.4f} (R² = {r_v8**2:.4f}, p = {p_v8:.2e})')
    
    plt.title("Dispersión Precio vs. Odómetro en Subpoblación V8 (8 Cilindros)", pad=12)
    plt.xlabel("Kilometraje / Odómetro en Millas (odometer)")
    plt.ylabel("Precio de Venta en USD (price)")
    plt.legend(loc='upper right', frameon=True, fontsize=10)
    plt.tight_layout()
    ruta_scatter_v8 = os.path.join(carpeta_salida, "grafica_5_correlacion_v8_price_vs_odometer.png")
    plt.savefig(ruta_scatter_v8, dpi=300, bbox_inches='tight')
    plt.close()
    
    metricas = {
        'n_full': len(df),
        'r_full': r_full,
        'p_full': p_full,
        'rho_full': rho_full,
        'p_rho_full': p_rho_full,
        'n_v8': len(df_v8),
        'r_v8': r_v8,
        'p_v8': p_v8,
        'rho_v8': rho_v8,
        'p_rho_v8': p_rho_v8,
        'slope_v8': slope_v8,
        'intercept_v8': intercept_v8,
        'r2_v8': r_v8**2
    }
    
    return metricas

if __name__ == "__main__":
    carpeta_practica = os.path.dirname(os.path.abspath(__file__))
    carpeta_graficas = os.path.join(carpeta_practica, "graficas")
    crear_directorio_graficas(carpeta_graficas)
    
    ruta_dataset = os.path.join(carpeta_practica, "..", "Practica 1 - Limpieza de Datos", "muscle_cars_clean.csv")
    
    print("=" * 80)
    print("EJECUTANDO AGRUPAMIENTO DE DATOS CON K-MEANS (PRÁCTICA 7)")
    print("=" * 80)
    
    df, X, features = cargar_y_preparar_datos(ruta_dataset)
    print(f"Observaciones cargadas: {len(df):,} registros.")
    print(f"Variables seleccionadas para agrupamiento: {features}")
    
    # 1. Estandarización
    X_scaled, scaler = estandarizar_caracteristicas(X)
    print("\n>>> [1/5] Estandarización completada con StandardScaler (media=0, std=1).")
    
    # 2. Evaluación de K de 2 a 10
    print("\n>>> [2/5] Evaluando rango de K = 2 a K = 10...")
    df_eval = evaluar_rango_k(X_scaled, k_min=2, k_max=10, n_init=30, random_state=42, carpeta_salida=carpeta_graficas)
    print(df_eval.to_string(index=False))
    
    # 3. Entrenamiento del Modelo Final (K = 4)
    k_elegido = 4
    print(f"\n>>> [3/5] Entrenando modelo definitivo con K = {k_elegido} (n_init=30, k-means++)...")
    km_final, df_resultado, df_centroides, resumen_media, resumen_mediana, resumen_std, inercia_final, sil_final = entrenar_modelo_final(
        X_scaled, scaler, df, features, k_elegido=k_elegido, n_init=30, random_state=42
    )
    
    print(f"\nMétricas del Modelo Final (K={k_elegido}):")
    print(f" - Inercia Global (WCSS):  {inercia_final:.2f}")
    print(f" - Silhouette Score Global: {sil_final:.4f}")
    
    print("\nTABLA DE CENTROIDES (UNIDADES ORIGINALES DE NEGOCIO):")
    print(df_centroides.round(2).to_string())
    
    print("\nRESUMEN DE MEDIANAS POR CLÚSTER:")
    print(resumen_mediana.round(2).to_string())
    
    print("\nRESUMEN DE DESVIACIONES ESTÁNDAR (DISPERSIÓN INTERNA):")
    print(resumen_std.round(2).to_string())
    
    # 4. Gráficas y Tablas Cruzadas
    print(f"\n>>> [4/5] Generando visualizaciones y matrices de validación cruzada...")
    generar_graficas_clusters(df_resultado, df_centroides, X_scaled, df_resultado['cluster'].values, k_elegido, carpeta_graficas)
    
    ct_cond_abs, ct_cond_pct, ct_manuf_abs, ct_manuf_pct = generar_tablas_cruzadas(df_resultado)
    
    print("\nTABLA CRUZADA: CLÚSTER VS CONDITION (% POR FILA):")
    print(ct_cond_pct.round(2).to_string())
    
    print("\nTABLA CRUZADA: CLÚSTER VS MANUFACTURER (% POR FILA):")
    print(ct_manuf_pct.round(2).to_string())
    
    # 5. Verificación de Correlación V8
    print(f"\n>>> [5/5] Evaluando correlación Odómetro vs. Precio en subconjunto V8...")
    res_v8 = evaluar_correlacion_v8_odometro_precio(df, carpeta_graficas)
    print(f"Dataset Completo (N={res_v8['n_full']}): Pearson r = {res_v8['r_full']:.4f} (p={res_v8['p_full']:.2e}), Spearman rho = {res_v8['rho_full']:.4f}")
    print(f"Subconjunto V8   (N={res_v8['n_v8']}): Pearson r = {res_v8['r_v8']:.4f} (p={res_v8['p_v8']:.2e}), Spearman rho = {res_v8['rho_v8']:.4f}")
    print(f"Ecuación V8: Price = {res_v8['slope_v8']:.4f} * Odometer + ${res_v8['intercept_v8']:,.2f} (R² = {res_v8['r2_v8']:.4f})")
    
    print("\n" + "=" * 80)
    print("EJECUCIÓN COMPLETADA CON ÉXITO. TODAS LAS FIGURAS EXPORTADAS A 300 DPI.")
    print("=" * 80)
