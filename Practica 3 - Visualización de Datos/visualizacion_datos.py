import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Configuración estética global de Seaborn y Matplotlib
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.titlesize': 14,
    'figure.dpi': 150
})

def cargar_y_preparar_datos(ruta_csv: str) -> pd.DataFrame:
    if not os.path.exists(ruta_csv):
        raise FileNotFoundError(f"No se encontró el dataset en: {ruta_csv}")
    
    df = pd.read_csv(ruta_csv)
    
    # Derivación de la columna 'decade' con intervalos históricos estándar
    bins = [1959, 1969, 1974, 1979, 1988, 2004]
    labels = ['1960-1969', '1970-1974', '1975-1979', '1980-1988', '1989-2004']
    df['decade'] = pd.cut(df['year'], bins=bins, labels=labels, right=True)
    
    return df

def crear_directorio_graficas(carpeta_salida: str) -> None:
    os.makedirs(carpeta_salida, exist_ok=True)

# ==============================================================================
# 1. HISTOGRAMAS
# ==============================================================================
def generar_histogramas_numericos(df: pd.DataFrame, num_cols: list, carpeta_salida: str) -> str:
    fig, axes = plt.subplots(1, len(num_cols), figsize=(16, 5))
    
    # BUCLE FOR: Automatización de la generación de subplots
    for i, col in enumerate(num_cols):
        ax = axes[i]
        sns.histplot(df[col], kde=True, ax=ax, color='steelblue', bins=30, edgecolor='black', alpha=0.6)
        
        # Formato técnico de ejes y títulos
        media_val = df[col].mean()
        mediana_val = df[col].median()
        ax.axvline(media_val, color='red', linestyle='--', linewidth=1.5, label=f'Media: {media_val:,.1f}')
        ax.axvline(mediana_val, color='green', linestyle='-', linewidth=1.5, label=f'Mediana: {mediana_val:,.1f}')
        
        ax.set_title(f"Distribución de '{col}'")
        ax.set_xlabel(col.capitalize())
        ax.set_ylabel("Frecuencia (Conteo)")
        ax.legend(loc='upper right')
        
    plt.tight_layout()
    ruta_guardado = os.path.join(carpeta_salida, "grafica_1a_histogramas_numericos.png")
    plt.savefig(ruta_guardado, dpi=300, bbox_inches='tight')
    plt.close()
    return ruta_guardado

def generar_histograma_facetado_por_decada(df: pd.DataFrame, carpeta_salida: str) -> str:
    decadas = df['decade'].cat.categories
    fig, axes = plt.subplots(len(decadas), 1, figsize=(10, 14), sharex=True)
    
    paleta = sns.color_palette("tab10", len(decadas))
    
    # BUCLE FOR: Automatización por cada estrato temporal
    for i, dec in enumerate(decadas):
        ax = axes[i]
        subset = df[df['decade'] == dec]['price']
        
        sns.histplot(subset, kde=True, ax=ax, color=paleta[i], bins=35, edgecolor='black', alpha=0.6)
        
        mediana = subset.median()
        media = subset.mean()
        ax.axvline(mediana, color='black', linestyle='-', linewidth=1.5, label=f'Mediana: ${mediana:,.0f}')
        ax.axvline(media, color='darkred', linestyle='--', linewidth=1.5, label=f'Media: ${media:,.0f}')
        
        ax.set_title(f"Década: {dec} (n = {len(subset):,})", fontsize=11, fontweight='bold')
        ax.set_ylabel("Frecuencia")
        ax.legend(loc='upper right')
        
    axes[-1].set_xlabel("Precio Ofertado en USD ($)")
    plt.tight_layout()
    ruta_guardado = os.path.join(carpeta_salida, "grafica_1b_histograma_price_por_decada.png")
    plt.savefig(ruta_guardado, dpi=300, bbox_inches='tight')
    plt.close()
    return ruta_guardado

# ==============================================================================
# 2. DIAGRAMAS DE CAJA / BOXPLOTS
# ==============================================================================
def crear_boxplot_agrupado(df: pd.DataFrame, cat_col: str, num_col: str, 
                           carpeta_salida: str, nombre_archivo: str, 
                           orden: list = None, rotacion_x: int = 0) -> str:
    plt.figure(figsize=(11, 6))
    
    if orden is None:
        orden = df.groupby(cat_col)[num_col].median().sort_values(ascending=False).index.tolist()
        
    ax = sns.boxplot(
        data=df, 
        x=cat_col, 
        y=num_col, 
        hue=cat_col,
        order=orden,
        palette="Set2",
        legend=False,
        showmeans=True,
        meanprops={"marker":"o", "markerfacecolor":"red", "markeredgecolor":"black", "markersize":"6"}
    )
    
    ax.set_title(f"Distribución de '{num_col}' Agrupada por '{cat_col}' (Mediana y Rango Intercuartílico)", pad=12)
    ax.set_xlabel(cat_col.capitalize())
    ax.set_ylabel(f"{num_col.capitalize()} (USD)")
    if rotacion_x > 0:
        plt.xticks(rotation=rotacion_x, ha='right')
        
    # Elemento visual para la leyenda de la media aritmética
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='Media Aritmética', markerfacecolor='red', markeredgecolor='black', markersize=8)
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    ruta_guardado = os.path.join(carpeta_salida, nombre_archivo)
    plt.savefig(ruta_guardado, dpi=300, bbox_inches='tight')
    plt.close()
    return ruta_guardado

# ==============================================================================
# 3. DIAGRAMAS DE DISPERSIÓN / SCATTER PLOTS (AUTOMATIZADOS)
# ==============================================================================
def generar_scatter_plots(df: pd.DataFrame, carpeta_salida: str) -> list:
    rutas = []
    
    # A. Price vs. Odometer
    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        data=df, 
        x='odometer', 
        y='price', 
        alpha=0.35, 
        color='navy', 
        edgecolor=None,
        s=30
    )
    sns.regplot(
        data=df, 
        x='odometer', 
        y='price', 
        scatter=False, 
        color='crimson', 
        label='Tendencia Lineal',
        line_kws={'linewidth': 2}
    )
    plt.title("Diagrama de Dispersión: Precio vs. Odómetro (Millas)", pad=12)
    plt.xlabel("Odómetro (Millas Registradas)")
    plt.ylabel("Precio Ofertado (USD)")
    plt.legend(loc='upper right')
    plt.tight_layout()
    ruta_a = os.path.join(carpeta_salida, "grafica_3a_scatter_price_vs_odometer.png")
    plt.savefig(ruta_a, dpi=300, bbox_inches='tight')
    plt.close()
    rutas.append(ruta_a)
    
    # B. Price vs. Year (Coloreado por los Fabricantes Principales)
    top_mfg = ['chevrolet', 'ford', 'pontiac', 'dodge']
    df_mfg = df.copy()
    df_mfg['mfg_group'] = df_mfg['manufacturer'].apply(lambda x: x if x in top_mfg else 'Otros')
    
    plt.figure(figsize=(11, 6))
    sns.scatterplot(
        data=df_mfg, 
        x='year', 
        y='price', 
        hue='mfg_group', 
        alpha=0.45, 
        palette='Set1',
        s=35
    )
    plt.title("Diagrama de Dispersión: Precio vs. Año de Fabricación (Segmentado por Marca)", pad=12)
    plt.xlabel("Año de Fabricación (Year)")
    plt.ylabel("Precio Ofertado (USD)")
    plt.legend(title="Fabricante", loc='upper left')
    plt.tight_layout()
    ruta_b = os.path.join(carpeta_salida, "grafica_3b_scatter_price_vs_year_hue_fabricante.png")
    plt.savefig(ruta_b, dpi=300, bbox_inches='tight')
    plt.close()
    rutas.append(ruta_b)
    
    return rutas

# ==============================================================================
# 4. DIAGRAMAS DE PASTEL / PIE CHARTS 
# ==============================================================================
def crear_pie_chart(df: pd.DataFrame, col: str, carpeta_salida: str, 
                    nombre_archivo: str, top_n: int = 5) -> str:
    conteo = df[col].value_counts()
    
    if len(conteo) > top_n:
        top_cats = conteo.head(top_n)
        otros_sum = conteo.iloc[top_n:].sum()
        serie_pie = pd.concat([top_cats, pd.Series({'Otros': otros_sum})])
    else:
        serie_pie = conteo
        
    plt.figure(figsize=(8, 8))
    colores = sns.color_palette("pastel", len(serie_pie))
    
    pie_res = plt.pie(
        serie_pie, 
        labels=serie_pie.index, 
        autopct='%1.1f%%', 
        startangle=140,
        colors=colores,
        wedgeprops=dict(width=0.7, edgecolor='white', linewidth=2), # Estilo Donut elegante
        pctdistance=0.75
    )
    
    # Manejo seguro del retorno de plt.pie (evita error de desempaquetado de tuplas de tamaño 2 o 3)
    if len(pie_res) == 3:
        wedges, texts, autotexts = pie_res
        for autotext in autotexts:
            autotext.set_color('black')
            autotext.set_fontweight('bold')
    elif len(pie_res) == 2:
        wedges, texts = pie_res
        
    plt.title(f"Proporción Composicional de Anuncios por '{col.capitalize()}' (n = {len(df):,})", pad=15)
    plt.tight_layout()
    ruta_guardado = os.path.join(carpeta_salida, nombre_archivo)
    plt.savefig(ruta_guardado, dpi=300, bbox_inches='tight')
    plt.close()
    return ruta_guardado

# ==============================================================================
# 5. QUINTO TIPO: GRÁFICO DE LÍNEAS DE TENDENCIA TEMPORAL CON BANDAS DE CONFIANZA
# ==============================================================================
def generar_lineplot_tendencia_temporal(df: pd.DataFrame, carpeta_salida: str) -> str:
    top_brands = ['chevrolet', 'ford', 'pontiac', 'dodge']
    df_sub = df[df['manufacturer'].isin(top_brands)].copy()
    
    plt.figure(figsize=(12, 6))
    ax = sns.lineplot(
        data=df_sub, 
        x='year', 
        y='price', 
        hue='manufacturer', 
        estimator='mean', 
        errorbar=('ci', 95),
        palette='tab10',
        linewidth=2.2,
        marker='o',
        markersize=5
    )
    
    # Línea de referencia global
    media_global = df['price'].mean()
    ax.axhline(media_global, color='gray', linestyle=':', label=f'Media Global (${media_global:,.0f})')
    
    ax.set_title("Evolución Temporal del Precio Medio por Año de Fabricación (Intervalo de Confianza al 95%)", pad=12)
    ax.set_xlabel("Año de Fabricación (1960 - 2004)")
    ax.set_ylabel("Precio Medio Estimado (USD)")
    ax.set_xlim(1960, 2004)
    plt.legend(title="Fabricante", loc='upper right')
    plt.tight_layout()
    
    ruta_guardado = os.path.join(carpeta_salida, "grafica_5_lineplot_tendencia_temporal_fabricante.png")
    plt.savefig(ruta_guardado, dpi=300, bbox_inches='tight')
    plt.close()
    return ruta_guardado

# ==============================================================================
# 6. SEXTO TIPO: DIAGRAMAS DE VIOLÍN
# ==============================================================================
def generar_violinplots_densidad(df: pd.DataFrame, carpeta_salida: str) -> str:
    plt.figure(figsize=(12, 6))
    
    ax = sns.violinplot(
        data=df, 
        x='decade', 
        y='price', 
        hue='decade',
        palette='Spectral',
        legend=False,
        inner='quartile',
        cut=0,
        density_norm='width'
    )
    
    ax.set_title("Diagrama de Violín: Densidad y Distribución de Precio por Década", pad=12)
    ax.set_xlabel("Década de Fabricación")
    ax.set_ylabel("Precio Ofertado (USD)")
    plt.tight_layout()
    
    ruta_guardado = os.path.join(carpeta_salida, "grafica_6_violinplot_price_por_decada.png")
    plt.savefig(ruta_guardado, dpi=300, bbox_inches='tight')
    plt.close()
    return ruta_guardado

# ==============================================================================
# FUNCIÓN PRINCIPAL DE EJECUCIÓN
# ==============================================================================
def main():
    ruta_dataset = os.path.join('Practica 1 - Limpieza de Datos', 'muscle_cars_clean.csv')
    carpeta_salida = os.path.join('Practica 3 - Visualización de Datos', 'graficas')
    
    print("="*80)
    print("INICIANDO GENERACIÓN AUTOMATIZADA DE GRÁFICAS (PRÁCTICA 3)")
    print("="*80)
    
    df = cargar_y_preparar_datos(ruta_dataset)
    crear_directorio_graficas(carpeta_salida)
    print(f"Dataset cargado con éxito: {len(df):,} filas. Carpeta de destino: '{carpeta_salida}'\n")
    
    # 1. Histogramas
    print(">>> [1/6] Generando histogramas univariados y facetados con loops...")
    r1a = generar_histogramas_numericos(df, ['price', 'year', 'odometer'], carpeta_salida)
    r1b = generar_histograma_facetado_por_decada(df, carpeta_salida)
    print(f"    Guardado: {r1a}")
    print(f"    Guardado: {r1b}")
    
    # 2. Boxplots
    print("\n>>> [2/6] Generando diagramas de caja con función reutilizable...")
    r2a = crear_boxplot_agrupado(df, 'manufacturer', 'price', carpeta_salida, 'grafica_2a_boxplot_price_por_fabricante.png', rotacion_x=30)
    r2b = crear_boxplot_agrupado(df, 'condition', 'price', carpeta_salida, 'grafica_2b_boxplot_price_por_condicion.png')
    print(f"    Guardado: {r2a}")
    print(f"    Guardado: {r2b}")
    
    # 3. Scatter Plots
    print("\n>>> [3/6] Generando diagramas de dispersión bivariados...")
    rutas_scatter = generar_scatter_plots(df, carpeta_salida)
    for r in rutas_scatter:
        print(f"    Guardado: {r}")
        
    # 4. Pie Charts
    print("\n>>> [4/6] Generando diagramas de pastel con agrupación automática...")
    r4a = crear_pie_chart(df, 'manufacturer', carpeta_salida, 'grafica_4a_pie_anuncios_por_fabricante.png', top_n=5)
    r4b = crear_pie_chart(df, 'decade', carpeta_salida, 'grafica_4b_pie_anuncios_por_decada.png', top_n=5)
    print(f"    Guardado: {r4a}")
    print(f"    Guardado: {r4b}")
    
    # 5. Quinto Tipo: Líneas de Tendencia Temporal (Lineplot con IC 95%)
    print("\n>>> [5/6] Generando gráfico de líneas de tendencia temporal por fabricante...")
    r5 = generar_lineplot_tendencia_temporal(df, carpeta_salida)
    print(f"    Guardado: {r5}")
    
    # 6. Sexto Tipo: Diagramas de Violín (Violin Plots)
    print("\n>>> [6/6] Generando diagramas de violín para densidad de probabilidad...")
    r6 = generar_violinplots_densidad(df, carpeta_salida)
    print(f"    Guardado: {r6}")
    
    print("\n" + "="*80)
    print("TODAS LAS VISUALIZACIONES FUERON GENERADAS Y EXPORTADAS EXITOSAMENTE A 300 DPI.")
    print("="*80)

if __name__ == '__main__':
    main()
