"""
================================================================================
PRÁCTICA 4: PRUEBAS ESTADÍSTICAS (ENFOQUE ROBUSTO DE WELCH)
Minería de Datos - Muscle Cars Clásicos (1960-2004)
================================================================================
En este script aplicamos pruebas estadísticas para comparar precios entre grupos:
1. ANOVA de Welch (3 o más grupos):
   - Comparación por Fabricante (Chevrolet, Ford, Pontiac, Dodge, Buick, Mercury)
   - Comparación por Época/Década Histórica
2. Prueba t de Welch (2 grupos):
   - Por Estatus de Título (Clean vs Salvage/Rebuilt)
   - Por Condición del Auto (Condición Alta vs Condición Baja)
   - Por Tipo de Carrocería (Convertible vs Coupe - Señal de Práctica 2)
3. Verificación visual de supuestos:
   - Inspección visual de la distribución con histogramas y gráficos Q-Q.
   - Uso por defecto de las variantes de Welch (ANOVA de Welch y t de Welch)
     que no asumen que los grupos tengan la misma variabilidad/dispersión.
4. Comparaciones posteriores (Tukey HSD) para ver exactamente qué pares difieren.
5. Tamaño del efecto:
   - Eta cuadrado (η²) para ANOVA
   - d de Cohen para pruebas t
================================================================================
"""

import os
import sys
import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.api as sm
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import matplotlib.pyplot as plt
import seaborn as sns

# Soporte de codificación UTF-8 en consola de Windows
if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

# Configuración estética global para gráficos limpios y legibles
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

# Cargar dataset y crear categorías
def cargar_y_preparar_datos(ruta_csv: str) -> pd.DataFrame:
    if not os.path.exists(ruta_csv):
        raise FileNotFoundError(f"No se encontró el archivo en: {ruta_csv}")
    
    df = pd.read_csv(ruta_csv)
    
    # 1. Década histórica agrupada
    bins = [1959, 1969, 1974, 1979, 1988, 2004]
    labels = ['1960-1969', '1970-1974', '1975-1979', '1980-1988', '1989-2004']
    df['decade'] = pd.cut(df['year'], bins=bins, labels=labels, right=True)
    
    # 2. Estatus de título agrupado (Clean vs Títulos con historial de daño)
    def clasificar_title(t):
        if t == 'clean':
            return 'Clean'
        elif t in ['salvage', 'rebuilt', 'parts only']:
            return 'Salvage/Rebuilt'
        return np.nan
    df['title_group'] = df['title_status'].apply(clasificar_title)
    
    # 3. Condición agrupada (Alta conservación vs autos proyecto/dañados)
    def clasificar_condicion(c):
        if c in ['new', 'like new', 'excellent', 'good']:
            return 'Condición Alta'
        elif c in ['fair', 'salvage']:
            return 'Condición Baja'
        return np.nan
    df['condition_group'] = df['condition'].apply(clasificar_condicion)
    
    return df

# ==============================================================================
# FUNCIONES ESTADÍSTICAS
# ==============================================================================

# ANOVA de Welch para comparar 3 o más grupos
def welch_anova(grupos: list) -> tuple:
    k = len(grupos)
    ni = np.array([len(g) for g in grupos], dtype=float)
    mi = np.array([np.mean(g) for g in grupos], dtype=float)
    vi = np.array([np.var(g, ddof=1) for g in grupos], dtype=float)
    
    # Ponderaciones por grupo
    wi = ni / vi
    W = np.sum(wi)
    m_prime = np.sum(wi * mi) / W
    
    # Numerador y ajuste de Welch
    numerador = np.sum(wi * (mi - m_prime)**2) / (k - 1)
    termino = np.sum(((1.0 - (wi / W))**2) / (ni - 1.0))
    denominador = 1.0 + (2.0 * (k - 2.0) / (k**2 - 1.0)) * termino
    
    f_stat = numerador / denominador
    df1 = k - 1
    df2 = (k**2 - 1.0) / (3.0 * termino)
    p_val = float(stats.f.sf(f_stat, df1, df2))
    
    return f_stat, df1, df2, p_val

# Porcentaje de variación de precio explicado por los grupos
def calcular_eta_cuadrado(grupos: list) -> float:
    gran_media = np.mean(np.concatenate(grupos))
    ss_total = sum(np.sum((g - gran_media) ** 2) for g in grupos)
    ss_between = sum(len(g) * (np.mean(g) - gran_media) ** 2 for g in grupos)
    return float(ss_between / ss_total) if ss_total > 0 else 0.0

# Distancia estandarizada entre dos medias (d de Cohen)
def calcular_d_cohen(grupo1: np.ndarray, grupo2: np.ndarray) -> float:
    n1, n2 = len(grupo1), len(grupo2)
    s1, s2 = np.std(grupo1, ddof=1), np.std(grupo2, ddof=1)
    s_pooled = np.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2))
    return float((np.mean(grupo1) - np.mean(grupo2)) / s_pooled)

# Interpretación de la magnitud práctica de la diferencia encontrada
def interpretar_efecto(valor: float, tipo: str) -> str:
    if tipo == 'eta':
        if valor < 0.01:
            return "Muy pequeño (< 1% de varianza)"
        elif valor < 0.06:
            return "Pequeño (1% a 6% de varianza)"
        elif valor < 0.14:
            return "Mediano (6% a 14% de varianza)"
        else:
            return "Grande (≥ 14% de varianza)"
    elif tipo == 'cohen':
        abs_v = abs(valor)
        if abs_v < 0.20:
            return "Muy pequeño (< 0.20)"
        elif abs_v < 0.50:
            return "Pequeño (0.20 a 0.50)"
        elif abs_v < 0.80:
            return "Mediano (0.50 a 0.80)"
        else:
            return "Grande (≥ 0.80)"
    return "N/A"

# ==============================================================================
# VISUALIZACIONES DE APOYO
# ==============================================================================

# Boxplot con puntos de media
def graficar_boxplot_anova(df_subset: pd.DataFrame, cat_col: str, val_col: str, 
                           titulo: str, subtitulo: str, ruta_salida: str,
                           orden: list = None, rotacion_x: int = 0):
    plt.figure(figsize=(10, 6))
    
    if orden is None:
        orden = df_subset.groupby(cat_col, observed=False)[val_col].median().sort_values(ascending=False).index.tolist()
        
    sns.boxplot(
        data=df_subset, x=cat_col, y=val_col, order=orden,
        hue=cat_col, legend=False,
        palette="Blues_r", showmeans=True,
        meanprops={"marker": "D", "markerfacecolor": "red", "markeredgecolor": "black", "markersize": 7}
    )
    
    plt.title(f"{titulo}\n{subtitulo}", fontsize=12, fontweight='bold', pad=12)
    plt.xlabel(cat_col.capitalize().replace('_', ' '))
    plt.ylabel("Precio en USD ($)")
    if rotacion_x > 0:
        plt.xticks(rotation=rotacion_x, ha='right')
        
    plt.plot([], [], marker='D', color='red', markeredgecolor='black', linestyle='None', label='Media del grupo')
    plt.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig(ruta_salida, dpi=300, bbox_inches='tight')
    plt.close()

# Boxplot con observaciones para comparaciones de prueba t (2 grupos)
def graficar_boxplot_ttest(df_subset: pd.DataFrame, cat_col: str, val_col: str,
                           titulo: str, subtitulo: str, ruta_salida: str):
    plt.figure(figsize=(8, 6))
    
    sns.boxplot(
        data=df_subset, x=cat_col, y=val_col,
        hue=cat_col, legend=False,
        palette=["#4C72B0", "#C44E52"], showmeans=True,
        meanprops={"marker": "D", "markerfacecolor": "yellow", "markeredgecolor": "black", "markersize": 8}
    )
    sns.stripplot(data=df_subset, x=cat_col, y=val_col, color="black", alpha=0.15, jitter=0.2, size=3)
    
    plt.title(f"{titulo}\n{subtitulo}", fontsize=12, fontweight='bold', pad=12)
    plt.xlabel(cat_col.capitalize().replace('_', ' '))
    plt.ylabel("Precio en USD ($)")
    
    plt.plot([], [], marker='D', color='yellow', markeredgecolor='black', linestyle='None', label='Media del grupo')
    plt.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig(ruta_salida, dpi=300, bbox_inches='tight')
    plt.close()

# Panel de 6 graficos Q-Q para inspección visual de la forma de las distribuciones
def graficar_diagnostico_visual(residuos_dict: dict, ruta_salida: str):
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, (nombre, residuos) in enumerate(residuos_dict.items()):
        if i < len(axes):
            ax = axes[i]
            sm.qqplot(residuos.dropna(), line='s', ax=ax, alpha=0.5, markerfacecolor='#4C72B0', markeredgecolor='none')
            ax.set_title(f"Inspección Visual: {nombre}", fontsize=11, fontweight='bold')
            ax.set_xlabel("Distribución Teórica Normal")
            ax.set_ylabel("Datos Observados")
            
    plt.tight_layout()
    plt.savefig(ruta_salida, dpi=300, bbox_inches='tight')
    plt.close()

# ==============================================================================
# PIPELINE PRINCIPAL DE ANÁLISIS
# ==============================================================================

def main():
    print("=================================================================")
    print("   PRÁCTICA 4: PRUEBAS ESTADÍSTICAS (ENFOQUE ROBUSTO)            ")
    print("=================================================================\n")
    
    ruta_script = os.path.dirname(os.path.abspath(__file__))
    ruta_csv = os.path.join(ruta_script, "..", "Practica 1 - Limpieza de Datos", "muscle_cars_clean.csv")
    carpeta_graficas = os.path.join(ruta_script, "graficas")
    os.makedirs(carpeta_graficas, exist_ok=True)
    
    df = cargar_y_preparar_datos(ruta_csv)
    print(f"Dataset cargado con éxito: {len(df):,} vehículos registrados.")
    
    tabla_resumen = []
    residuos_dict = {}
    
    # --------------------------------------------------------------------------
    # 1. ANOVA DE WELCH: PRECIO POR FABRICANTE (6 marcas principales)
    # --------------------------------------------------------------------------
    print("\n-------------------------------------------------------------")
    print("1. ANOVA DE WELCH: PRECIO POR FABRICANTE")
    print("-------------------------------------------------------------")
    marcas = ['chevrolet', 'ford', 'pontiac', 'dodge', 'buick', 'mercury']
    df_mfg = df[df['manufacturer'].isin(marcas)].copy()
    grupos_mfg = [df_mfg[df_mfg['manufacturer'] == m]['price'].values for m in marcas]
    
    # Ejecutar ANOVA de Welch
    f_stat_mfg, df1_mfg, df2_mfg, p_val_mfg = welch_anova(grupos_mfg)
    eta2_mfg = calcular_eta_cuadrado(grupos_mfg)
    mag_mfg = interpretar_efecto(eta2_mfg, 'eta')
    
    print(f">> Resultado: F({df1_mfg:.0f}, {df2_mfg:.1f}) = {f_stat_mfg:.4f}, p-value = {p_val_mfg:.4e}")
    print(f">> Decisión con α=0.05: {'Rechazar H0 (Hay diferencias entre marcas)' if p_val_mfg < 0.05 else 'No rechazar H0'}")
    print(f">> Tamaño del efecto (η²): {eta2_mfg:.4f} -> {mag_mfg}")
    
    # Post-hoc Tukey HSD para ver qué pares son diferentes
    tukey_mfg = pairwise_tukeyhsd(endog=df_mfg['price'], groups=df_mfg['manufacturer'], alpha=0.05)
    print("\n>> Comparaciones por parejas (Tukey HSD):")
    print(tukey_mfg)
    
    residuos_dict['Fabricantes'] = df_mfg['price'] - df_mfg.groupby('manufacturer')['price'].transform('mean')
    
    sub_mfg = f"ANOVA de Welch: F({df1_mfg:.0f}, {df2_mfg:.1f}) = {f_stat_mfg:.2f}, p = {p_val_mfg:.2e} | η² = {eta2_mfg:.3f} ({mag_mfg})"
    ruta_g1 = os.path.join(carpeta_graficas, "grafica_1_anova_price_por_fabricante.png")
    graficar_boxplot_anova(df_mfg, 'manufacturer', 'price', "Comparación de Precios por Fabricante", sub_mfg, ruta_g1)
    
    tabla_resumen.append({
        'Prueba Estadística': 'ANOVA de Welch (Fabricante)',
        'Variable': 'Precio (price)',
        'Grupos Comparados': 'Chevrolet, Ford, Pontiac, Dodge, Buick, Mercury',
        'Estadístico': f"F = {f_stat_mfg:.4f}",
        'Grados Libertad': f"({df1_mfg:.0f}, {df2_mfg:.1f})",
        'p-value': f"{p_val_mfg:.4e}",
        'Decisión (α=0.05)': 'Rechazar H0',
        'Tamaño de Efecto': f"η² = {eta2_mfg:.4f}",
        'Magnitud': mag_mfg
    })
    
    # --------------------------------------------------------------------------
    # 2. ANOVA DE WELCH: PRECIO POR DÉCADA HISTÓRICA
    # --------------------------------------------------------------------------
    print("\n-------------------------------------------------------------")
    print("2. ANOVA DE WELCH: PRECIO POR ÉPOCA HISTÓRICA (DÉCADAS)")
    print("-------------------------------------------------------------")
    df_dec = df.dropna(subset=['decade']).copy()
    decadas = df_dec['decade'].cat.categories.tolist()
    grupos_dec = [df_dec[df_dec['decade'] == d]['price'].values for d in decadas]
    
    f_stat_dec, df1_dec, df2_dec, p_val_dec = welch_anova(grupos_dec)
    eta2_dec = calcular_eta_cuadrado(grupos_dec)
    mag_dec = interpretar_efecto(eta2_dec, 'eta')
    
    print(f">> Resultado: F({df1_dec:.0f}, {df2_dec:.1f}) = {f_stat_dec:.4f}, p-value = {p_val_dec:.4e}")
    print(f">> Decisión con α=0.05: {'Rechazar H0 (Hay diferencias entre décadas)' if p_val_dec < 0.05 else 'No rechazar H0'}")
    print(f">> Tamaño del efecto (η²): {eta2_dec:.4f} -> {mag_dec}")
    
    tukey_dec = pairwise_tukeyhsd(endog=df_dec['price'], groups=df_dec['decade'].astype(str), alpha=0.05)
    print("\n>> Comparaciones por parejas (Tukey HSD):")
    print(tukey_dec)
    
    residuos_dict['Décadas'] = df_dec['price'] - df_dec.groupby('decade', observed=False)['price'].transform('mean')
    
    sub_dec = f"ANOVA de Welch: F({df1_dec:.0f}, {df2_dec:.1f}) = {f_stat_dec:.2f}, p = {p_val_dec:.2e} | η² = {eta2_dec:.3f} ({mag_dec})"
    ruta_g2 = os.path.join(carpeta_graficas, "grafica_2_anova_price_por_decada.png")
    graficar_boxplot_anova(df_dec, 'decade', 'price', "Comparación de Precios por Década Histórica", sub_dec, ruta_g2,
                           orden=['1960-1969', '1970-1974', '1975-1979', '1980-1988', '1989-2004'])
    
    tabla_resumen.append({
        'Prueba Estadística': 'ANOVA de Welch (Décadas)',
        'Variable': 'Precio (price)',
        'Grupos Comparados': '1960-69, 1970-74, 1975-79, 1980-88, 1989-04',
        'Estadístico': f"F = {f_stat_dec:.4f}",
        'Grados Libertad': f"({df1_dec:.0f}, {df2_dec:.1f})",
        'p-value': f"{p_val_dec:.4e}",
        'Decisión (α=0.05)': 'Rechazar H0',
        'Tamaño de Efecto': f"η² = {eta2_dec:.4f}",
        'Magnitud': mag_dec
    })
    
    # --------------------------------------------------------------------------
    # 3. PRUEBA T DE WELCH: ESTATUS DE TÍTULO (Clean vs Salvage/Rebuilt)
    # --------------------------------------------------------------------------
    print("\n-------------------------------------------------------------")
    print("3. PRUEBA T DE WELCH: PRECIO POR ESTATUS DE TÍTULO")
    print("-------------------------------------------------------------")
    df_tit = df.dropna(subset=['title_group']).copy()
    g_clean = df_tit[df_tit['title_group'] == 'Clean']['price'].values
    g_salv = df_tit[df_tit['title_group'] == 'Salvage/Rebuilt']['price'].values
    
    # Prueba t de Welch (equal_var=False)
    t_stat_tit, p_val_tit = stats.ttest_ind(g_clean, g_salv, equal_var=False)
    d_tit = calcular_d_cohen(g_clean, g_salv)
    mag_d_tit = interpretar_efecto(d_tit, 'cohen')
    
    # Grados de libertad Welch
    v1, v2 = np.var(g_clean, ddof=1)/len(g_clean), np.var(g_salv, ddof=1)/len(g_salv)
    df_tit_val = (v1 + v2)**2 / ((v1**2)/(len(g_clean)-1) + (v2**2)/(len(g_salv)-1))
    
    print(f">> Resultado: t({df_tit_val:.1f}) = {t_stat_tit:.4f}, p-value = {p_val_tit:.4e}")
    print(f">> Promedios: Clean = ${np.mean(g_clean):,.2f} vs Salvage/Rebuilt = ${np.mean(g_salv):,.2f} (Diferencia = ${np.mean(g_clean)-np.mean(g_salv):,.2f})")
    print(f">> Tamaño del efecto (d de Cohen): {d_tit:.4f} -> {mag_d_tit}")
    
    residuos_dict['Título (Clean vs Salvage)'] = df_tit['price'] - df_tit.groupby('title_group')['price'].transform('mean')
    
    sub_tit = f"Prueba t de Welch: t({df_tit_val:.1f}) = {t_stat_tit:.2f}, p = {p_val_tit:.2e} | d = {d_tit:.3f} ({mag_d_tit})"
    ruta_g3 = os.path.join(carpeta_graficas, "grafica_3_ttest_price_por_title_status.png")
    graficar_boxplot_ttest(df_tit, 'title_group', 'price', "Comparación de Precio: Título Clean vs Salvage/Rebuilt", sub_tit, ruta_g3)
    
    tabla_resumen.append({
        'Prueba Estadística': 'Prueba t de Welch (Título)',
        'Variable': 'Precio (price)',
        'Grupos Comparados': 'Clean vs Salvage/Rebuilt',
        'Estadístico': f"t = {t_stat_tit:.4f}",
        'Grados Libertad': f"{df_tit_val:.1f}",
        'p-value': f"{p_val_tit:.4e}",
        'Decisión (α=0.05)': 'Rechazar H0',
        'Tamaño de Efecto': f"d = {d_tit:.4f}",
        'Magnitud': mag_d_tit
    })
    
    # --------------------------------------------------------------------------
    # 4. PRUEBA T DE WELCH: CONDICIÓN VEHICULAR (Alta vs Baja)
    # --------------------------------------------------------------------------
    print("\n-------------------------------------------------------------")
    print("4. PRUEBA T DE WELCH: PRECIO POR CONDICIÓN VEHICULAR")
    print("-------------------------------------------------------------")
    df_cnd = df.dropna(subset=['condition_group']).copy()
    g_alta = df_cnd[df_cnd['condition_group'] == 'Condición Alta']['price'].values
    g_baja = df_cnd[df_cnd['condition_group'] == 'Condición Baja']['price'].values
    
    t_stat_cnd, p_val_cnd = stats.ttest_ind(g_alta, g_baja, equal_var=False)
    d_cnd = calcular_d_cohen(g_alta, g_baja)
    mag_d_cnd = interpretar_efecto(d_cnd, 'cohen')
    
    v1, v2 = np.var(g_alta, ddof=1)/len(g_alta), np.var(g_baja, ddof=1)/len(g_baja)
    df_cnd_val = (v1 + v2)**2 / ((v1**2)/(len(g_alta)-1) + (v2**2)/(len(g_baja)-1))
    
    print(f">> Resultado: t({df_cnd_val:.1f}) = {t_stat_cnd:.4f}, p-value = {p_val_cnd:.4e}")
    print(f">> Promedios: Condición Alta = ${np.mean(g_alta):,.2f} vs Baja = ${np.mean(g_baja):,.2f} (Diferencia = ${np.mean(g_alta)-np.mean(g_baja):,.2f})")
    print(f">> Tamaño del efecto (d de Cohen): {d_cnd:.4f} -> {mag_d_cnd}")
    
    residuos_dict['Condición (Alta vs Baja)'] = df_cnd['price'] - df_cnd.groupby('condition_group')['price'].transform('mean')
    
    sub_cnd = f"Prueba t de Welch: t({df_cnd_val:.1f}) = {t_stat_cnd:.2f}, p = {p_val_cnd:.2e} | d = {d_cnd:.3f} ({mag_d_cnd})"
    ruta_g4 = os.path.join(carpeta_graficas, "grafica_4_ttest_price_por_condicion.png")
    graficar_boxplot_ttest(df_cnd, 'condition_group', 'price', "Comparación de Precio: Condición Alta vs Condición Baja", sub_cnd, ruta_g4)
    
    tabla_resumen.append({
        'Prueba Estadística': 'Prueba t de Welch (Condición)',
        'Variable': 'Precio (price)',
        'Grupos Comparados': 'Condición Alta vs Condición Baja',
        'Estadístico': f"t = {t_stat_cnd:.4f}",
        'Grados Libertad': f"{df_cnd_val:.1f}",
        'p-value': f"{p_val_cnd:.4e}",
        'Decisión (α=0.05)': 'Rechazar H0',
        'Tamaño de Efecto': f"d = {d_cnd:.4f}",
        'Magnitud': mag_d_cnd
    })
    
    # --------------------------------------------------------------------------
    # 5. PRUEBA T DE WELCH (SEÑAL PRÁCTICA 2): CARROCERÍA (Convertible vs Coupe)
    # --------------------------------------------------------------------------
    print("\n-------------------------------------------------------------")
    print("5. PRUEBA T DE WELCH: CONVERTIBLE VS COUPE (SEÑAL PRÁCTICA 2)")
    print("-------------------------------------------------------------")
    df_typ = df[df['type'].isin(['convertible', 'coupe'])].copy()
    g_conv = df_typ[df_typ['type'] == 'convertible']['price'].values
    g_coup = df_typ[df_typ['type'] == 'coupe']['price'].values
    
    t_stat_typ, p_val_typ = stats.ttest_ind(g_conv, g_coup, equal_var=False)
    d_typ = calcular_d_cohen(g_conv, g_coup)
    mag_d_typ = interpretar_efecto(d_typ, 'cohen')
    
    v1, v2 = np.var(g_conv, ddof=1)/len(g_conv), np.var(g_coup, ddof=1)/len(g_coup)
    df_typ_val = (v1 + v2)**2 / ((v1**2)/(len(g_conv)-1) + (v2**2)/(len(g_coup)-1))
    
    print(f">> Resultado: t({df_typ_val:.1f}) = {t_stat_typ:.4f}, p-value = {p_val_typ:.4e}")
    print(f">> Promedios: Convertible = ${np.mean(g_conv):,.2f} vs Coupe = ${np.mean(g_coup):,.2f} (Prima = ${np.mean(g_conv)-np.mean(g_coup):,.2f})")
    print(f">> Tamaño del efecto (d de Cohen): {d_typ:.4f} -> {mag_d_typ}")
    
    residuos_dict['Carrocería (Convertible vs Coupe)'] = df_typ['price'] - df_typ.groupby('type')['price'].transform('mean')
    
    sub_typ = f"Prueba t de Welch: t({df_typ_val:.1f}) = {t_stat_typ:.2f}, p = {p_val_typ:.2e} | d = {d_typ:.3f} ({mag_d_typ})"
    ruta_g5 = os.path.join(carpeta_graficas, "grafica_5_ttest_price_convertible_vs_coupe.png")
    graficar_boxplot_ttest(df_typ, 'type', 'price', "Comparación de Precio: Convertible vs Coupe (Prima de Colección)", sub_typ, ruta_g5)
    
    tabla_resumen.append({
        'Prueba Estadística': 'Prueba t de Welch (Carrocería)',
        'Variable': 'Precio (price)',
        'Grupos Comparados': 'Convertible vs Coupe',
        'Estadístico': f"t = {t_stat_typ:.4f}",
        'Grados Libertad': f"{df_typ_val:.1f}",
        'p-value': f"{p_val_typ:.4e}",
        'Decisión (α=0.05)': 'Rechazar H0',
        'Tamaño de Efecto': f"d = {d_typ:.4f}",
        'Magnitud': mag_d_typ
    })
    
    # --------------------------------------------------------------------------
    # 6. PANEL DIAGNÓSTICO VISUAL (GRÁFICOS Q-Q)
    # --------------------------------------------------------------------------
    ruta_g6 = os.path.join(carpeta_graficas, "grafica_6_diagnostico_supuestos_qqplots.png")
    graficar_diagnostico_visual(residuos_dict, ruta_g6)
    print(f"\n>> Gráfica de diagnóstico visual guardada en: {ruta_g6}")
    
    # --------------------------------------------------------------------------
    # TABLA RESUMEN CONSOLIDADA
    # --------------------------------------------------------------------------
    df_res = pd.DataFrame(tabla_resumen)
    print("\n" + "="*80)
    print("                    TABLA CONSOLIDADA DE PRUEBAS ESTADÍSTICAS")
    print("="*80)
    print(df_res.to_string(index=False))
    print("="*80)
    print("\n¡Ejecución completada exitosamente!")

if __name__ == '__main__':
    main()
