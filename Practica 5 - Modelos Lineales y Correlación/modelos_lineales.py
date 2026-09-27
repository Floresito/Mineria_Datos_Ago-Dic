import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.outliers_influence import variance_inflation_factor
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

# Cargar y preparar datos
def cargar_y_preparar_datos(ruta_csv: str) -> pd.DataFrame:
    if not os.path.exists(ruta_csv):
        raise FileNotFoundError(f"No se encontró el dataset en: {ruta_csv}")
    
    df = pd.read_csv(ruta_csv)
    
    # 1. Extracción numérica de cilindrada ('8 cylinders' -> 8)
    df['cylinders_num'] = df['cylinders'].str.extract(r'(\d+)').astype(float).fillna(8)
    
    # 2. Variables Dummy fundamentadas en la Práctica 4
    df['is_convertible'] = (df['type'] == 'convertible').astype(int)
    df['is_clean_title'] = (df['title_status'] == 'clean').astype(int)
    df['is_good_condition'] = df['condition'].isin(['excellent', 'like new', 'new', 'good']).astype(int)
    
    # 3. Categoría de Década para segmentaciones
    bins = [1959, 1969, 1974, 1979, 1988, 2004]
    labels = ['1960-1969', '1970-1974', '1975-1979', '1980-1988', '1989-2004']
    df['decade'] = pd.cut(df['year'], bins=bins, labels=labels, right=True)
    
    return df

def crear_directorio_graficas(carpeta_salida: str) -> None:
    os.makedirs(carpeta_salida, exist_ok=True)

# 1. MATRIZ DE CORRELACIÓN Y HEATMAP
# Calcula la matriz de correlación lineal de Pearson entre las variables numéricas
# y genera un mapa de calor visual con anotaciones numéricas.
def calcular_matriz_correlacion(df: pd.DataFrame, carpeta_salida: str) -> pd.DataFrame:
    num_cols = ['price', 'year', 'odometer', 'cylinders_num']
    corr_matrix = df[num_cols].corr(method='pearson')
    
    plt.figure(figsize=(8, 6))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    
    sns.heatmap(
        corr_matrix, 
        annot=True, 
        fmt=".3f", 
        cmap="coolwarm", 
        vmin=-0.5, 
        vmax=0.5, 
        linewidths=1, 
        cbar_kws={"label": "Coeficiente de Correlación de Pearson (r)"}
    )
    plt.title("Matriz de Correlación Lineal de Pearson (Variables Numéricas)", pad=15)
    plt.tight_layout()
    
    ruta_guardado = os.path.join(carpeta_salida, "grafica_4_heatmap_correlacion.png")
    plt.savefig(ruta_guardado, dpi=300, bbox_inches='tight')
    plt.close()
    
    return corr_matrix

# 2. AJUSTE Y COMPARACIÓN DE LOS 4 MODELOS CANDIDATOS
# Ajusta los 4 modelos de regresión lineal candidatos mediante MCO (OLS):
# - Modelo A: Lineal Simple (price ~ year)
# - Modelo B: Lineal Múltiple (price ~ year + odometer + dummies)
# - Modelo C: Polinomial Grado 2 (price ~ year + year^2)
# - Modelo D: Múltiple Polinomial (price ~ year + year^2 + odometer + dummies)
def ajustar_modelos_candidatos(df: pd.DataFrame) -> tuple:
    # Modelo A
    mod_a = ols('price ~ year', data=df).fit()
    
    # Modelo B
    mod_b = ols('price ~ year + odometer + is_convertible + is_clean_title + is_good_condition + cylinders_num', data=df).fit()
    
    # Modelo C
    mod_c = ols('price ~ year + I(year**2)', data=df).fit()
    
    # Modelo D (Modelo Oficial Seleccionado)
    mod_d = ols('price ~ year + I(year**2) + odometer + is_convertible + is_clean_title + is_good_condition + cylinders_num', data=df).fit()
    
    # Tabla resumen comparativa
    modelos = {
        'Modelo A (Lineal Simple: year)': mod_a,
        'Modelo B (Múltiple Lineal: year + control)': mod_b,
        'Modelo C (Polinomial Grado 2: year + year^2)': mod_c,
        'Modelo D (Múltiple Polinomial: Oficial)': mod_d
    }
    
    res_list = []
    for name, m in modelos.items():
        res_list.append({
            'Modelo': name,
            'R2': round(m.rsquared, 4),
            'R2_Ajustado': round(m.rsquared_adj, 4),
            'AIC': round(m.aic, 1),
            'BIC': round(m.bic, 1),
            'RMSE ($)': round(np.sqrt(m.mse_resid), 2),
            'F-Statistic': round(m.fvalue, 2),
            'p-value (F)': m.f_pvalue,
            'Num_Parametros': len(m.params)
        })
        
    df_comparativa = pd.DataFrame(res_list)
    return mod_a, mod_b, mod_c, mod_d, df_comparativa

# 3. ANÁLISIS DE MULTICOLINEALIDAD (VIF)
# Calcula el Factor de Inflación de la Varianza (VIF) para las variables
# predictoras del Modelo Múltiple.
# - VIF = 1: Ausencia total de colinealidad.
# - 1 < VIF < 5: Colinealidad baja/moderada (completamente aceptable).
# - VIF > 10: Multicolinealidad severa que infla los errores estándar.
def calcular_vif(df: pd.DataFrame) -> pd.DataFrame:
    pred_cols = ['year', 'odometer', 'is_convertible', 'is_clean_title', 'is_good_condition', 'cylinders_num']
    X = df[pred_cols].copy()
    X = sm.add_constant(X)
    
    vif_data = pd.DataFrame()
    vif_data['Variable'] = X.columns
    vif_data['VIF'] = [variance_inflation_factor(X.values, i) for i in range(len(X.columns))]
    vif_data['VIF'] = vif_data['VIF'].round(3)
    
    return vif_data

# 4. Graficas para comparacion de modelos
# Genera una gráfica de dispersión con la recta de regresión simple superpuesta
# y una gráfica de residuos vs. valores predichos para el Modelo A.
def generar_grafica_paso_1_modelo_a(df: pd.DataFrame, mod_a, carpeta_salida: str) -> tuple:
    # 4.1 Scatter con recta
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x='year', y='price', alpha=0.35, color='steelblue', s=30, label='Vehículos Observados')
    
    # Línea de regresión estimada
    years_grid = np.linspace(df['year'].min(), df['year'].max(), 200)
    pred_price = mod_a.params['Intercept'] + mod_a.params['year'] * years_grid
    plt.plot(years_grid, pred_price, color='crimson', linewidth=2.5, 
             label=f'Ajuste OLS: Price = {mod_a.params["Intercept"]:,.0f} - {abs(mod_a.params["year"]):,.1f}·Year\n(R² = {mod_a.rsquared:.4f})')
    
    plt.title("Modelo A: Regresión Lineal Simple (Precio vs. Año de Fabricación)", pad=12)
    plt.xlabel("Año de Fabricación (Year)")
    plt.ylabel("Precio Ofertado en USD ($)")
    plt.legend(loc='upper left')
    plt.tight_layout()
    ruta_scatter = os.path.join(carpeta_salida, "grafica_1_scatter_regresion_simple.png")
    plt.savefig(ruta_scatter, dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4.2 Residuos vs. Valores Predichos (Modelo A)
    plt.figure(figsize=(10, 6))
    predichos_a = mod_a.fittedvalues
    residuos_a = mod_a.resid
    
    sns.scatterplot(x=predichos_a, y=residuos_a, alpha=0.35, color='darkorange', s=30, label='Residuos Observados')
    plt.axhline(0, color='black', linestyle='--', linewidth=1.5, label='Línea Cero Residual')
    
    # Curva Lowess para evidenciar la no linealidad (forma de U)
    sns.regplot(x=predichos_a, y=residuos_a, scatter=False, lowess=True, color='crimson', 
                label='Tendencia Lowess (Patrón en U)',
                line_kws={'linewidth': 2.5})
    
    plt.title("Modelo A: Diagnóstico de Residuos vs. Valores Predichos (Falta de Linealidad)", pad=12)
    plt.xlabel("Valores Predichos ($\\hat{Y}$ en USD)")
    plt.ylabel("Residuos ($e_i = Y_i - \\hat{Y}_i$)")
    plt.legend(loc='upper right')
    plt.tight_layout()
    ruta_residuos_a = os.path.join(carpeta_salida, "grafica_2_residuos_modelo_a.png")
    plt.savefig(ruta_residuos_a, dpi=300, bbox_inches='tight')
    plt.close()
    
    return ruta_scatter, ruta_residuos_a

# Grafica de residuos vs valores predichos para el modelo D
def generar_grafica_paso_2_modelo_d(df: pd.DataFrame, mod_d, carpeta_salida: str) -> str:
    plt.figure(figsize=(10, 6))
    predichos_d = mod_d.fittedvalues
    residuos_d = mod_d.resid
    
    sns.scatterplot(x=predichos_d, y=residuos_d, alpha=0.35, color='forestgreen', s=30, label='Residuos Observados')
    plt.axhline(0, color='black', linestyle='--', linewidth=1.5, label='Línea Cero Residual')
    
    # Curva Lowess
    sns.regplot(x=predichos_d, y=residuos_d, scatter=False, lowess=True, color='navy', 
                label='Tendencia Lowess (Alineación Mejorada)',
                line_kws={'linewidth': 2.5})
    
    plt.title(f"Modelo D (Oficial): Residuos vs. Valores Predichos (R² Ajustado = {mod_d.rsquared_adj:.4f})", pad=12)
    plt.xlabel("Valores Predichos ($\\hat{Y}$ en USD)")
    plt.ylabel("Residuos ($e_i = Y_i - \\hat{Y}_i$)")
    plt.legend(loc='upper right')
    plt.tight_layout()
    
    ruta_residuos_d = os.path.join(carpeta_salida, "grafica_3_residuos_modelo_d.png")
    plt.savefig(ruta_residuos_d, dpi=300, bbox_inches='tight')
    plt.close()
    
    return ruta_residuos_d

# Gráfica segmentada por décadas (comparación de precios por décadas)
def generar_grafica_segmentada_decadas(df: pd.DataFrame, carpeta_salida: str) -> str:
    plt.figure(figsize=(12, 6))
    
    decadas = ['1960-1969', '1970-1974', '1975-1979', '1980-1988', '1989-2004']
    paleta = sns.color_palette("tab10", len(decadas))
    
    for i, dec in enumerate(decadas):
        sub = df[df['decade'] == dec]
        sns.regplot(
            data=sub, 
            x='year', 
            y='price', 
            scatter_kws={'alpha': 0.15, 's': 20},
            line_kws={'linewidth': 2.2, 'label': f'{dec} (n={len(sub):,})'},
            color=paleta[i]
        )
        
    plt.title("Comparación de Rectas de Regresión Lineal Segmentadas por Década Histórica", pad=12)
    plt.xlabel("Año de Fabricación (Year)")
    plt.ylabel("Precio Ofertado en USD ($)")
    plt.legend(title="Época Histórica", loc='upper right')
    plt.tight_layout()
    
    ruta_guardado = os.path.join(carpeta_salida, "grafica_5_regresion_segmentada_decadas.png")
    plt.savefig(ruta_guardado, dpi=300, bbox_inches='tight')
    plt.close()
    
    return ruta_guardado

def main():
    ruta_dataset = os.path.join('Practica 1 - Limpieza de Datos', 'muscle_cars_clean.csv')
    carpeta_salida = os.path.join('Practica 5 - Modelos Lineales y Correlación', 'graficas')
    
    print("="*80)
    print("EJECUTANDO ANÁLISIS DE MODELOS LINEALES Y CORRELACIÓN (PRÁCTICA 5)")
    print("="*80)
    
    df = cargar_y_preparar_datos(ruta_dataset)
    crear_directorio_graficas(carpeta_salida)
    print(f"Dataset cargado: {len(df):,} observaciones. Carpeta de destino: '{carpeta_salida}'\n")
    
    # 1. Matriz de Correlación
    print(">>> [1/5] Calculando Matriz de Correlación de Pearson y Heatmap...")
    corr_mat = calcular_matriz_correlacion(df, carpeta_salida)
    print(corr_mat.round(3).to_string())
    print("\n" + "-"*80)
    
    # 2. Ajuste de Modelos Candidatos
    print("\n>>> [2/5] Ajustando los 4 Modelos Candidatos y comparando métricas...")
    mod_a, mod_b, mod_c, mod_d, df_comp = ajustar_modelos_candidatos(df)
    print("\nTABLA COMPARATIVA DE RENDIMIENTO:")
    print(df_comp.to_string(index=False))
    print("\n" + "-"*80)
    
    # 3. Factor de Inflación de la Varianza (VIF)
    print("\n>>> [3/5] Evaluando Multicolinealidad (VIF) en predictores del Modelo Múltiple...")
    df_vif = calcular_vif(df)
    print(df_vif.to_string(index=False))
    print("\n" + "-"*80)
    
    # 4. Generación de Gráficas de Narrativa (Paso 1 y Paso 2)
    print("\n>>> [4/5] Generando Gráficas de Diagnóstico de Residuos y Regresión...")
    r1, r2 = generar_grafica_paso_1_modelo_a(df, mod_a, carpeta_salida)
    r3 = generar_grafica_paso_2_modelo_d(df, mod_d, carpeta_salida)
    r5 = generar_grafica_segmentada_decadas(df, carpeta_salida)
    print(f"    Guardado Paso 1A: {r1}")
    print(f"    Guardado Paso 1B: {r2}")
    print(f"    Guardado Paso 2:  {r3}")
    print(f"    Guardado Paso 3:  {r5}")
    print("\n" + "-"*80)
    
    # 5. Coeficientes del Modelo Oficial (Modelo D)
    print("\n>>> [5/5] Coeficientes y Significancia Estadística del Modelo D (Oficial):")
    print(mod_d.summary().tables[1])
    
    print("\n" + "="*80)
    print("EJECUCIÓN COMPLETADA EXITOSAMENTE. TODAS LAS FIGURAS EXPORTADAS A 300 DPI.")
    print("="*80)

if __name__ == '__main__':
    main()
