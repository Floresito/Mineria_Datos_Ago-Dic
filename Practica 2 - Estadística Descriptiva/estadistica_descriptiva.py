import pandas as pd
import numpy as np
import os

# Cargar dataset limpio
def cargar_datos(ruta_csv: str) -> pd.DataFrame:
    if not os.path.exists(ruta_csv):
        raise FileNotFoundError(f"No se encontró el archivo: {ruta_csv}")
    df = pd.read_csv(ruta_csv)
    return df

def calcular_descriptiva_numerica(df: pd.DataFrame, num_cols: list) -> pd.DataFrame:
    """
    Calcula medidas de tendencia central, dispersión, posición y forma para
    las variables cuantitativas continuas y discretas.
    
    Métricas calculadas:
    - n (conteo)
    - Media aritmética
    - Mediana (Q2 / percentil 50)
    - Desviación estándar muestral (std)
    - Mínimo y Máximo
    - Cuartil 1 (Q1 / percentil 25) y Cuartil 3 (Q3 / percentil 75)
    - Rango Intercuartílico (IQR = Q3 - Q1)
    - Coeficiente de Asimetría de Fisher-Pearson (Skewness)
    - Curtosis de Fisher (Kurtosis)
    """
    stats_dict = {}
    
    for col in num_cols:
        series = df[col].dropna()
        q1 = series.quantile(0.25)
        q2 = series.median()
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        
        stats_dict[col] = {
            'count': int(series.count()),
            'mean': series.mean(),
            'median': q2,
            'std': series.std(),
            'min': series.min(),
            'q25 (Q1)': q1,
            'q50 (Q2)': q2,
            'q75 (Q3)': q3,
            'max': series.max(),
            'IQR': iqr,
            'skewness': series.skew(),
            'kurtosis': series.kurtosis()
        }
        
    df_desc = pd.DataFrame(stats_dict).T
    return df_desc

# Distribucion de frecuencias (absolute y relative) y moda
def calcular_descriptiva_categorica(df: pd.DataFrame, cat_cols: list) -> dict:
    res_dict = {}
    
    for col in cat_cols:
        freq_abs = df[col].value_counts(dropna=False)
        freq_rel = df[col].value_counts(normalize=True, dropna=False) * 100
        moda_val = df[col].mode().iloc[0]
        moda_freq = freq_abs.iloc[0]
        
        res_table = pd.DataFrame({
            'Frecuencia_Absoluta (n)': freq_abs,
            'Frecuencia_Relativa (%)': freq_rel.round(2)
        })
        
        res_dict[col] = {
            'tabla_frecuencias': res_table,
            'moda': moda_val,
            'frecuencia_moda': moda_freq,
            'total_categorias': len(freq_abs)
        }
        
    return res_dict

def main():
    ruta_dataset = os.path.join('Practica 1 - Limpieza de Datos', 'muscle_cars_clean.csv')
    print("="*80)
    print("PROCESANDO ESTADÍSTICA DESCRIPTIVA BÁSICA (PRÁCTICA 2)")
    print("="*80)
    
    df = cargar_datos(ruta_dataset)
    print(f"Dataset cargado exitosamente: {df.shape[0]:,} filas × {df.shape[1]} columnas.\n")
    
    # -------------------------------------------------------------------------
    # 1. ESTADÍSTICA DESCRIPTIVA PARA VARIABLES NUMÉRICAS
    # -------------------------------------------------------------------------
    num_cols = ['price', 'year', 'odometer']
    df_num_desc = calcular_descriptiva_numerica(df, num_cols)
    
    print(">>> 1. RESUMEN ESTADÍSTICO DE VARIABLES NUMÉRICAS:")
    print(df_num_desc.round(2).to_string())
    print("\n" + "-"*80 + "\n")
    
    # -------------------------------------------------------------------------
    # 2. ESTADÍSTICA DESCRIPTIVA PARA VARIABLES CATEGÓRICAS
    # -------------------------------------------------------------------------
    cat_cols = [
        'manufacturer', 'model', 'condition', 'cylinders',
        'title_status', 'transmission', 'drive', 'type', 'paint_color'
    ]
    cat_results = calcular_descriptiva_categorica(df, cat_cols)
    
    print(">>> 2. RESUMEN DE MODAS Y FRECUENCIAS DE VARIABLES CATEGÓRICAS:")
    for col in cat_cols:
        info = cat_results[col]
        print(f"\n[Variable: '{col}']")
        print(f"  - Moda: '{info['moda']}' con n = {info['frecuencia_moda']:,} registros ({info['tabla_frecuencias'].iloc[0]['Frecuencia_Relativa (%)']}%)")
        print(f"  - Total de clases/niveles únicos: {info['total_categorias']}")
        print("  - Top 5 categorías con mayor frecuencia:")
        print(info['tabla_frecuencias'].head(5).to_string())
        
    print("\n" + "="*80)
    print("PROCESAMIENTO DESCRIPTIVO COMPLETADO CON ÉXITO.")
    print("="*80)

if __name__ == '__main__':
    main()
