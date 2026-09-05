import pandas as pd
import numpy as np
import os

# Cargar dataset limpio
def cargar_datos(ruta_csv: str) -> pd.DataFrame:
    if not os.path.exists(ruta_csv):
        raise FileNotFoundError(f"No se encontró el archivo: {ruta_csv}")
    return pd.read_csv(ruta_csv)

def calcular_precio_por_fabricante(df: pd.DataFrame) -> pd.DataFrame:
    """
    Agrupa por 'manufacturer' y calcula estadísticas de precio:
    - n (conteo)
    - media
    - mediana
    - desviación estándar
    - Q1 (25%) y Q3 (75%)
    - Rango Intercuartílico (IQR)
    """
    grouped = df.groupby('manufacturer')['price'].agg(
        n='count',
        media='mean',
        mediana='median',
        std='std',
        q25=lambda x: x.quantile(0.25),
        q75=lambda x: x.quantile(0.75)
    )
    grouped['IQR'] = grouped['q75'] - grouped['q25']
    grouped = grouped.sort_values(by='n', ascending=False)
    return grouped

# Precio agrupado por modelo para los top_k modelos con más anuncios
def calcular_precio_por_modelo(df: pd.DataFrame, top_k: int = 20) -> pd.DataFrame:
    top_models = df['model'].value_counts().head(top_k).index
    df_top = df[df['model'].isin(top_models)]
    
    grouped = df_top.groupby('model')['price'].agg(
        n='count',
        media='mean',
        mediana='median',
        std='std',
        q25=lambda x: x.quantile(0.25),
        q75=lambda x: x.quantile(0.75)
    )
    grouped['IQR'] = grouped['q75'] - grouped['q25']
    grouped = grouped.sort_values(by='n', ascending=False)
    return grouped

def calcular_precio_por_decada(df: pd.DataFrame) -> pd.DataFrame:
    """
    Crea la variable ordinal derivada 'decade' y calcula métricas de precio.
    Intervalos de segmentación histórica:
    - 1960 - 1969: Era Dorada inicial de Muscle Cars y Pony Cars
    - 1970 - 1974: Clásicos de alta compresión previa a regulaciones de emisiones
    - 1975 - 1979: Era de transición y reajuste de potencia
    - 1980 - 1988: Resurgimiento ochentero (Fox-Body Mustang, F-Body 3ra gen, GNX)
    - 1989 - 2004: Cuarta generación / SN95 y evolución pre-retro
    """
    bins = [1959, 1969, 1974, 1979, 1988, 2004]
    labels = ['1960-1969', '1970-1974', '1975-1979', '1980-1988', '1989-2004']
    df_copy = df.copy()
    df_copy['decade'] = pd.cut(df_copy['year'], bins=bins, labels=labels, right=True)
    
    grouped = df_copy.groupby('decade', observed=False)['price'].agg(
        n='count',
        media='mean',
        mediana='median',
        std='std',
        q25=lambda x: x.quantile(0.25),
        q75=lambda x: x.quantile(0.75)
    )
    grouped['IQR'] = grouped['q75'] - grouped['q25']
    return grouped

# Odómetro agrupado por condición
def calcular_odometro_por_condicion(df: pd.DataFrame) -> pd.DataFrame:
    grouped = df.groupby('condition')['odometer'].agg(
        n='count',
        media='mean',
        mediana='median',
        std='std',
        q25=lambda x: x.quantile(0.25),
        q75=lambda x: x.quantile(0.75)
    )
    grouped['IQR'] = grouped['q75'] - grouped['q25']
    grouped = grouped.sort_values(by='mediana', ascending=False)
    return grouped

# Distribución geográfica por estado y región
def calcular_distribucion_geografica(df: pd.DataFrame, top_k: int = 15) -> tuple:
    # Agrupación por Estado
    state_counts = df['state'].value_counts()
    state_table = pd.DataFrame({
        'n': state_counts,
        'porcentaje (%)': (state_counts / len(df) * 100).round(2)
    })
    
    # Agrupación por Región
    region_counts = df['region'].value_counts()
    region_table = pd.DataFrame({
        'n': region_counts,
        'porcentaje (%)': (region_counts / len(df) * 100).round(2)
    })
    
    return state_table.head(top_k), region_table.head(top_k)

def main():
    ruta_dataset = os.path.join('Practica 1 - Limpieza de Datos', 'muscle_cars_clean.csv')
    print("="*80)
    print("PROCESANDO MÉTRICAS DE DATOS AGRUPADOS (PRÁCTICA 2)")
    print("="*80)
    
    df = cargar_datos(ruta_dataset)
    print(f"Dataset cargado: {df.shape[0]:,} registros.\n")
    
    # 1. PRECIO POR FABRICANTE
    print(">>> 1. PRECIO AGRUPADO POR FABRICANTE (MANUFACTURER):")
    tabla_mfg = calcular_precio_por_fabricante(df)
    print(tabla_mfg.round(2).to_string())
    print("\n" + "-"*80 + "\n")
    
    # 2. PRECIO POR MODELO (TOP 20)
    print(">>> 2. PRECIO AGRUPADO POR MODELO (TOP 20 MODELOS CON MÁS ANUNCIOS):")
    tabla_modelo = calcular_precio_por_modelo(df, top_k=20)
    print(tabla_modelo.round(2).to_string())
    print("\n" + "-"*80 + "\n")
    
    # 3. PRECIO POR DÉCADA / ERA
    print(">>> 3. PRECIO AGRUPADO POR DÉCADA DE FABRICACIÓN (DECADE):")
    tabla_decada = calcular_precio_por_decada(df)
    print(tabla_decada.round(2).to_string())
    print("\n" + "-"*80 + "\n")
    
    # 4. ODÓMETRO POR CONDICIÓN
    print(">>> 4. ODÓMETRO (MILLAS) AGRUPADO POR CONDICIÓN (CONDITION):")
    tabla_cond = calcular_odometro_por_condicion(df)
    print(tabla_cond.round(2).to_string())
    print("\n" + "-"*80 + "\n")
    
    # 5. CONCENTRACIÓN GEOGRÁFICA
    print(">>> 5. CONCENTRACIÓN GEOGRÁFICA (TOP 15 ESTADOS Y REGIONES):")
    tabla_state, tabla_region = calcular_distribucion_geografica(df, top_k=15)
    print("[Top 15 Estados]:")
    print(tabla_state.to_string())
    print("\n[Top 15 Regiones de Craigslist]:")
    print(tabla_region.to_string())
    
    print("\n" + "="*80)
    print("PROCESAMIENTO DE MÉTRICAS AGRUPADAS COMPLETADO CON ÉXITO.")
    print("="*80)

if __name__ == '__main__':
    main()
