import pandas as pd
import numpy as np
import os

def main():
    # -------------------------------------------------------------------------
    # 1. CARGA DEL DATASET ORIGINAL
    # -------------------------------------------------------------------------
    raw_path = 'Dataset Original/vehicles.csv'
    output_dir = 'Practica 1 - Limpieza de Datos'
    output_clean_csv = os.path.join(output_dir, 'muscle_cars_clean.csv')

    print(">>> [1/7] Cargando dataset original...")
    df_raw = pd.read_csv(raw_path)
    total_raw_rows = len(df_raw)
    print(f"    Filas totales originales: {total_raw_rows:,}")
    print(f"    Columnas originales: {df_raw.shape[1]}")

    # -------------------------------------------------------------------------
    # 2. FILTRADO A MUSCLE CARS (OPCIÓN A: 1960 - 2004)
    # Razón técnica:
    # Craigslist clasifica 'manufacturer' de forma restringida (omitiendo marcas
    # como Plymouth, Oldsmobile, AMC en el menú desplegable). Por tanto, se usa
    # una taxonomía amplia de expresiones regulares sobre 'model' y 'manufacturer'
    # en el rango temporal 1960-2004 para abarcar era dorada, era de transición
    # (Fox-body, 3ra gen F-body, Grand National) y cuarta generación (SN95, etc.),
    # garantizando una muestra robusta superior a las 5,000 observaciones.
    # -------------------------------------------------------------------------
    print("\n>>> [2/7] Aplicando filtrado temático a Muscle Cars (1960 - 2004)...")
    
    muscle_keywords = [
        'mustang', 'camaro', 'corvette', 'chevelle', 'charger', 'challenger', 'gto', 'firebird',
        'trans am', 'road runner', 'barracuda', 'cuda', 'cutlass', '442', 'grand national', 'javelin',
        'amx', 'torino', 'cougar', 'nova', 'monza', 'dart', 'demon', 'super bee', 'fury', 'bel air',
        'impala', 'monte carlo', 'el camino', 'riviera', 'skylark', 'grand prix', 'galaxie', 'thunderbird',
        'plymouth', 'oldsmobile', 'amc', 'lemans', 'tempest', 'corvair', 'cyclone', 'fairlane'
    ]
    pattern = '|'.join(muscle_keywords)

    # Filtrar por año y modelo
    mask_year = (df_raw['year'] >= 1960) & (df_raw['year'] <= 2004)
    mask_model = df_raw['model'].str.contains(pattern, case=False, na=False)
    
    df = df_raw[mask_year & mask_model].copy()
    total_filtered_rows = len(df)
    print(f"    Filas obtenidas tras filtrado: {total_filtered_rows:,}")

    # Liberar memoria del dataframe original crudo
    del df_raw

    # -------------------------------------------------------------------------
    # 3. ELIMINACIÓN Y TRANSFORMACIÓN DE COLUMNAS NO INFORMATIVAS / METADATOS
    # - county: 100% nula.
    # - url, image_url, region_url: identificadores web de Craigslist sin valor analítico.
    # - fuel: el segmento es 100% gasolina; no aporta varianza predictiva.
    # - VIN: se transforma a indicador booleano 'is_vin_missing' antes de descartar el hash.
    # -------------------------------------------------------------------------
    print("\n>>> [3/7] Transformación y eliminación de columnas...")
    df['is_vin_missing'] = df['VIN'].isnull().astype(int)
    
    cols_to_drop = ['county', 'url', 'image_url', 'region_url', 'fuel', 'VIN']
    df.drop(columns=cols_to_drop, inplace=True, errors='ignore')
    print(f"    Columnas eliminadas: {cols_to_drop}")
    print(f"    Nueva variable creada: 'is_vin_missing'")

    # -------------------------------------------------------------------------
    # 4. ELIMINACIÓN DE REGISTROS DUPLICADOS POR CONTENIDO
    # -------------------------------------------------------------------------
    print("\n>>> [4/7] Verificando y eliminando duplicados...")
    cols_check = [c for c in df.columns if c not in ['id']]
    dups_count = df.duplicated(subset=cols_check).sum()
    print(f"    Duplicados exactos encontrados: {dups_count}")
    if dups_count > 0:
        df.drop_duplicates(subset=cols_check, keep='first', inplace=True)

    # -------------------------------------------------------------------------
    # 5. TRATAMIENTO DE OUTLIERS Y VALORES FALTANTES
    # -------------------------------------------------------------------------
    print("\n>>> [5/7] Tratando nulos y valores extremos en variables cuantitativas y cualitativas...")

    # A. PRECIO (price - Variable Objetivo)
    # Diagnóstico técnico: No existen nulos reales (NaN = 0), pero existen 208 registros
    # con error evidente de captura (price < $500, incluyendo $0/$1 o price > $500,000).
    # Dado que tras descartar estos 208 registros aún se conservan
    # 5,703 filas, se opta por eliminar estos registros
    invalid_price = (df['price'] < 500) | (df['price'] > 500000)
    print(f"    Registros con precio anómalo / error de captura (< $500 o > $500k): {invalid_price.sum():,}")
    df = df[~invalid_price].copy()
    print(f"    Filas tras depuración de outliers en price: {len(df):,}")

    # B. ODÓMETRO (odometer)
    # Valores <= 0 o > 500,000 millas se consideran faltantes/anómalos.
    # Se crea el indicador booleano 'odometer_is_missing' antes de imputar la mediana por año.
    invalid_odo = df['odometer'].isnull() | (df['odometer'] <= 0) | (df['odometer'] > 500000)
    df['odometer_is_missing'] = invalid_odo.astype(int)
    print(f"    Registros de odómetro faltantes o anómalos: {invalid_odo.sum():,}")
    df.loc[invalid_odo, 'odometer'] = np.nan
    
    # Imputación con mediana por año
    group_median_odo = df.groupby('year')['odometer'].transform('median')
    df['odometer'] = df['odometer'].fillna(group_median_odo)
    df['odometer'] = df['odometer'].fillna(df['odometer'].median())

    # C. CILINDROS (cylinders)
    # Se estandariza el texto. Para los nulos, se infiere mediante la moda del modelo
    # o bien asignando la configuración clásica de 8 cilindros predominante en muscle cars.
    print("    Imputando 'cylinders' por moda del modelo / motorización...")
    model_mode_cyl = df.groupby('model')['cylinders'].apply(
        lambda x: x.mode().iloc[0] if not x.mode().empty else '8 cylinders'
    )
    df['cylinders'] = df['cylinders'].fillna(df['model'].map(model_mode_cyl))
    df['cylinders'] = df['cylinders'].fillna('8 cylinders')

    # D. VARIABLES CATEGÓRICAS (condition, title_status, transmission, drive, type, paint_color, size)
    # Se asigna la categoría explícita 'unknown' para conservar todas las observaciones sin introducir sesgo.
    cat_cols_unknown = ['condition', 'title_status', 'transmission', 'drive', 'type', 'paint_color', 'manufacturer', 'size']
    for col in cat_cols_unknown:
        df[col] = df[col].fillna('unknown')

    # D2. COORDENADAS (lat, long)
    # Imputación por mediana espacial agrupada por estado (state)
    df['lat'] = df['lat'].fillna(df.groupby('state')['lat'].transform('median'))
    df['long'] = df['long'].fillna(df.groupby('state')['long'].transform('median'))
    df['lat'] = df['lat'].fillna(df['lat'].median())
    df['long'] = df['long'].fillna(df['long'].median())

    # E. DESCRIPCIÓN (description)
    # Se reemplaza NaN por cadena vacía "" para análisis posterior de procesamiento de lenguaje natural (NLP).
    df['description'] = df['description'].fillna('')

    # F. FECHA DE PUBLICACIÓN (posting_date)
    # Conversión a formato datetime64 estandarizado con zona horaria UTC.
    df['posting_date'] = pd.to_datetime(df['posting_date'], errors='coerce', utc=True)
    # Para los pocos nulos (si existiesen), se imputa con la mediana temporal
    if df['posting_date'].isnull().sum() > 0:
        median_date = df['posting_date'].dropna().median()
        df['posting_date'] = df['posting_date'].fillna(median_date)

    # -------------------------------------------------------------------------
    # 6. CONVERSIÓN FINAL DE TIPOS DE DATOS (CASTING)
    # -------------------------------------------------------------------------
    print("\n>>> [6/7] Ajustando tipos de datos finales...")
    df['year'] = df['year'].astype(int)
    df['price'] = df['price'].round(2)
    df['odometer'] = df['odometer'].round(1)

    # -------------------------------------------------------------------------
    # 7. GUARDADO DEL DATASET LIMPIO Y RESUMEN TÉCNICO
    # -------------------------------------------------------------------------
    print(f"\n>>> [7/7] Guardando dataset procesado en: {output_clean_csv}...")
    df.to_csv(output_clean_csv, index=False)
    
    print("\n" + "="*80)
    print("RESUMEN TÉCNICO DE EJECUCIÓN - PRÁCTICA 1")
    print("="*80)
    print(f"Filas iniciales (Dataset Crudo):           {total_raw_rows:,}")
    print(f"Filas tras filtrado (Muscle Cars 60-04):   {total_filtered_rows:,}")
    print(f"Filas finales limpias:                     {len(df):,}")
    print(f"Total de columnas resultantes:             {df.shape[1]}")
    print("\nPorcentaje de valores nulos por columna:")
    null_summary = (df.isnull().sum() / len(df) * 100).round(2)
    print(null_summary.to_string())
    print("\nRango temporal cubierto por posting_date:")
    print(f"Fecha mínima: {df['posting_date'].min()}")
    print(f"Fecha máxima: {df['posting_date'].max()}")
    print(f"Total de días cubiertos: {(df['posting_date'].max() - df['posting_date'].min()).days} días")
    print("="*80)

if __name__ == '__main__':
    main()
