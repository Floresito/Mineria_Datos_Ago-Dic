import pandas as pd
import numpy as np

path = 'Dataset Original/vehicles.csv'
df = pd.read_csv(path)

print('=== FORMA DEL DATASET ===')
print(f'Total de filas: {df.shape[0]:,}')
print(f'Total de columnas: {df.shape[1]}')
print(f'Memoria en RAM: {df.memory_usage(deep=True).sum() / (1024**2):.2f} MB')

print('\n=== TIPOS DE DATOS Y VALORES NULOS ===')
null_df = pd.DataFrame({
    'Tipo_Dato': df.dtypes,
    'Valores_No_Nulos': df.notnull().sum(),
    'Valores_Nulos': df.isnull().sum(),
    'Pct_Nulos': (df.isnull().sum() / len(df) * 100).round(2)
})
print(null_df.to_string())

print('\n=== DUPLICADOS ===')
dups_id = df.duplicated(subset=['id']).sum()
print(f'Duplicados por ID: {dups_id}')

cols_content = [c for c in df.columns if c not in ['id', 'url', 'region_url', 'image_url']]
dups_content = df.duplicated(subset=cols_content).sum()
print(f'Duplicados por contenido (excluyendo IDs y URLs): {dups_content}')

print('\n=== ESTADÍSTICAS BÁSICAS (NUMÉRICAS) ===')
num_cols = ['price', 'year', 'odometer']
print(df[num_cols].describe(percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]).to_string())

print('\n=== ANÁLISIS DE POSTING_DATE ===')
print('Nulos en posting_date:', df['posting_date'].isnull().sum())
sample_dates = df['posting_date'].dropna().head(5).tolist()
print('Muestra de posting_date:', sample_dates)
posting_dt = pd.to_datetime(df['posting_date'], errors='coerce', utc=True)
print('Fecha mínima:', posting_dt.min())
print('Fecha máxima:', posting_dt.max())
print('Meses/Años representados:')
print(posting_dt.dt.to_period('M').value_counts().sort_index())

del df

### =============================================================================

df = pd.read_csv('Dataset Original/vehicles.csv', usecols=['manufacturer', 'year', 'model'])
print('=== FABRICANTES EN EL DATASET ===')
print(df['manufacturer'].value_counts(dropna=False))

print('\n=== FILTRO PREVIO: MARCAS MUSCLE Y AÑOS 1964-1988 ===')
brands = ['chevrolet', 'pontiac', 'ford', 'dodge', 'plymouth', 'amc', 'buick', 'oldsmobile', 'mercury']
df_filtered = df[df['manufacturer'].str.lower().isin(brands) & (df['year'] >= 1964) & (df['year'] <= 1988)]
print(f'Total de filas con marca y año (1964-1988): {len(df_filtered):,}')
print('\nDistribución por marca en este rango de años:')
print(df_filtered['manufacturer'].value_counts())

del df
del df_filtered

### =============================================================================

df = pd.read_csv('Dataset Original/vehicles.csv', usecols=['manufacturer', 'year', 'model', 'description'])

# Revisar si hay Plymouth, Oldsmobile, AMC en model o description
print('Modelos que contienen Plymouth:', df['model'].str.contains('plymouth|barracuda|road runner|gtx|duster', case=False, na=False).sum())
print('Modelos que contienen Oldsmobile o Cutlass o 442:', df['model'].str.contains('oldsmobile|cutlass|442', case=False, na=False).sum())
print('Modelos que contienen AMC o Javelin:', df['model'].str.contains('amc|javelin|amx', case=False, na=False).sum())
print('Modelos que contienen Corvette, Camaro, Mustang, Chevelle, Charger, Challenger, GTO, Firebird, Nova, Impala, Monte Carlo, Cougar, Skylark, LeSabre, Trans Am:')
muscle_pattern = 'mustang|camaro|corvette|chevelle|charger|challenger|gto|firebird|trans am|road runner|barracuda|cutlass|442|grand national|javelin|amx|torino|cougar|nova|monza|dart|demon|super bee|fury|bel air|impala|monte carlo|el camino'
match_model = df['model'].str.contains(muscle_pattern, case=False, na=False)
print('Total matches en model (todos los años):', match_model.sum())

print('Matches en model (1964-1988):', (match_model & (df['year'] >= 1964) & (df['year'] <= 1988)).sum())
print('Matches en model (1960-1995):', (match_model & (df['year'] >= 1960) & (df['year'] <= 1995)).sum())
print('Matches en model (1960-2004):', (match_model & (df['year'] >= 1960) & (df['year'] <= 2004)).sum())
print('Matches en model (1964-2022):', (match_model & (df['year'] >= 1964)).sum())

del df

### ==============================================================================

df = pd.read_csv('Dataset Original/vehicles.csv', usecols=['manufacturer', 'year', 'model'])

# Conteo por marca amplia
brands = ['chevrolet', 'ford', 'dodge', 'pontiac', 'buick', 'mercury', 'cadillac', 'chrysler', 'lincoln']

print('--- RANGOS DE AÑOS PARA MARCAS AMERICANAS ---')
for y_start, y_end in [(1964, 1988), (1960, 1993), (1960, 2004), (1964, 2021)]:
    sub = df[df['manufacturer'].isin(brands) & (df['year'] >= y_start) & (df['year'] <= y_end)]
    print(f'Marcas americanas ({y_start}-{y_end}) sin filtro de modelo: {len(sub):,} filas')

muscle_keywords = [
    'mustang', 'camaro', 'corvette', 'chevelle', 'charger', 'challenger', 'gto', 'firebird',
    'trans am', 'road runner', 'barracuda', 'cuda', 'cutlass', '442', 'grand national', 'javelin',
    'amx', 'torino', 'cougar', 'nova', 'monza', 'dart', 'demon', 'super bee', 'fury', 'bel air',
    'impala', 'monte carlo', 'el camino', 'riviera', 'skylark', 'grand prix', 'galaxie', 'thunderbird',
    'plymouth', 'oldsmobile', 'amc', 'lemans', 'tempest', 'corvair', 'cyclone', 'fairlane'
]
pattern = '|'.join(muscle_keywords)

print('\n--- FILTRO CON PALABRAS CLAVE DE MODELO/MARCA MUSCLE ---')
for y_start, y_end in [(1964, 1988), (1960, 1993), (1960, 2004), (1960, 2014), (1964, 2021)]:
    sub = df[df['model'].str.contains(pattern, case=False, na=False) & (df['year'] >= y_start) & (df['year'] <= y_end)]
    print(f'Modelos/Líneas Muscle ({y_start}-{y_end}): {len(sub):,} filas')