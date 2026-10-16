import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
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

#Carga de datos
def cargar_datos_temporales(ruta_csv: str) -> tuple:
    if not os.path.exists(ruta_csv):
        raise FileNotFoundError(f"No se encontró el dataset en: {ruta_csv}")
    
    df = pd.read_csv(ruta_csv)
    df['posting_date_dt'] = pd.to_datetime(df['posting_date'], utc=True)
    df['date'] = df['posting_date_dt'].dt.date
    df['year_week'] = df['posting_date_dt'].dt.to_period('W').astype(str)
    
    # Serie diaria completa
    daily = df.groupby('date').agg(
        volumen=('price', 'count'),
        precio_medio=('price', 'mean'),
        precio_mediana=('price', 'median'),
        precio_std=('price', 'std')
    ).reset_index()
    daily['date'] = pd.to_datetime(daily['date'])
    
    # Serie semanal
    weekly = df.groupby('year_week').agg(
        volumen=('price', 'count'),
        precio_medio=('price', 'mean'),
        precio_mediana=('price', 'median'),
        precio_std=('price', 'std')
    ).reset_index()
    
    return df, daily, weekly

# Graficas de diagnostico
#     1. Gráfica 1: Serie diaria de Volumen y Precio Medio (con bandas de dispersión).
#     2. Gráfica 2: Agrupación semanal (6 semanas) ilustrando la escasez de períodos macro.
def generar_graficas_diagnostico(daily: pd.DataFrame, weekly: pd.DataFrame, carpeta_salida: str) -> None:

    os.makedirs(carpeta_salida, exist_ok=True)
    
    # --------------------------------------------------------------------------
    # Gráfica 1: Diagnóstico Diario (32 días)
    # --------------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    # Panel 1: Volumen diario
    ax1.plot(daily['date'], daily['volumen'], marker='o', color='darkblue', linewidth=2, label='Volumen Diario de Publicaciones')
    ax1.axvspan(daily['date'].iloc[0], daily['date'].iloc[1], color='gray', alpha=0.15, label='Días Parciales (Borde de Scraping)')
    ax1.axvspan(daily['date'].iloc[-2], daily['date'].iloc[-1], color='gray', alpha=0.15)
    ax1.set_title("A. Volumen Diario de Anuncios Publicados (Posting Volume)", pad=10, fontweight='bold')
    ax1.set_ylabel("Cantidad de Anuncios")
    ax1.legend(loc='upper left', frameon=True)
    ax1.grid(True, linestyle='--', alpha=0.6)
    
    # Anotación del despegue de volumen
    ax1.annotate("Aceleración del scrape\n(acumulación de regiones)", 
                 xy=(daily['date'].iloc[27], daily['volumen'].iloc[27]),
                 xytext=(daily['date'].iloc[15], 420),
                 arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=6),
                 fontsize=10, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="navy"))

    # Panel 2: Precio medio y mediana
    ax2.plot(daily['date'], daily['precio_medio'], marker='s', color='crimson', linewidth=2, label='Precio Promedio Diario (Mean Asking Price)')
    ax2.plot(daily['date'], daily['precio_mediana'], marker='^', color='forestgreen', linestyle='--', linewidth=1.8, label='Mediana Diaria de Precio')
    
    # Banda de dispersión +/- 1 std
    ax2.fill_between(daily['date'], 
                     daily['precio_medio'] - daily['precio_std'], 
                     daily['precio_medio'] + daily['precio_std'], 
                     color='crimson', alpha=0.12, label='Dispersión Diaria (±1 Desv. Estándar)')
    
    ax2.set_title("B. Comportamiento Diario del Precio Pedido (USD)", pad=10, fontweight='bold')
    ax2.set_xlabel("Fecha de Publicación (posting_date)")
    ax2.set_ylabel("Precio en USD")
    ax2.yaxis.set_major_formatter('${x:,.0f}')
    ax2.legend(loc='upper right', frameon=True)
    ax2.grid(True, linestyle='--', alpha=0.6)
    
    fig.suptitle("Diagnóstico Temporal: Rango de 30 Días (04-Abr-2021 a 05-May-2021)", fontsize=14, y=0.98, fontweight='bold')
    plt.tight_layout()
    ruta_g1 = os.path.join(carpeta_salida, "grafica_1_diagnostico_serie_diaria.png")
    plt.savefig(ruta_g1, dpi=300, bbox_inches='tight')
    plt.close()
    
    # --------------------------------------------------------------------------
    # Gráfica 2: Diagnóstico Semanal (6 Semanas Calendario)
    # --------------------------------------------------------------------------
    fig, (ax_w1, ax_w2) = plt.subplots(1, 2, figsize=(13, 5))
    
    # Semanal: Volumen
    bars = ax_w1.bar(range(len(weekly)), weekly['volumen'], color='steelblue', edgecolor='black', alpha=0.85)
    ax_w1.set_title("Volumen de Anuncios por Semana Calendario", pad=12, fontweight='bold')
    ax_w1.set_xlabel("Semana de Publicación")
    ax_w1.set_ylabel("Conteo de Anuncios")
    ax_w1.set_xticks(range(len(weekly)))
    ax_w1.set_xticklabels([f"Sem {i+1}\n({w[:10]})" for i, w in enumerate(weekly['year_week'])], rotation=0)
    for bar in bars:
        height = bar.get_height()
        ax_w1.annotate(f'{height:,}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),  # 3 points vertical offset
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=9, fontweight='bold')
    ax_w1.grid(True, linestyle='--', alpha=0.6)
    
    # Semanal: Precio Promedio y Mediana
    ax_w2.plot(range(len(weekly)), weekly['precio_medio'], marker='o', color='crimson', linewidth=2.2, label='Precio Medio')
    ax_w2.plot(range(len(weekly)), weekly['precio_mediana'], marker='s', color='forestgreen', linestyle='--', linewidth=2, label='Precio Mediana')
    ax_w2.set_title("Precio Promedio y Mediana por Semana", pad=12, fontweight='bold')
    ax_w2.set_xlabel("Semana de Publicación")
    ax_w2.set_ylabel("Precio en USD")
    ax_w2.yaxis.set_major_formatter('${x:,.0f}')
    ax_w2.set_xticks(range(len(weekly)))
    ax_w2.set_xticklabels([f"Sem {i+1}\n({w[:10]})" for i, w in enumerate(weekly['year_week'])], rotation=0)
    ax_w2.legend(loc='lower right', frameon=True)
    ax_w2.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    ruta_g2 = os.path.join(carpeta_salida, "grafica_2_diagnostico_serie_semanal.png")
    plt.savefig(ruta_g2, dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    carpeta_practica = os.path.dirname(os.path.abspath(__file__))
    carpeta_graficas = os.path.join(carpeta_practica, "graficas")
    ruta_csv = os.path.join(carpeta_practica, "..", "Practica 1 - Limpieza de Datos", "muscle_cars_clean.csv")
    
    print("=" * 80)
    print("EJECUTANDO DIAGNÓSTICO DE LA SERIE TEMPORAL (PRÁCTICA 8)")
    print("=" * 80)
    
    df, daily, weekly = cargar_datos_temporales(ruta_csv)
    
    print(f"Total observaciones: {len(df):,} registros.")
    print(f"Fecha mínima: {df['posting_date_dt'].min()}")
    print(f"Fecha máxima: {df['posting_date_dt'].max()}")
    print(f"Amplitud temporal: {(df['posting_date_dt'].max() - df['posting_date_dt'].min()).days} días calendario.")
    print(f"Días únicos registrados: {len(daily)} días.")
    print(f"Semanas únicas registradas: {len(weekly)} semanas.")
    
    generar_graficas_diagnostico(daily, weekly, carpeta_graficas)
    print("\nGráficas de diagnóstico exportadas a 300 DPI en:")
    print(f" - {os.path.join(carpeta_graficas, 'grafica_1_diagnostico_serie_diaria.png')}")
    print(f" - {os.path.join(carpeta_graficas, 'grafica_2_diagnostico_serie_semanal.png')}")
    print("=" * 80)
