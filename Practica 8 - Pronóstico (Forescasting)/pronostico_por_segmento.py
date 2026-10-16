import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
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

# Carga de datos
# Segmentación  
# Carga el dataset de muscle cars limpios, mapea la condición en 3 segmentos:
#    - 'Regular / Proyecto' (fair, salvage)
#    - 'Buena' (good)
#    - 'Excelente' (new, like new, excellent)
#    
#    Acota la serie temporal a los 30 días continuos con recolección completa
#    (2021-04-05 a 2021-05-04 UTC) para asegurar una cuadrícula regular sin huecos.
def cargar_y_preparar_series_por_segmento(ruta_csv: str) -> tuple:
    if not os.path.exists(ruta_csv):
        raise FileNotFoundError(f"No se encontró el dataset en: {ruta_csv}")
    
    df = pd.read_csv(ruta_csv)
    df['posting_date_dt'] = pd.to_datetime(df['posting_date'], utc=True)
    df['date'] = df['posting_date_dt'].dt.date
    
    map_condicion = {
        'new': 'Excelente',
        'like new': 'Excelente',
        'excellent': 'Excelente',
        'good': 'Buena',
        'fair': 'Regular / Proyecto',
        'salvage': 'Regular / Proyecto'
    }
    df['segmento'] = df['condition'].map(map_condicion)
    df_known = df.dropna(subset=['segmento']).copy()
    
    # Rango de 30 días continuos y completos
    fecha_min = pd.to_datetime('2021-04-05').date()
    fecha_max = pd.to_datetime('2021-05-04').date()
    df_periodo = df_known[(df_known['date'] >= fecha_min) & (df_known['date'] <= fecha_max)].copy()
    
    # Serie de precio promedio diario por segmento
    daily_prices = df_periodo.groupby(['date', 'segmento'])['price'].mean().unstack()
    daily_counts = df_periodo.groupby(['date', 'segmento'])['price'].count().unstack()
    
    daily_prices.index = pd.to_datetime(daily_prices.index)
    daily_counts.index = pd.to_datetime(daily_counts.index)
    
    return df_periodo, daily_prices, daily_counts

# ==============================================================================
# 1. PARTICIÓN TEMPORAL CRONOLÓGICA (TRAIN / TEST)
# ==============================================================================

#  Aplica una división cronológica estricta:
#     - Conjunto de Entrenamiento (Train): Primeros 23 días (05-Abr a 27-Abr, 76.7% de la muestra).
#     - Conjunto de Prueba (Test): Últimos 7 días (28-Abr a 04-May, 23.3% de la muestra, 1 semana completa).
def particion_temporal(daily_prices: pd.DataFrame, dias_test: int = 7) -> tuple:
    train = daily_prices.iloc[:-dias_test].copy()
    test = daily_prices.iloc[-dias_test:].copy()
    return train, test

# ==============================================================================
# 2. COMPARACIÓN Y EVALUACIÓN DE MODELOS CANDIDATOS
# ==============================================================================
# Compara 3 enfoques de modelado para cada segmento sobre el conjunto de test:
#     1. Regresión Lineal sobre tiempo (Tendencia determinista t).
#     2. Modelo Clásico ARIMA(0,1,1) / ARIMA(1,0,0).
#     3. Modelo Base (Baseline: Media histórica en Train).
def comparar_modelos_por_segmento(train: pd.DataFrame, test: pd.DataFrame) -> dict:
    segmentos = ['Regular / Proyecto', 'Buena', 'Excelente']
    resultados = {}
    
    for seg in segmentos:
        y_tr = train[seg].values
        y_te = test[seg].values
        t_tr = np.arange(len(y_tr))
        t_te = np.arange(len(y_tr), len(y_tr) + len(y_te))
        
        # Modelo A: Regresión Lineal
        slope, intercept, r_val, p_val, std_err = stats.linregress(t_tr, y_tr)
        pred_reg = slope * t_te + intercept
        
        # Modelo B: ARIMA (orden optimizado por AIC)
        mejor_arima = None
        mejor_aic = float('inf')
        mejor_orden = None
        for orden in [(0, 1, 1), (1, 0, 0), (1, 1, 0), (0, 0, 1)]:
            try:
                m = ARIMA(y_tr, order=orden).fit()
                if m.aic < mejor_aic:
                    mejor_aic = m.aic
                    mejor_arima = m
                    mejor_orden = orden
            except Exception:
                pass
        
        pred_arima = mejor_arima.forecast(steps=len(y_te))
        
        # Modelo C: Baseline (Media de Train)
        pred_media = np.full_like(y_te, y_tr.mean())
        
        # Cálculo de métricas
        def calc_metricas(reales, pred):
            mae = mean_absolute_error(reales, pred)
            rmse = np.sqrt(mean_squared_error(reales, pred))
            mape = mean_absolute_percentage_error(reales, pred) * 100
            return mae, rmse, mape
        
        mae_reg, rmse_reg, mape_reg = calc_metricas(y_te, pred_reg)
        mae_ar, rmse_ar, mape_ar = calc_metricas(y_te, pred_arima)
        mae_base, rmse_base, mape_base = calc_metricas(y_te, pred_media)
        
        resultados[seg] = {
            'y_test_real': y_te,
            'pred_reg': pred_reg,
            'pred_arima': pred_arima,
            'pred_media': pred_media,
            'reg_slope': slope,
            'reg_pvalue': p_val,
            'arima_order': mejor_orden,
            'metricas': {
                'Regresion': {'MAE': mae_reg, 'RMSE': rmse_reg, 'MAPE': mape_reg},
                'ARIMA': {'MAE': mae_ar, 'RMSE': rmse_ar, 'MAPE': mape_ar},
                'Baseline_Media': {'MAE': mae_base, 'RMSE': rmse_base, 'MAPE': mape_base}
            }
        }
        
    return resultados

# ==============================================================================
# 3. PRONÓSTICO A FUTURO CON INTERVALO DE INCERTIDUMBRE (95%)
# ==============================================================================
# Reentrena el modelo óptimo sobre la totalidad de los 30 días históricos
# y proyecta 7 días hacia el futuro (05-May a 11-May de 2021) con intervalos de confianza al 95%.
# Selección del modelo final por segmento:
#     - 'Regular / Proyecto': ARIMA(0,1,1) / Media (serie estacionaria con media constante).
#     - 'Buena': ARIMA(0,1,1) (serie altamente estable alrededor de ~$11,300).
#     - 'Excelente': Regresión con Tendencia (captura la tasa de apreciación estadísticamente significativa).
def generar_pronostico_futuro(daily_prices: pd.DataFrame, pasos_futuros: int = 7) -> tuple:
    fechas_futuras = pd.date_range(start='2021-05-05', periods=pasos_futuros, freq='D')
    pronosticos = {}
    
    # 1. Regular / Proyecto: ARIMA(0,1,1)
    mod_reg = ARIMA(daily_prices['Regular / Proyecto'], order=(0, 1, 1)).fit()
    pred_res_reg = mod_reg.get_forecast(steps=pasos_futuros)
    ci_reg = pred_res_reg.conf_int(alpha=0.05)
    pronosticos['Regular / Proyecto'] = {
        'forecast': pred_res_reg.predicted_mean.values,
        'ci_lower': ci_reg.iloc[:, 0].values,
        'ci_upper': ci_reg.iloc[:, 1].values,
        'modelo': 'ARIMA(0,1,1)'
    }
    
    # 2. Buena: ARIMA(0,1,1)
    mod_bue = ARIMA(daily_prices['Buena'], order=(0, 1, 1)).fit()
    pred_res_bue = mod_bue.get_forecast(steps=pasos_futuros)
    ci_bue = pred_res_bue.conf_int(alpha=0.05)
    pronosticos['Buena'] = {
        'forecast': pred_res_bue.predicted_mean.values,
        'ci_lower': ci_bue.iloc[:, 0].values,
        'ci_upper': ci_bue.iloc[:, 1].values,
        'modelo': 'ARIMA(0,1,1)'
    }
    
    # 3. Excelente: Regresión Lineal con Tendencia
    t_full = np.arange(len(daily_prices))
    y_exc = daily_prices['Excelente'].values
    slope_exc, intercept_exc, r_val, p_val, std_err = stats.linregress(t_full, y_exc)
    t_fut = np.arange(len(daily_prices), len(daily_prices) + pasos_futuros)
    pred_exc = slope_exc * t_fut + intercept_exc
    
    # Intervalo de predicción analítico al 95%
    residuos = y_exc - (slope_exc * t_full + intercept_exc)
    s_err = np.sqrt(np.sum(residuos**2) / (len(y_exc) - 2))
    t_crit = stats.t.ppf(0.975, df=len(y_exc) - 2)
    s_pred = s_err * np.sqrt(1 + 1/len(y_exc) + (t_fut - np.mean(t_full))**2 / np.sum((t_full - np.mean(t_full))**2))
    ci_exc_lower = pred_exc - t_crit * s_pred
    ci_exc_upper = pred_exc + t_crit * s_pred
    
    pronosticos['Excelente'] = {
        'forecast': pred_exc,
        'ci_lower': ci_exc_lower,
        'ci_upper': ci_exc_upper,
        'modelo': f'Regresión Lineal (+{slope_exc:.2f} USD/día)'
    }
    
    return fechas_futuras, pronosticos

# ==============================================================================
# 4. EXPORTACIÓN DE VISUALIZACIONES A 300 DPI
# ==============================================================================
# Genera y guarda las figuras complementarias para la Práctica 8:
#     - Gráfica 3: Series históricas diarias de los 3 segmentos superpuestas.
#     - Gráfica 4: Evaluación de Valores Reales vs Predichos en Test (1 semana).
#     - Gráfica 5: Pronóstico a Futuro (7 días) para los 3 segmentos con bandas de incertidumbre al 95%.
def generar_graficas_pronostico(daily_prices: pd.DataFrame, test: pd.DataFrame, 
                                 resultados: dict, fechas_futuras: pd.DatetimeIndex, 
                                 pronosticos: dict, carpeta_salida: str) -> None:
    colores = {
        'Regular / Proyecto': '#d95f02', # Naranja rojizo
        'Buena': '#7570b3',              # Púrpura azulado
        'Excelente': '#1b9e77'           # Verde azulado
    }
    
    # --------------------------------------------------------------------------
    # Gráfica 3: Series Históricas de los 3 Segmentos
    # --------------------------------------------------------------------------
    plt.figure(figsize=(11, 6))
    for seg, col in colores.items():
        plt.plot(daily_prices.index, daily_prices[seg], marker='o', markersize=5, 
                 linewidth=2, color=col, label=f'{seg} (Media: ${daily_prices[seg].mean():,.0f})')
        
    plt.title("Evolución Histórica Diaria del Precio Promedio Pedido por Segmento de Condición", pad=12, fontweight='bold')
    plt.xlabel("Fecha de Publicación (2021)")
    plt.ylabel("Precio Promedio Diario (USD)")
    plt.gca().yaxis.set_major_formatter('${x:,.0f}')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=True)
    plt.tight_layout()
    ruta_g3 = os.path.join(carpeta_salida, "grafica_3_series_historicas_por_segmento.png")
    plt.savefig(ruta_g3, dpi=300, bbox_inches='tight')
    plt.close()
    
    # --------------------------------------------------------------------------
    # Gráfica 4: Valores Reales vs Predichos en Conjunto de Test (7 días)
    # --------------------------------------------------------------------------
    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
    dias_test = test.index
    
    for i, seg in enumerate(['Excelente', 'Buena', 'Regular / Proyecto']):
        ax = axes[i]
        res = resultados[seg]
        reales = res['y_test_real']
        pred_reg = res['pred_reg']
        pred_ar = res['pred_arima']
        
        ax.plot(dias_test, reales, marker='o', color='black', linewidth=2.2, label='Valor Real (Test)')
        ax.plot(dias_test, pred_reg, marker='s', linestyle='--', color='crimson', label=f"Regresión (MAPE={res['metricas']['Regresion']['MAPE']:.1f}%)")
        ax.plot(dias_test, pred_ar, marker='^', linestyle=':', color='royalblue', label=f"ARIMA {res['arima_order']} (MAPE={res['metricas']['ARIMA']['MAPE']:.1f}%)")
        
        ax.set_title(f"Segmento: {seg}", pad=6, fontweight='bold', fontsize=11)
        ax.set_ylabel("USD")
        ax.yaxis.set_major_formatter('${x:,.0f}')
        ax.legend(loc='upper right', frameon=True, fontsize=9)
        ax.grid(True, linestyle='--', alpha=0.6)
        
    axes[-1].set_xlabel("Fecha de Prueba (Última Semana de Recolección)")
    fig.suptitle("Evaluación en Conjunto de Prueba: Valores Reales vs. Predichos", fontsize=13, y=0.99, fontweight='bold')
    plt.tight_layout()
    ruta_g4 = os.path.join(carpeta_salida, "grafica_4_evaluacion_test_reales_vs_predichos.png")
    plt.savefig(ruta_g4, dpi=300, bbox_inches='tight')
    plt.close()
    
    # --------------------------------------------------------------------------
    # Gráfica 5: Pronóstico Futuro Comparativo a 7 Días con Bandas de Confianza (95%)
    # --------------------------------------------------------------------------
    plt.figure(figsize=(12, 7))
    
    for seg, col in colores.items():
        # Serie histórica
        plt.plot(daily_prices.index, daily_prices[seg], marker='.', color=col, alpha=0.65, label=f'{seg} (Histórico)')
        
        # Pronóstico futuro
        p_info = pronosticos[seg]
        plt.plot(fechas_futuras, p_info['forecast'], marker='s', color=col, linewidth=2.5, 
                 linestyle='--', label=f"{seg} (Pronóstico: {p_info['modelo']})")
        
        # Banda de incertidumbre
        plt.fill_between(fechas_futuras, p_info['ci_lower'], p_info['ci_upper'], 
                         color=col, alpha=0.18)
        
    plt.axvline(daily_prices.index[-1], color='black', linestyle=':', linewidth=1.5, label='Límite de Datos Históricos (04-May)')
    plt.title("Pronóstico a 7 Días Futuros del Precio Promedio por Segmento (con Bandas de Confianza al 95%)", pad=12, fontweight='bold')
    plt.xlabel("Línea Temporal (Abril - Mayo 2021)")
    plt.ylabel("Precio Promedio Diario (USD)")
    plt.gca().yaxis.set_major_formatter('${x:,.0f}')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=True, fontsize=9)
    plt.tight_layout()
    ruta_g5 = os.path.join(carpeta_salida, "grafica_5_pronostico_futuro_comparativo.png")
    plt.savefig(ruta_g5, dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    carpeta_practica = os.path.dirname(os.path.abspath(__file__))
    carpeta_graficas = os.path.join(carpeta_practica, "graficas")
    os.makedirs(carpeta_graficas, exist_ok=True)
    ruta_csv = os.path.join(carpeta_practica, "..", "Practica 1 - Limpieza de Datos", "muscle_cars_clean.csv")
    
    print("=" * 80)
    print("EJECUTANDO MODELADO Y PRONÓSTICO POR SEGMENTO (PRÁCTICA 8)")
    print("=" * 80)
    
    df_periodo, daily_prices, daily_counts = cargar_y_preparar_series_por_segmento(ruta_csv)
    print(f"Observaciones válidas en los 30 días: {len(df_periodo):,} registros.")
    print("Conteo medio diario de anuncios por segmento:")
    print(daily_counts.mean().round(1).to_string())
    
    # 1. Partición cronológica
    print("\n>>> [1/4] Aplicando partición temporal cronológica (23 días Train / 7 días Test)...")
    train, test = particion_temporal(daily_prices, dias_test=7)
    
    # 2. Comparación de modelos
    print("\n>>> [2/4] Comparando modelos (Regresión Lineal vs. ARIMA vs. Baseline)...")
    resultados = comparar_modelos_por_segmento(train, test)
    
    for seg, data in resultados.items():
        print(f"\n--- SEGMENTO: {seg.upper()} ---")
        m = data['metricas']
        print(f"  Regresión Lineal: MAE = ${m['Regresion']['MAE']:,.2f} | RMSE = ${m['Regresion']['RMSE']:,.2f} | MAPE = {m['Regresion']['MAPE']:.2f}% (Pendiente: {data['reg_slope']:.2f} USD/día, p={data['reg_pvalue']:.4f})")
        print(f"  ARIMA {data['arima_order']}:     MAE = ${m['ARIMA']['MAE']:,.2f} | RMSE = ${m['ARIMA']['RMSE']:,.2f} | MAPE = {m['ARIMA']['MAPE']:.2f}%")
        print(f"  Baseline Media:   MAE = ${m['Baseline_Media']['MAE']:,.2f} | RMSE = ${m['Baseline_Media']['RMSE']:,.2f} | MAPE = {m['Baseline_Media']['MAPE']:.2f}%")
    
    # 3. Pronóstico futuro a 7 días
    print("\n>>> [3/4] Generando pronóstico a 7 días futuros (05-May a 11-May 2021) con IC 95%...")
    fechas_futuras, pronosticos = generar_pronostico_futuro(daily_prices, pasos_futuros=7)
    
    for seg, p in pronosticos.items():
        media_fut = p['forecast'].mean()
        ic_inf = p['ci_lower'].mean()
        ic_sup = p['ci_upper'].mean()
        print(f"  {seg:20s} -> Media Proyectada: ${media_fut:,.2f} (IC 95% promedio: [${ic_inf:,.2f} - ${ic_sup:,.2f}]) [{p['modelo']}]")
        
    # 4. Generación de Gráficas
    print("\n>>> [4/4] Exportando gráficas de diagnóstico, test y pronóstico a 300 DPI...")
    generar_graficas_pronostico(daily_prices, test, resultados, fechas_futuras, pronosticos, carpeta_graficas)
    
    print("\n" + "=" * 80)
    print("EJECUCIÓN COMPLETADA CON ÉXITO. TODAS LAS FIGURAS GUARDADAS EN 'graficas/'.")
    print("=" * 80)
    