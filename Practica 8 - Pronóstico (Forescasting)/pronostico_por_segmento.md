## Objetivo y Planteamiento del Problema

El objetivo de esta práctica es modelar y evaluar la trayectoria temporal del precio promedio pedido (*asking price*) en el mercado de muscle cars mediante técnicas de **series de tiempo y regresión temporal**, segmentando el mercado según la **condición del vehículo (`condition`)**:

$$\text{Segmentos Evaluados: } S = [\text{Regular / Proyecto}, \text{Buena}, \text{Excelente}]$$

### Contexto Metodológico y Lógica de Negocio
A diferencia de un pronóstico agregado sobre todo el mercado combinado, este enfoque aborda la pregunta de decisión de compra en distintos niveles de estado físico: *¿el precio de un automóvil para restaurar ("coche proyecto") exhibe una tendencia temporal definida (conveniencia de timing de compra) o se comporta como un proceso estocástico estacionario alrededor de una media constante? ¿Difiere este comportamiento respecto a vehículos en condición óptima o de colección?*

### Delimitación Crucial de los Datos y Marco Legal
1. **Asking Price vs. Transaction Price:** La variable `posting_date` registra la fecha y hora de publicación del anuncio en Craigslist, y `price` representa el precio solicitado por el vendedor, no el valor de una transacción cerrada. Los pronósticos reflejan expectativas de oferta del mercado vendedor durante el período analizado.
2. **Validación Estrictamente Interna:** Siguiendo los Términos de Servicio de Craigslist (que prohíben la recolección automatizada no autorizada con precedentes judiciales formales, como *Craigslist Inc. v. 3Taps Inc.*), la validación del modelo se realiza exclusivamente mediante la división cronológica *Train / Test* dentro de la muestra histórica recolectada, sin extracción de datos externos.

---

## Diagnóstico del Rango Temporal y Granularidad

### Amplitud de la Ventana Temporal
Al analizar la variable `posting_date` (en formato UTC) sobre las observaciones con condición identificada ($n = 4,051$):
* **Fecha Mínima:** `2021-04-04 09:54:27 UTC`
* **Fecha Máxima:** `2021-05-05 03:15:18 UTC`
* **Amplitud Total:** $30$ días calendario continuos y completos (del **05 de abril al 04 de mayo de 2021**), excluyendo los días extremos de corte parcial.
* **Tratamiento del Vacío en Semana 1 (04 de abril):** La recolección del dataset inició el domingo 04 de abril a las 09:54 AM, registrando únicamente 31 anuncios en total y **cero observaciones para `Regular / Proyecto`** ($n = 0$). Para evitar distorsiones estadísticas críticas (como valores indeterminados `NaN` por división entre cero o caídas artificiales a $\$0$ USD que sesgarían la inercia, la correlación serial y las pendientes), **dicho día inicial fue formalmente omitido del modelado**. A partir del 05 de abril, el segmento cuenta con registros reales en el $100\%$ de los 30 días continuos (mínimo de 3 anuncios/día, promedio de 12.8 anuncios/día).

### Justificación de la Granularidad Diaria y Horizonte Prudente
* **Inviabilidad Mensual y Semanal:** A nivel mensual solo existen $N = 2$ meses, y a nivel semanal solo $N = 6$ semanas calendario, lo cual imposibilita matemáticamente el ajuste de modelos de series de tiempo por falta de grados de libertad.
* **Cuadrícula Diaria Regular ($N = 30$):** Permite contar con una serie continua sin discontinuidades temporales.
* **Horizonte de Pronóstico Máximo:** Siguiendo el principio econométrico de que el horizonte futuro $h$ no debe superar el $15\% - 25\%$ de la longitud de la serie, el límite técnicamente defendible es de **$h = 7$ días futuros** (una semana adelante). Pronósticos a meses o años con una serie de 30 días serían estadísticamente inválidos.

---

## Justificación Teórica: Partición Cronológica vs. Partición Aleatoria

A diferencia del aprendizaje supervisado clásico (como el clasificador KNN de la Práctica 6), en series de tiempo **la división de datos debe respetar estrictamente el orden temporal**:

* **Conjunto de Entrenamiento (*Train*):** Primeros $23$ días ($05\text{-Abr}$ a $27\text{-Abr}$, $76.7\%$ de la serie).
* **Conjunto de Prueba (*Test*):** Últimos $7$ días ($28\text{-Abr}$ a $04\text{-May}$, $23.3\%$ de la serie, 1 semana completa).

---

## Definición Matemática de Métricas de Error

Para comparar el rendimiento predictivo en la semana de prueba se calcularon tres métricas complementarias:

1. **Error Absoluto Medio (*Mean Absolute Error*, MAE):**
   $$\text{MAE} = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|$$
   Mide la magnitud promedio del error en las unidades monetarias originales (USD). Pondera todos los desvíos de forma lineal.

2. **Raíz del Error Cuadrático Medio (*Root Mean Squared Error*, RMSE):**
   $$\text{RMSE} = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2}$$
   Al elevar las discrepancias al cuadrado antes del promedio, penaliza severamente los errores grandes o atípicos. Si $\text{RMSE} \gg \text{MAE}$, existen errores puntuales de magnitud sustancial.

3. **Error Porcentual Absoluto Medio (*Mean Absolute Percentage Error*, MAPE):**
   $$\text{MAPE} = \frac{100\%}{n} \sum_{i=1}^{n} \left| \frac{y_i - \hat{y}_i}{y_i} \right|$$
   Proporciona una medida de error adimensional en porcentaje relativo al valor real, permitiendo comparar la precisión predictiva entre segmentos con escalas de precio radicalmente dispares ($\$5,000$ vs. $\$20,000$ USD).

---

## Comparación y Evaluación de Modelos en la Semana de Prueba (Test)

Para cada uno de los tres segmentos, se ajustaron y compararon sobre los $23$ días de *train* tres familias de modelos, evaluándose sobre los $7$ días de *test*:
* **Modelo A (Regresión Lineal sobre tiempo $t$):** $\hat{y}_t = \beta_0 + \beta_1 t$.
* **Modelo B (Modelo Clásico ARIMA):** Seleccionado mediante minimización del Criterio de Información de Akaike (AIC), convergiendo a $\text{ARIMA}(0,1,1)$ con integración de primer orden y término de media móvil.
* **Modelo C (Línea Base / Naive Mean):** $\hat{y}_t = \bar{y}_{\text{train}}$ (pronóstico constante igual al promedio histórico).

### Tabla Comparativa de Desempeño Predictivo en Test ($n = 7$ días)

| Segmento de Condición | Modelo Evaluado | MAE (USD) | RMSE (USD) | MAPE (%) | Comportamiento del Ajuste |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **`Regular / Proyecto`** | Regresión Lineal | $\$1,758.94$ | $\$1,846.05$ | $33.74\%$ | Sobreajusta pendiente bajista artificial en train; sobre-penaliza en test. |
| | **ARIMA(0,1,1)** | $\$755.45$ | $\$921.19$ | $13.83\%$ | Ajuste adaptativo de media móvil; captura la variabilidad local. |
| | **Baseline (Media Histórica)** | **`$527.57`** | **`$645.88`** | **`9.98%`** | **Mejor modelo:** confirma comportamiento estacionario alrededor de la media. |
| **`Buena`** | Regresión Lineal | $\$793.87$ | $\$1,009.17$ | $6.81\%$ | Pendiente casi nula ($+\$9.86$ USD/día, $p = 0.8864$). |
| | **ARIMA(0,1,1)** | **`$754.48`** | **`$1,029.70`** | **`6.34%`** | **Mejor modelo:** menor error porcentual medio ($MAPE = 6.34\%$). |
| | Baseline (Media Histórica) | $\$767.11$ | $\$1,012.92$ | $6.50\%$ | Error virtualmente idéntico a ARIMA; mercado con precio anclado. |
| **`Excelente`** | **Regresión Lineal** | **`$1,555.58`** | **`$1,978.69`** | **`7.36%`** | **Mejor modelo:** captura la tendencia positiva sostenida ($+\$72.80$ USD/día). |
| | ARIMA(0,1,1) | $\$1,898.10$ | $\$2,168.67$ | $9.62\%$ | Subestima el valor medio al suavizar la tendencia hacia el pasado reciente. |
| | Baseline (Media Histórica) | $\$2,085.96$ | $\$2,585.96$ | $9.60\%$ | Queda sistemáticamente por debajo de los precios de cierre. |

---

## Pronóstico a Futuro a 7 Días (05-May a 11-May 2021) con Intervalos al 95%

Habiendo identificado el enfoque superior para cada dinámica temporal, se reentrenó cada modelo sobre los $30$ días completos de datos históricos para proyectar los $7$ días siguientes con sus bandas de incertidumbre al $95\%$:

### Resumen del Pronóstico por Segmento

| Segmento de Condición | Modelo Definitivo | Precio Medio Histórico | Precio Proyectado Medio | Intervalo de Confianza al 95% | Interpretación Matemática de la Banda |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **`Regular / Proyecto`** | $\text{ARIMA}(0,1,1)$ | $\$4,897.01$ | **$\$4,800.45$** | $[\$2,398.07 - \$7,202.84]$ | Banda amplia ($\pm \$2,400$) reflejo del bajo soporte muestral diario ($12.8$ ads/día). |
| **`Buena`** | $\text{ARIMA}(0,1,1)$ | $\$11,366.37$ | **$\$11,311.88$** | $[\$6,737.70 - \$15,886.06]$ | Trayectoria horizontal plana; estabilidad robusta del precio pedido. |
| **`Excelente`** | Regresión Lineal | $\$19,223.04$ | **$\$21,055.61$** | $[\$16,572.25 - \$25,538.97]$ | Proyección con pendiente positiva ($+\$99.06$ USD/día, $p = 0.0268$). |

### Detalle Diario del Pronóstico Futuro a 7 Días (USD)

| Fecha Futura | `Regular / Proyecto` (Puntual [IC 95%]) | `Buena` (Puntual [IC 95%]) | `Excelente` (Puntual [IC 95%]) |
| :---: | :---: | :---: | :---: |
| **2021-05-05** | $\$4,800.45$ $[\$2,398.84 - \$7,202.06]$ | $\$11,311.88$ $[\$6,821.65 - \$15,802.11]$ | $\$20,758.44$ $[\$16,364.60 - \$25,152.27]$ |
| **2021-05-06** | $\$4,800.45$ $[\$2,398.58 - \$7,202.33]$ | $\$11,311.88$ $[\$6,793.67 - \$15,830.09]$ | $\$20,857.49$ $[\$16,434.61 - \$25,280.38]$ |
| **2021-05-07** | $\$4,800.45$ $[\$2,398.31 - \$7,202.59]$ | $\$11,311.88$ $[\$6,765.68 - \$15,858.07]$ | $\$20,956.55$ $[\$16,504.47 - \$25,408.63]$ |
| **2021-05-08** | $\$4,800.45$ $[\$2,398.05 - \$7,202.86]$ | $\$11,311.88$ $[\$6,737.70 - \$15,886.06]$ | $\$21,055.61$ $[\$16,574.19 - \$25,537.03]$ |
| **2021-05-09** | $\$4,800.45$ $[\$2,397.79 - \$7,203.12]$ | $\$11,311.88$ $[\$6,709.71 - \$15,914.05]$ | $\$21,154.67$ $[\$16,643.76 - \$25,665.57]$ |
| **2021-05-10** | $\$4,800.45$ $[\$2,397.52 - \$7,203.38]$ | $\$11,311.88$ $[\$6,681.72 - \$15,942.04]$ | $\$21,253.72$ $[\$16,713.20 - \$25,794.25]$ |
| **2021-05-11** | $\$4,800.45$ $[\$2,397.26 - \$7,203.65]$ | $\$11,311.88$ $[\$6,653.74 - \$15,970.02]$ | $\$21,352.78$ $[\$16,782.50 - \$25,923.06]$ |

---

## Catálogo de Gráficas Exportadas

1. **`grafica_1_diagnostico_serie_diaria.png`:** Diagnóstico general en dos paneles: panel A (volumen de anuncios por día mostrando la rampa del web crawler) y panel B (precio promedio y mediana diaria con banda de dispersión $\pm 1\sigma$).
2. **`grafica_2_diagnostico_serie_semanal.png`:** Diagnóstico comparativo por semana calendario ($N = 6$ semanas), ilustrando la insuficiencia muestral para modelos agregados a nivel semanal o mensual.
3. **`grafica_3_series_historicas_por_segmento.png`:** Serie histórica diaria ($30$ días) de los 3 segmentos simultáneos, evidenciando las tres franjas de precio y sus volatilidades relativas.
4. **`grafica_4_evaluacion_test_reales_vs_predichos.png`:** Contraste de valores reales observados vs. proyecciones de Regresión Lineal y ARIMA en la semana de prueba (últimos 7 días).
5. **`grafica_5_pronostico_futuro_comparativo.png`:** Proyección a $7$ días futuros superpuesta para los 3 segmentos con sus bandas de incertidumbre al $95\%$.
