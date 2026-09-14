## 1. Justificación Metodológica y Estrategias de Visualización

En esta práctica se diseñaron e implementaron **6 tipos distintos de visualizaciones** orientadas al análisis exploratorio del mercado de muscle cars (1960–2004), priorizando la **automatización modular del código** mediante funciones reutilizables y estructuras iterativas (`for` loops):

| N° | Tipo de Visualización | Variables Involucradas | Justificación Técnica de Selección | Patrón de Automatización Aplicado |
| :--- | :--- | :--- | :--- | :--- |
| **1A** | **Histogramas Univariados** | `price`, `year`, `odometer` | Permite evaluar simultáneamente la simetría, curtosis y la presencia de colas pesadas en las variables cuantitativas continuas. | `for` loop iterando sobre la lista de columnas numéricas en un multipanel `plt.subplots(1, 3)`. |
| **1B** | **Histograma Facetado por Década** | `price` vs. `decade` | Evita la superposición caótica de datos y permite comparar la evolución de la forma de la distribución de precios en cada era histórica. | `for` loop iterando sobre los niveles de la variable ordinal `decade` con `sharex=True`. |
| **2A/2B** | **Diagramas de Caja (Boxplots)** | `price` por `manufacturer` y `condition` | Idóneos para contrastar la mediana, dispersión intercuartílica (IQR) y la densidad de valores atípicos (outliers) entre categorías sin asumir normalidad. | Función genérica reutilizable `crear_boxplot_agrupado(df, cat_col, num_col)` con ordenamiento automático por mediana. |
| **3A/3B** | **Diagramas de Dispersión (Scatter)** | `price` vs. `odometer` y `price` vs. `year` (hue: `manufacturer`) | Permite inspeccionar relaciones bivariadas continuas, detectar no linealidades temporales y evaluar el decaimiento por desgaste kilométrico. | Función modular con ajuste de transparencia (`alpha=0.35`) y línea de regresión de tendencia (`sns.regplot`). |
| **4A/4B** | **Diagramas de Pastel (Pie / Donut)** | Participación por `manufacturer` y `decade` | Visualiza la composición porcentual del mercado. | Función `crear_pie_chart()` con algoritmo automático de colapso de categorías minoritarias en `"Otros"` ($\le 6$ rebanadas). |
| **5** | **Líneas de Tendencia Temporal** | `price` medio vs. `year` por fabricante | Muestra la trayectoria longitudinal interanual con bandas de intervalo de confianza al 95% calculadas por *bootstrapping*. | Agrupación temporal continua con `sns.lineplot` desglosada por marcas principales. |
| **6** | **Diagramas de Violín (Violin Plots)** | `price` vs. `decade` | Combina boxplots internos con funciones de estimación de densidad de kernel (KDE) para detectar bimodalidad o subsegmentos dentro de cada década. | Función `sns.violinplot` con normalización de ancho y visualización de cuartiles internos. |

---

## 2. Catálogo de Gráficas Generadas

Todas las figuras fueron renderizadas con `Seaborn` sobre backend de `Matplotlib` y exportadas a alta resolución:

1. **`grafica_1a_histogramas_numericos.png`:**
   - Panel de 3 subplots para `price`, `year` y `odometer` con curvas KDE y líneas verticales indicando Media (rojo discontinuo) y Mediana (verde continuo).
2. **`grafica_1b_histograma_price_por_decada.png`:**
   - Desglose apilado de 5 histogramas de precio ($1960-1969$, $1970-1974$, $1975-1979$, $1980-1988$, $1989-2004$) compartiendo el mismo eje X.
3. **`grafica_2a_boxplot_price_por_fabricante.png`:**
   - Diagrama de caja de precio ordenado descendentemente por mediana de fabricante con marcas de media aritmética.
4. **`grafica_2b_boxplot_price_por_condicion.png`:**
   - Diagrama de caja de precio ordenado por nivel de conservación vehicular (`excellent`, `good`, `fair`, `like new`, etc.).
5. **`grafica_3a_scatter_price_vs_odometer.png`:**
   - Dispersión bivariada precio vs. odómetro con recta de tendencia lineal superpuesta.
6. **`grafica_3b_scatter_price_vs_year_hue_fabricante.png`:**
   - Dispersión precio vs. año de fabricación con codificación cromática para Chevrolet, Ford, Pontiac, Dodge y Otros.
7. **`grafica_4a_pie_anuncios_por_fabricante.png`:**
   - Gráfico de pastel/dona de participación de mercado por fabricante con las marcas minoritarias agrupadas en "Otros".
8. **`grafica_4b_pie_anuncios_por_decada.png`:**
   - Gráfico de pastel/dona de volumen de anuncios por intervalo de décadas históricas.
9. **`grafica_5_lineplot_tendencia_temporal_fabricante.png`:**
   - Gráfico longitudinal de líneas de precio medio año por año (1960–2004) con intervalos de confianza al 95% para las principales marcas y línea guía de media global.
10. **`grafica_6_violinplot_price_por_decada.png`:**
    - Gráfico de violines que representa la densidad probabilística completa de precios por década.
