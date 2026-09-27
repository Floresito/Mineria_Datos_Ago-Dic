
## 1. Introducción Teórica: Correlación vs. Causalidad

Antes de construir los modelos, evaluamos la relación lineal entre las variables numéricas del dataset mediante el **Coeficiente de Correlación de Pearson ($r$)**:

$$r = \frac{\sum (X_i - \bar{X})(Y_i - \bar{Y})}{\sqrt{\sum (X_i - \bar{X})^2 \sum (Y_i - \bar{Y})^2}}$$

### ¿Por qué Correlación no implica Causalidad?
- **Correlación:** Mide únicamente el grado de asociación matemática o covarianza conjunta entre dos variables continuas en una escala de $-1$ a $+1$. Si dos variables están correlacionadas, simplemente significa que cuando una cambia, la otra tiende a variar en una dirección predecible.
- **Causalidad:** Exige demostrar que la variación en la variable $X$ produce de manera directa el cambio en $Y$, descartando variables ocultas de confusión (*confounding variables*) o causalidad inversa. En datos observacionales como Craigslist, un modelo de regresión describe patrones empíricos y predicciones condicionales, no leyes causales deterministas.

### Matriz de Correlación Numérica Observada

| Variable | `price` | `year` | `odometer` | `cylinders_num` |
| :--- | :---: | :---: | :---: | :---: |
| **`price`** | `1.000` | `-0.376` | `-0.420` | `+0.278` |
| **`year`** | `-0.376` | `1.000` | `+0.378` | `-0.239` |
| **`odometer`** | `-0.420` | `+0.378` | `1.000` | `-0.307` |
| **`cylinders_num`** | `+0.278` | `-0.239` | `-0.307` | `1.000` |

La matriz completa se encuentra renderizada en `grafica_4_heatmap_correlacion.png`.

---

## Comparación de Modelos Candidatos y Selección del Modelo Oficial

Para no asumir arbitrariamente una sola estructura, evaluamos **4 modelos candidatos** mediante Mínimos Cuadrados Ordinarios (OLS), seleccionando el **Modelo D** como el modelo oficial del proyecto:

| Modelo | Especificación Matemática | $R^2$ | $R^2$ Ajustado | AIC | BIC | Error Medio (RMSE) | Diagnóstico Visual de Residuos |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Modelo A** (Lineal Simple) | $\text{Price} = \beta_0 + \beta_1 \text{Year}$ | `0.1417` | `0.1416` | 125,565.1 | 125,578.4 | \$14,611.60 | **Fuerte patrón en "U"**: subestima autos clásicos y modernos, sobrestimando los 80s. |
| **Modelo B** (Múltiple Lineal) | $\text{Price} = \beta_0 + \beta_1 \text{Year} + \beta_2 \text{Odo} + \sum \beta_k \text{Dummy}_k$ | `0.2591` | `0.2583` | 124,736.4 | 124,783.0 | \$13,581.77 | Reduce el error global, pero mantiene curvatura en el año. |
| **Modelo C** (Polinomial Grado 2) | $\text{Price} = \beta_0 + \beta_1 \text{Year} + \beta_2 \text{Year}^2$ | `0.1628` | `0.1625` | 125,425.0 | 125,444.9 | \$14,432.00 | La parábola absorbe la caída de los 70s y el repunte de los 60s. |
| **Modelo D** (Oficial Seleccionado) | $\text{Price} = \beta_0 + \beta_1 \text{Year} + \beta_2 \text{Year}^2 + \beta_3 \text{Odo} + \sum \beta_k \text{Dummy}_k$ | **`0.2928`** | **`0.2919`** | **124,473.1** | **124,526.3** | **\$13,270.68** | **Mejor dispersión**: menor AIC/BIC, menor error y coeficientes 100% significativos. |

---

## ¿Por qué la Regresión Polinomial es un "Modelo Lineal"?

Una duda teórica común es si incluir un término al cuadrado ($\text{Year}^2$) deja de ser una regresión lineal. 

En estadística matemática, **un modelo es lineal si es lineal respecto a sus parámetros ($\beta$)**, sin importar que las variables independientes tengan transformaciones cuadráticas, logarítmicas o polinomiales:

$$\text{Price} = \beta_0 + \beta_1 X_1 + \beta_2 X_2 + \dots + \epsilon \quad \text{donde } X_1 = \text{Year}, \ X_2 = \text{Year}^2$$

Dado que las derivadas parciales respecto a los coeficientes ($\frac{\partial Y}{\partial \beta_j}$) son funciones directas de las variables y no contienen parámetros elevados a potencias ($\beta_j^2$) ni productos entre parámetros ($\beta_1 \cdot \beta_2$), el sistema se resuelve de manera exacta mediante la solución matricial de Mínimos Cuadrados Ordinarios:
$$\hat{\beta} = (X^T X)^{-1} X^T Y$$
Por esta razón, la regresión polinomial forma parte de la familia de **Modelos Lineales Clásicos**.

---

## Narrativa de Diagnóstico en 3 Pasos y Gráficas del Reporte

### Paso 1: El Problema de la No Linealidad en el Modelo A
- **Ecuación estimada:** $\text{Price} = 833,900 - 412.03 \cdot \text{Year}$ ($R^2 = 0.1417$).
- **Gráfica de Dispersión:** (`grafica_1_scatter_regresion_simple.png`) Traza una línea recta descendente de -\$412 USD por año que no logra ajustarse a la realidad del mercado.
- **Gráfica de Residuos:** (`grafica_2_residuos_modelo_a.png`) La curva Lowess sobre los residuos dibuja una **forma de "U"**. Esto demuestra que una línea recta simple viola el supuesto de linealidad, ya que los autos clásicos de los 60s se venden muy por encima de lo que predice la recta, los autos de los 80s caen por debajo, y los autos de los 2000s vuelven a subir.

### Paso 2: La Solución Multivariada con el Modelo D (Oficial)
- **Ecuación estimada del Modelo D:**
  $$\begin{aligned}
  \widehat{\text{Price}} = & \ 84,810,000 - 85,240 \cdot \text{Year} + 21.42 \cdot \text{Year}^2 - 0.085 \cdot \text{Odometer} \\
  & + 2,258 \cdot \text{is\_convertible} + 6,219 \cdot \text{is\_clean\_title} \\
  & + 758 \cdot \text{is\_good\_condition} + 2,672 \cdot \text{cylinders\_num}
  \end{aligned}$$
- **Significancia de Coeficientes:** Todos los predictores resultaron significativos al nivel $\alpha = 0.05$:
  - `odometer`: Descuenta en promedio **-\$0.085 USD por cada milla** recorrida ($p < 0.001$).
  - `cylinders_num`: Agrega en promedio **+\$2,671.98 USD por cada cilindro adicional** ($p < 0.001$), reflejando el valor de los motores V8.
  - `is_clean_title`: Otorga una prima promedio de **+\$6,218.83 USD** frente a títulos salvage ($p < 0.001$).
  - `is_convertible`: Aporta una prima de **+\$2,257.77 USD** ($p < 0.001$).
- **Gráfica de Residuos:** (`grafica_3_residuos_modelo_d.png`) Muestra una distribución mucho más equilibrada alrededor del cero residual, reduciendo drásticamente la distorsión del Modelo A y alcanzando un $R^2$ ajustado de **`0.2919`**.

### Paso 3: Verificación de Multicolinealidad ($VIF$)
Evaluamos el Factor de Inflación de la Varianza ($VIF$) para confirmar que las variables predictoras no estuvieran duplicando información:
- `odometer`: $VIF = 1.250$
- `year`: $VIF = 1.235$
- `cylinders_num`: $VIF = 1.138$
- `is_convertible`: $VIF = 1.064$
- `is_good_condition`: $VIF = 1.024$
- `is_clean_title`: $VIF = 1.006$

Al estar todos los valores entre **1.0 y 1.25** (muy por debajo del límite de alerta de 5 o 10), confirmamos la **ausencia de multicolinealidad** entre las distintas dimensiones explicativas.

---

## Regresiones Segmentadas por Década Histórica

Al ajustar modelos simples $\text{Price} \sim \text{Year}$ por separado para cada época (`grafica_5_regresion_segmentada_decadas.png`), comprobamos que la relación año-precio cambia radicalmente de comportamiento según la era:

- **1960–1969 ($n = 1,329$):** Pendiente $\beta_1 = \mathbf{+\$592.12\text{ USD/año}}$ ($p = 0.031$). *En la era clásica, a mayor año dentro de la década, mayor fue el precio promedio.*
- **1970–1974 ($n = 612$):** Pendiente $\beta_1 = \mathbf{-\$3,800.71\text{ USD/año}}$ ($p = 5.60 \times 10^{-12}$). *Fuerte desplome de precios año tras año durante la entrada de regulaciones de emisiones.*
- **1975–1979 ($n = 357$):** Pendiente $\beta_1 = +\$714.09\text{ USD/año}$ ($p = 0.069$).
- **1980–1988 ($n = 712$):** Pendiente $\beta_1 = +\$45.72\text{ USD/año}$ ($p = 0.668$). *Pendiente totalmente plana.*
- **1989–2004 ($n = 2,693$):** Pendiente $\beta_1 = +\$19.07\text{ USD/año}$ ($p = 0.625$). *Pendiente plana.*

---

## ¿Por qué persiste dispersión en los residuos? 

Aunque el Modelo D mejora notablemente el ajuste ($R^2$ ajustado cercano al 30% y error reducido en más de \$1,300 USD por auto), la gráfica de residuos todavía muestra cierta dispersión. En el contexto de este dataset, esto se explica de manera clara y natural por dos razones fundamentales:

1. **Precio Pedido en el Anuncio vs. Precio Real de Venta:**
   El dataset de Craigslist registra el *asking price* (lo que el vendedor pide inicialmente al publicar el anuncio), no el precio final en el que se cierra el trato tras una negociación. En autos usados y clásicos, el precio pedido suele tener un margen variable de negociación que introduce ruido natural no explicable únicamente por las variables del catálogo.

2. **Información Clave Oculta en el Texto Libre de la Descripción:**
   El precio de un muscle car clásico depende de factores de valor muy específicos que **no existen como columnas numéricas en la tabla**, pero que los vendedores describen detalladamente en el texto libre de la columna `description`:
   - Si el auto conserva el motor original de fábrica (*"numbers matching"*).
   - El nivel de restauración (*"frame-off restoration"*, *"restored"*, *"garage kept"*).
   - Si se vende como auto de exhibición terminado o como auto para restaurar (*"project car"*, *"barn find"*).
   - Modificaciones mecánicas de alto desempeño (*"supercharged"*, *"crate engine"*).

Esta limitación natural de los modelos tabulares establece el puente directo hacia la **Práctica 9: Análisis de Texto (NLP)**, donde procesaremos las palabras clave de las descripciones para capturar ese valor cualitativo y mejorar la capacidad explicativa del precio.

---

## Catálogo de Figuras Generadas

1. **`grafica_1_scatter_regresion_simple.png`:** Dispersión con la recta ajustada del Modelo A.
2. **`grafica_2_residuos_modelo_a.png`:** Diagnóstico de residuos del Modelo A con curva Lowess en forma de U.
3. **`grafica_3_residuos_modelo_d.png`:** Diagnóstico de residuos del Modelo D oficial.
4. **`grafica_4_heatmap_correlacion.png`:** Mapa de calor de la matriz de correlación de Pearson.
5. **`grafica_5_regresion_segmentada_decadas.png`:** Comparación de rectas de regresión ajustadas por década histórica.
