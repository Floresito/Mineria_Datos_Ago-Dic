## Objetivo y Planteamiento del Problema

El objetivo de esta práctica es aplicar el algoritmo de aprendizaje no supervisado **K-Means Clustering** sobre el conjunto de muscle cars limpios ($N = 5,703$) para descubrir segmentos y agrupamientos naturales en el espacio multidimensional de características cuantitativas:

$$\text{Espacio de Atributos: } X = [\text{Price, Odometer, Year, Cylinders}]$$

### Lógica de Negocio y Conexión con la Investigación
A diferencia de la Práctica 6 (donde se intentó clasificar una etiqueta ordinal prefijada, `condition`), el enfoque de agrupamiento busca identificar estructuras latentes que combinan simultáneamente valor de mercado, kilometraje acumulado, antigüedad y configuración mecánica. 

Este análisis se conecta con la hipótesis central del proyecto: si los clústers formados presentan una **elevada dispersión interna en el precio** ($\sigma_{\text{price}}$ elevado en cada grupo) o solapamiento en sus fronteras, se aporta evidencia matemática de que las variables tabulares tradicionales no logran segmentar de manera determinista el mercado, reforzando la necesidad de analizar atributos no estructurados en el texto de las descripciones (**Práctica 9: NLP**).

---

## Justificación Teórica: Estandarización y Sensibilidad de K-Means

### Estandarización de Variables (`StandardScaler`)
El algoritmo K-Means asigna cada observación $x_i$ al clúster cuyo centroide $\mu_k$ minimiza la distancia euclidiana en el espacio $m$-dimensional:

$$d(x_i, \mu_k) = \sqrt{\sum_{j=1}^{m} (x_{ij} - \mu_{kj})^2}$$

* **Necesidad Técnica:** Si se ejecutara sobre los datos crudos, la magnitud cuadrática de `odometer` (escala $0 - 364,000$) o `price` (escala $500 - 225,000$) superaría a la de `year` ($1960 - 2004$) o `cylinders` ($3 - 10$) en un factor de $10^6$, haciendo que la partición dependiera exclusivamente de las variables con mayor rango numérico.
* **Transformación $Z$:** Mediante `StandardScaler`, cada feature se centra en $\mu = 0$ y se escala a $\sigma = 1$:
  $$Z = \frac{X - \mu}{\sigma}$$

### Sensibilidad a Outliers y Optimización de Inicialización (`n_init`)
* **Sensibilidad de los Centroides:** Dado que cada centroide se calcula como la media aritmética de su partición ($\mu_k = \frac{1}{|C_k|}\sum_{x \in C_k} x$), observaciones anómalas o extremas pueden distorsionar la posición de los centroides. En la Práctica 1 se aplicaron filtros basados en límites físicos e intercuartiles para mitigar este efecto.
* **Múltiples Inicializaciones (`n_init = 30`):** K-Means es un algoritmo voraz que converge a óptimos locales según la posición inicial de los centroides. El parámetro `n_init = 30` con inicialización `k-means++` garantiza que el algoritmo se ejecute 30 veces con semillas iniciales dispersas, seleccionando la solución que alcanza el mínimo global de inercia.

---

## Selección del Número de Clústers ($K$)

Se evaluó el espacio de agrupamiento en un rango exhaustivo de **$K = 2$ hasta $K = 10$**:

$$\text{Inercia (WCSS)} = \sum_{k=1}^{K} \sum_{x \in C_k} \|x - \mu_k\|^2 \quad\quad s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))}$$

### Tabla Comparativa de Métricas de Agrupamiento

| Número de Clústers ($K$) | Inercia Intra-Clúster (WCSS) | Silhouette Score Promedio |
| :---: | :---: | :---: |
| **$K = 2$** | $15,684.70$ | $0.3188$ |
| **$K = 3$** | $11,423.57$ | $0.3447$ |
| **$K = 4$** | **$9,250.77$** | **$0.3508$** *(Máximo Global)* |
| **$K = 5$** | $7,785.85$ | $0.3297$ |
| **$K = 6$** | $6,616.42$ | $0.3506$ *(Pico Secundario)* |
| **$K = 7$** | $5,780.96$ | $0.3420$ |
| **$K = 8$** | $5,214.13$ | $0.3412$ |
| **$K = 9$** | $4,746.12$ | $0.3319$ |
| **$K = 10$** | $4,368.96$ | $0.3135$ |

### Diagnóstico de Estabilidad Multi-Semilla
Para confirmar que la oscilación entre $K=4$, $K=5$ y $K=6$ no era ruido de inicialización, se evaluaron 5 semillas independientes (`random_state`: 42, 100, 2024, 7, 999) con $n\_init = 50$, obteniendo una desviación estándar de $\pm 0.0000$ en el coeficiente de silueta.

* **Criterio de Selección de $K = 4$:**
  1. **Método del Codo (`grafica_1_metodo_codo.png`):** La curva de WCSS muestra su desaceleración más clara en $K=4$, pasando de caídas abruptas a decrementos marginales lineales.
  2. **Silhouette Score (`grafica_2_silhouette_score.png`):** Alcanza el pico más alto en $K=4$ ($s = 0.3508$).
  3. **Principio de Parsimonia:** Modela la estructura fundamental del mercado en 4 grupos compactos sin sobrefragmentar subpoblaciones.

---

## Caracterización Cuantitativa de los Clústers ($K = 4$)

### Tabla de Centroides (Medias en Unidades Originales de Negocio)

| Clúster | Tamaño ($n$) | Proporción (%) | Precio Medio (USD) | Odómetro Medio (mi) | Año Medio | Cilindros Medio |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Clúster 0** | $1,058$ | $18.55\%$ | $\$5,072.72$ | $124,836.58$ | $1996.72$ | $5.76$ |
| **Clúster 1** | $2,103$ | $36.88\%$ | $\$17,913.07$ | $55,109.86$ | $1970.87$ | $7.85$ |
| **Clúster 2** | $2,245$ | $39.37\%$ | $\$13,003.59$ | $85,636.16$ | $1995.33$ | $8.00$ |
| **Clúster 3** | $297$ | $5.21\%$ | $\$63,905.10$ | $35,614.94$ | $1967.94$ | $7.89$ |

### Resumen de Medianas por Clúster

| Clúster | Precio Mediana (USD) | Odómetro Mediana (mi) | Año Mediana | Cilindros Mediana |
| :--- | :---: | :---: | :---: | :---: |
| **Clúster 0** | $\$3,700.00$ | $123,456.00$ | $2001.0$ | $6.0$ |
| **Clúster 1** | $\$16,500.00$ | $57,000.00$ | $1969.0$ | $8.0$ |
| **Clúster 2** | $\$12,000.00$ | $82,000.00$ | $1997.0$ | $8.0$ |
| **Clúster 3** | $\$55,000.00$ | $32,000.00$ | $1968.0$ | $8.0$ |

### Dispersión Interna (Desviación Estándar por Clúster)

| Clúster | Std Precio ($\sigma_{\text{price}}$) | Std Odómetro ($\sigma_{\text{odo}}$) | Std Año ($\sigma_{\text{year}}$) | Std Cilindros ($\sigma_{\text{cyl}}$) |
| :--- | :---: | :---: | :---: | :---: |
| **Clúster 0** | $\$5,256.28$ | $58,208.06$ | $9.65$ | $0.68$ |
| **Clúster 1** | $\$10,444.65$ | $37,100.68$ | $6.26$ | $0.54$ |
| **Clúster 2** | $\$7,510.87$ | $45,817.84$ | $6.74$ | $0.09$ |
| **Clúster 3** | $\$25,822.69$ | $33,677.23$ | $5.71$ | $0.47$ |

---

## Validación Cruzada con Variables Externas al Modelo

### 1. Cruzamiento con Condición Reportada (`condition`)

Distribución porcentual por fila (% de cada clúster):

| Clúster | `excellent` (%) | `good` (%) | `like new` (%) | `fair` (%) | `new` (%) | `salvage` (%) | `unknown` (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Clúster 0** | $24.95\%$ | $32.89\%$ | $6.71\%$ | $11.53\%$ | $0.09\%$ | $0.76\%$ | $23.06\%$ |
| **Clúster 1** | $28.96\%$ | $29.39\%$ | $4.09\%$ | $7.47\%$ | $0.33\%$ | $0.67\%$ | $29.10\%$ |
| **Clúster 2** | $32.96\%$ | $24.37\%$ | $9.98\%$ | $3.70\%$ | $0.53\%$ | $0.13\%$ | $28.33\%$ |
| **Clúster 3** | $36.36\%$ | $9.76\%$ | $12.12\%$ | $0.00\%$ | $1.35\%$ | $0.00\%$ | $40.40\%$ |

### 2. Cruzamiento con Fabricante (`manufacturer`)

Distribución porcentual por fila (% de cada clúster):

| Clúster | `chevrolet` (%) | `ford` (%) | `pontiac` (%) | `dodge` (%) | `buick` (%) | `mercury` (%) | `chrysler` (%) | `otros/unknown` (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Clúster 0** | $30.62\%$ | $20.42\%$ | $15.50\%$ | $1.13\%$ | $2.36\%$ | $1.42\%$ | $0.09\%$ | $28.45\%$ |
| **Clúster 1** | $36.99\%$ | $23.20\%$ | $9.13\%$ | $4.52\%$ | $2.47\%$ | $1.81\%$ | $0.00\%$ | $21.88\%$ |
| **Clúster 2** | $52.56\%$ | $29.35\%$ | $6.33\%$ | $0.53\%$ | $0.40\%$ | $0.71\%$ | $0.00\%$ | $10.11\%$ |
| **Clúster 3** | $55.22\%$ | $12.46\%$ | $6.06\%$ | $5.05\%$ | $0.34\%$ | $0.00\%$ | $0.34\%$ | $20.54\%$ |

---

## Verificación Adicional: Impacto del Millaje en la Subpoblación V8 (8 Cilindros)

Para comprobar si la configuración V8 protege o amortigua la pérdida de valor por kilometraje, se calculó la correlación entre `odometer` y `price` comparando el conjunto general contra la subpoblación exclusiva de 8 cilindros ($n = 4,463$, $78.3\%$ del dataset):

### Tabla Comparativa de Correlaciones y Ajuste Lineal

| Muestra / Subpoblación | Tamaño ($N$) | Pearson $r$ | Valor $p$ (Pearson) | Spearman $\rho$ | $R^2$ Lineal | Ecuación de Regresión Lineal |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Dataset Completo** | $5,703$ | **`-0.4199`** | $2.08 \times 10^{-242}$ | **`-0.5364`** | $0.1763$ | $\text{Price} = -0.1265 \cdot \text{Odo} + \$25,995$ |
| **Subpoblación V8 (8 Cilindros)** | $4,463$ | **`-0.3583`** | $2.85 \times 10^{-135}$ | **`-0.4430`** | $0.1284$ | $\text{Price} = -0.1279 \cdot \text{Odo} + \$27,363$ |
| **Subpoblación No-V8 (< 8 Cilindros)** | $1,240$ | **`-0.4569`** | $5.80 \times 10^{-65}$ | **`-0.5998`** | $0.2088$ | $\text{Price} = -0.0632 \cdot \text{Odo} + \$14,640$ |

### Interpretación Estadística de las Métricas
1. **Correlación de Pearson ($r$):** Mide la fuerza y dirección de la relación **lineal**. En la subpoblación V8, la correlación lineal ($r = -0.3583$) es moderada-baja y más débil que en la población general ($r = -0.4199$) y que en los no-V8 ($r = -0.4569$).
2. **Correlación de Spearman ($\rho$):** Mide la relación **monótona no paramétrica** basada en rangos. En autos V8 ($\rho = -0.4430$), el ordenamiento de precios respecto al kilometraje tiene un grado de dispersión considerablemente mayor que en los no-V8 ($\rho = -0.5998$).
3. **Pendiente de la Regresión:** En los autos V8 (`grafica_5_correlacion_v8_price_vs_odometer.png`), cada incremento de $10,000$ millas se asocia linealmente con una disminución promedio de $\$1,279$ USD en el precio pedido, pero con un coeficiente de determinación de apenas $R^2 = 12.84\%$, indicando que el $87.16\%$ de la variabilidad del precio en autos V8 no está explicada linealmente por el kilometraje.

---

## Proyecciones Visuales del Espacio de Clústers

1. **Proyección 2D en Variables Originales (`grafica_3_clusters_price_vs_year.png`):** Muestra el plano precio vs. año con los 4 centroides proyectados. Es una proyección bidimensional simplificada; la separación completa ocurre en el espacio 4D.
2. **Proyección 2D mediante PCA (`grafica_4_clusters_pca_2d.png`):** Reduce linealmente las 4 dimensiones estandarizadas a los 2 componentes ortogonales de máxima varianza, explicando el **$67.8\%$ de la varianza total** (PC1: $38.4\%$, PC2: $29.4\%$). Permite observar la separación geométrica global de los 4 grupos sin el colapso de dimensiones individuales.
3. **Dispersión V8 Precio vs. Odómetro (`grafica_5_correlacion_v8_price_vs_odometer.png`):** Mapeo de la subpoblación de 8 cilindros con su recta de regresión e intervalo de dispersión.

---

## Catálogo de Gráficas Exportadas

1. **`grafica_1_metodo_codo.png`:** Curva de inercia (WCSS) vs. $K \in [2, 10]$, señalando el punto de corte en $K=4$.
2. **`grafica_2_silhouette_score.png`:** Coeficiente de silueta promedio vs. $K \in [2, 10]$, destacando el máximo global en $K=4$ ($0.3508$).
3. **`grafica_3_clusters_price_vs_year.png`:** Scatter bidimensional de precio vs. año de fabricación con los centroides anotados.
4. **`grafica_4_clusters_pca_2d.png`:** Proyección bidimensional de Análisis de Componentes Principales (PCA) con porcentaje de varianza explicada.
5. **`grafica_5_correlacion_v8_price_vs_odometer.png`:** Scatter de precio vs. odómetro para el subconjunto de 8 cilindros ($n=4,463$) con la recta de regresión lineal.
