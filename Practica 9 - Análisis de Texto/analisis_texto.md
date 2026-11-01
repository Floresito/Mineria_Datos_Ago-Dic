## Objetivo y Planteamiento del Problema

El objetivo de esta práctica es realizar el procesamiento de lenguaje natural (NLP) y análisis exploratorio sobre el texto no estructurado de la columna `description` ($N = 5,703$ anuncios de Craigslist) para comprobar empíricamente las dos hipótesis que conectan las prácticas anteriores del proyecto semestral:

1. **Hipótesis de la Práctica 5 (Variables Omitidas):** El poder explicativo limitado del modelo tabular de precios ($R^2_{\text{adj}} \approx 0.29$) se debe a la ausencia de atributos cualitativos determinantes de originalidad, documentación y restauración que residen exclusivamente en el texto libre redactado por el vendedor.
2. **Hipótesis de la Práctica 8 (Volatilidad en "Regular / Proyecto"):** La elevada variabilidad residual y dificultad de pronóstico en los autos para restaurar responde a dos explicaciones tentativas: presencia de vendedores "highballers" (precio firme intransigente) vs. ruido de muestra pequeña.

---

## Metodología de Limpieza, Preprocesamiento y Lematización

Para balancear la depuración de ruido técnico con la conservación de especificidad mecánica, se aplicó un pipeline modular:

1. **Remoción de Ruido Técnico:** Eliminación de etiquetas y entidades HTML, enlaces web (URLs), correos electrónicos y números telefónicos mediante expresiones regulares especializadas.
2. **Lematización Canónica (WordNet Lemmatizer):** En lugar de *stemming* (que recorta palabras a raíces morfológicas inexistentes en el diccionario como `restor` o `condit`), se utilizó **lematización**, transformando cada palabra a su lema léxico canónico (`restored/restoring` $\to$ `restore`; `running/runs` $\to$ `run`; `cars` $\to$ `car`), garantizando la interpretabilidad de los términos en las nubes y tablas.
3. **Tratamiento de Cifras Numéricas (Opción B):** Se preservaron exclusivamente los números correspondientes a especificaciones mecánicas clásicas reconocidas (cilindradas en pulgadas cúbicas como `350`, `396`, `427`, `454`, `302`, `440`, y transmisiones como `4speed`), descartando precios, códigos postales y dígitos espurios.
4. **Vocabulario de Stopwords Ampliado:** Se combinaron las $179$ stopwords estándar del idioma inglés de NLTK con una lista de $18$ stopwords transaccionales de dominio (`call`, `text`, `phone`, `email`, `contact`, `ask`, `asking`, `obo`, `sale`, `sell`, `price`, `cash`, `title`, `miles`, `mileage`, `year`, `car`, `vehicle`, `auto`).
5. **Conservación del Texto Original:** Se mantuvo una copia intacta de la descripción cruda en minúsculas para la búsqueda de frases multi-palabra exactas sin alteración sintáctica.

---

## Nubes de Palabras y Frecuencia Léxica

Se generaron visualizaciones léxicas con parámetros estéticos unificados tanto a nivel global (`grafica_2_nube_palabras_global.png`) como segmentadas por condición (`grafica_3_nubes_palabras_por_segmento.png`).

### Tabla de las 20 Palabras Más Frecuentes en el Corpus Global ($N = 5,703$)

| Posición | Palabra Lematizada | Frecuencia Absoluta | Contexto Mecánico / Descriptivo |
| :---: | :--- | :---: | :--- |
| **1** | `new` | $7,018$ | Piezas reemplazadas, componentes recientes o estado general. |
| **2** | `power` | $4,432$ | Equipamiento asistido (power steering, power brakes, power windows). |
| **3** | `corvette` | $4,380$ | Modelo de mayor volumen y presencia en el corpus. |
| **4** | `chevrolet` | $4,185$ | Fabricante dominante de la muestra. |
| **5** | `drive` | $4,077$ | Capacidad de rodar o tipo de tracción (rear wheel drive). |
| **6** | `engine` | $3,339$ | Motorización y configuración bajo el cofre. |
| **7** | `original` | $3,319$ | Grado de preservación de componentes de fábrica. |
| **8** | `wheel` | $3,289$ | Rines, neumáticos o dirección. |
| **9** | `convertible` | $3,125$ | Tipo de carrocería valorada en el nicho. |
| **10** | `interior` | $2,911$ | Estado de vestiduras, tablero y tapicería. |
| **11** | `good` | $2,873$ | Condición funcional reportada por el anunciante. |
| **12** | `clean` | $2,845$ | Condición estética o estatus de título. |
| **13** | `speed` | $2,788$ | Configuración de transmisión (4-speed, automatic). |
| **14** | `run` | $2,746$ | Estado de operación mecánica ("runs and drives"). |
| **15** | `brake` | $2,735$ | Sistema de frenado (frenos de disco / tambor). |
| **16** | `paint` | $2,580$ | Calidad o presencia de repintado de carrocería. |
| **17** | `look` | $2,342$ | Apariencia estética general. |
| **18** | `350` | $2,233$ | Motor Chevrolet Small Block 350 ci (especificación mecánica preservada). |
| **19** | `great` | $2,185$ | Término evaluativo de condición. |
| **20** | `door` | $2,094$ | Configuración de 2 o 4 puertas. |

---

## Análisis de N-Gramas y Términos Distintivos por TF-IDF

### 1. Frecuencia de Bigramas y Trigramas Globales (`grafica_4_frecuencia_palabras_y_bigramas.png`)

| Posición | Bigrama Más Frecuente | Conteo | Trigrama Más Frecuente | Conteo |
| :---: | :--- | :---: | :--- | :---: |
| **1** | `chevrolet corvette` | $1,599$ | `power front disc` | $342$ |
| **2** | `ford mustang` | $1,179$ | `front disc brake` | $341$ |
| **3** | `run drive` | $1,063$ | `air condition power` | $256$ |
| **4** | `power steer` | $974$ | `good condition run` | $244$ |
| **5** | `chev chevy` | $950$ | `wheel power front` | $240$ |
| **6** | `disc brake` | $868$ | `rear wheel drive` | $239$ |
| **7** | `power brake` | $850$ | `steering power brake` | $238$ |
| **8** | `rear wheel` | $754$ | `power steering power` | $238$ |
| **9** | `clean title` | $696$ | `speed manual transmission` | $232$ |
| **10** | `air condition` | $692$ | `front wheel drive` | $210$ |

---

### 2. Términos y Frases Distintivas por Segmento mediante TF-IDF

El cálculo de **TF-IDF** (*Term Frequency - Inverse Document Frequency*) pondera la frecuencia local de un término $t$ en un segmento $d$ penalizando su prevalencia en el resto de los documentos $D$:

$$\text{TF-IDF}(t, d, D) = \text{TF}(t, d) \times \left[ \ln\left(\frac{1 + |D|}{1 + \text{DF}(t)}\right) + 1 \right]$$

A diferencia de la frecuencia simple (que resalta palabras comunes en todos los autos como `engine` o `drive`), TF-IDF aísla el vocabulario especializado que discrimina a cada segmento:

* **Segmento `Regular / Proyecto`:** Términos de reparación, piezas faltantes y deterioro estructural:
  $$\text{Top TF-IDF: } [\text{project}, \text{need}, \text{rust}, \text{part}, \text{work}, \text{run}, \text{motor}, \text{clean}, \text{trans}, \text{body}, \text{floor}, \text{rebuild}]$$
* **Segmento `Buena`:** Términos orientados a vehículos funcionales de uso cotidiano:
  $$\text{Top TF-IDF: } [\text{good}, \text{run}, \text{drive}, \text{great}, \text{tire}, \text{look}, \text{clean}, \text{paint}, \text{new}, \text{nice}, \text{original}]$$
* **Segmento `Excelente`:** Vocabulario centrado en originalidad, estética y componentes de alta gama:
  $$\text{Top TF-IDF: } [\text{condition}, \text{original}, \text{paint}, \text{leather}, \text{show}, \text{custom}, \text{corvette}, \text{power}, \text{speed}, \text{wheel}, \text{door}]$$
* **Segmento `Desconocido`:** Términos de concesionarios comerciales, créditos y trámites de agencia:
  $$\text{Top TF-IDF: } [\text{dealer}, \text{financing}, \text{warranty}, \text{credit}, \text{apply}, \text{tax}, \text{fee}, \text{approval}, \text{stock}, \text{vin}, \text{inspection}]$$

---

## Prueba de Hipótesis 1: Lenguaje "Highballer" / Precio Firme

Para verificar si la volatilidad en el precio de los autos proyecto se debe a vendedores intransigentes que fijan precios desproporcionados ("highballers"), se evaluó la presencia de frases clave sobre el texto crudo:
$$\text{Frases: } [\text{"no lowballers"}, \text{"serious buyers only"}, \text{"price is firm"}, \text{"firm price"}, \text{"i know what i have"}, \text{"no tire kickers"}, \dots]$$

### Prevalencia de Frases Highballer por Segmento (`grafica_5_prevalencia_lenguaje_highballer.png`)

| Segmento de Condición | Anuncios Totales ($n$) | Anuncios con Lenguaje Highballer ($n$) | Prevalencia Relativa (%) |
| :--- | :---: | :---: | :---: |
| **`Regular / Proyecto`** | $387$ | $23$ | **$5.94\%$** |
| **`Buena`** | $1,542$ | $95$ | **$6.16\%$** |
| **`Excelente`** | $2,162$ | $158$ | **$7.31\%$** |
| **`Desconocido`** | $1,612$ | $86$ | **$5.33\%$** |
| **Total Global** | **$5,703$** | **$362$** | **$6.35\%$** |

> [!IMPORTANT]
> **Resultado Estadístico sobre Highballers:**
> La tasa de anuncios con frases de intransigencia o precio firme se mantiene prácticamente uniforme entre todos los segmentos ($\approx 5.3\% - 7.3\%$). De hecho, el segmento `Excelente` presenta una proporción ligeramente mayor ($7.31\%$) que `Regular / Proyecto` ($5.94\%$). Los datos indican que el comportamiento "highballer" es un fenómeno transversal del mercado de autos clásicos y no un rasgo exclusivo de los vendedores de proyectos.

---

## Prueba de Hipótesis 2: Variables Omitidas y Calidad Oculta

Para contrastar el argumento de la Práctica 5 sobre las variables omitidas en los atributos numéricos, se crearon indicadores binarios de presencia de atributos clave en el texto crudo y se evaluó su asociación con el precio mediante la **Prueba t de Welch** (asumiendo varianzas desiguales) y el **tamaño del efecto d de Cohen**:

$$t = \frac{\bar{X}_1 - \bar{X}_2}{\sqrt{\frac{s_1^2}{n_1} + \frac{s_2^2}{n_2}}} \quad\quad d = \frac{\bar{X}_1 - \bar{X}_2}{s_{\text{pooled}}}$$

### Comparativa de Precios por Atributo Textual (`grafica_6_impacto_precio_variables_omitidas.png`)

| Variable Textual Identificada | Con Frase ($n$) | Media Con ($) | Mediana Con ($) | Sin Frase ($n$) | Media Sin ($) | Diferencia ($) | Diferencia (%) | Estadístico $t$ | Valor $p$ | $d$ de Cohen |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`Numbers Matching`** | $269$ | $\$30,361.57$ | $\$27,000$ | $5,434$ | $\$15,281.34$ | **$+\$15,080.23$** | **$+98.7\%$** | $12.42$ | $1.36 \times 10^{-28}$ | **$0.98$ (Grande)** |
| **`Restored / Frame-off`** | $477$ | $\$30,006.70$ | $\$26,500$ | $5,226$ | $\$14,713.53$ | **$+\$15,293.17$** | **$+103.9\%$** | $11.61$ | $1.03 \times 10^{-27}$ | **$1.01$ (Grande)** |
| **`Documented / Build Sheet`**| $229$ | $\$25,224.95$ | $\$21,500$ | $5,474$ | $\$15,606.43$ | **$+\$9,618.52$** | **$+61.6\%$** | $5.78$ | $2.31 \times 10^{-8}$ | **$0.61$ (Moderado)**|
| **`Survivor`** | $83$ | $\$20,263.02$ | $\$16,900$ | $5,620$ | $\$15,929.58$ | **$+\$4,333.44$** | **$+27.2\%$** | $2.22$ | $2.89 \times 10^{-2}$ | **$0.27$ (Pequeño)** |
| **`Rust Free / No Rust`** | $508$ | $\$18,354.52$ | $\$14,900$ | $5,195$ | $\$15,761.69$ | **$+\$2,592.83$** | **$+16.5\%$** | $3.66$ | $2.76 \times 10^{-4}$ | **$0.16$ (Bajo)** |
| **`All Original`** | $294$ | $\$17,838.15$ | $\$14,500$ | $5,409$ | $\$15,892.34$ | **$+\$1,945.81$** | **$+12.2\%$** | $2.36$ | $1.90 \times 10^{-2}$ | **$0.12$ (Bajo)** |
| **`Barn Find`** | $29$ | $\$12,676.83$ | $\$7,500$ | $5,674$ | $\$16,009.60$ | **$-\$3,332.77$** | **$-20.8\%$** | $-1.65$ | $1.09 \times 10^{-1}$ | **$-0.21$ (Negativo)**|

---

### Aumento del Modelo D de la Práctica 5 con Indicadores Textuales

Para contrastar cuantitativamente si las variables textuales resuelven el sesgo por omisión de variables del modelo econométrico de precios, se incorporaron los indicadores textuales significativos al **Modelo D Múltiple Polinomial**:

$$\text{Modelo D Base: } \text{Price} = \beta_0 + \beta_1 \text{Year} + \beta_2 \text{Year}^2 + \beta_3 \text{Odometer} + \sum \delta_k D_k + \epsilon$$

$$\text{Modelo D Aumentado: } \text{Modelo D Base} + \gamma_1 \text{NumbersMatching} + \gamma_2 \text{Restored} + \gamma_3 \text{Documented} + \gamma_4 \text{RustFree}$$

### Comparativa de Métricas de Regresión OLS

| Métrica de Ajuste | Modelo D Original (Práctica 5) | Modelo D Aumentado con Texto (Práctica 9) | Variación Absoluta | Interpretación Estadística |
| :--- | :---: | :---: | :---: | :--- |
| **$R^2$ (Varianza Explicada)** | $0.2928$ | **$0.3322$** | **$+0.0394$ ($+3.94\%$)** | Incremento sustancial en capacidad explicativa. |
| **$R^2$ Ajustado ($R^2_{\text{adj}}$)** | $0.2919$ | **$0.3309$** | **$+0.0390$ ($+3.90\%$)** | Penaliza la inclusión de parámetros y valida mejora neta. |
| **AIC (Akaike Info Criterion)** | $124,473.1$ | **$124,153.6$** | **$-319.5$ puntos** | Reducción masiva ($\Delta \text{AIC} > 10$ es evidencia abrumadora). |
| **BIC (Bayesian Info Criterion)** | $124,526.3$ | **$124,233.4$** | **$-292.9$ puntos** | Fuerte soporte parsimonioso a favor del modelo con texto. |
| **RMSE (Error Residual Estándar)**| $\$13,270.68$ | **$\$12,899.58$** | **$-\$371.10$ USD** | Menor dispersión residual en predicciones. |

#### Coeficientes Parciales de las Variables Textuales (ceteris paribus):
* **`Numbers Matching`:** Coeficiente $= \mathbf{+\$8,253.28\text{ USD}}$ ($p = 5.18 \times 10^{-23}$). Manteniendo idénticos el año, el odómetro, la carrocería, el título y los cilindros, la presencia de coincidencia de números agrega más de $\$8,200$ USD al precio ofertado.
* **`Restored / Frame-off`:** Coeficiente $= \mathbf{+\$8,051.93\text{ USD}}$ ($p = 8.72 \times 10^{-35}$). El indicador de restauración profesional aporta una prima de más de $\$8,000$ USD.
* **`Documented / Build Sheet`:** Coeficiente $= \mathbf{+\$6,411.18\text{ USD}}$ ($p = 3.64 \times 10^{-13}$). La presencia de documentación histórica comprobable añade más de $\$6,400$ USD.
* **`Rust Free / No Rust`:** Coeficiente $= +\$229.10\text{ USD}$ ($p = 0.706$). Al controlar conjuntamente por año y condición física, la mención de libre de óxido pierde significancia estadística como predictor independiente.

---

## Limitaciones Metodológicas Identificadas

1. **Ausencia de Verificación de Veracidad:** La aparición de un término como "numbers matching" o "rust free" en Craigslist refleja una afirmación del vendedor, no un peritaje mecánico verificado; vendedores particulares pueden exagerar el estado real.
2. **Contextos de Negación:** La búsqueda por expresiones regulares captura cadenas como "original", pero podría incluir ocurrencias aisladas de negación ("not original engine") sin un analizador de dependencias sintácticas exhaustivo.
3. **Muestra Reducida en Categorías Específicas:** Términos como `barn_find` registran solo $29$ ocurrencias en $5,703$ anuncios ($0.51\%$), lo que reduce la potencia estadística del test t ($p = 0.109$) a pesar de mostrar una diferencia media observable de $-\$3,332$ USD.

---

## Catálogo de Gráficas Exportadas

1. **`grafica_1_diagnostico_longitud_descripciones.png`:** Diagnóstico exploratorio en dos paneles: panel A (boxplot de palabras por segmento) y panel B (histograma y densidad global hasta el percentil 95).
2. **`grafica_2_nube_palabras_global.png`:** Nube de palabras global de los $5,703$ anuncios tras preprocesamiento, filtrado de números de ingeniería y lematización.
3. **`grafica_3_nubes_palabras_por_segmento.png`:** Mosaico comparativo 2x2 con nubes de palabras individuales para los 4 segmentos de condición.
4. **`grafica_4_frecuencia_palabras_y_bigramas.png`:** Gráficas de barras con el top 12 de palabras lematizadas y top 12 de bigramas más frecuentes.
5. **`grafica_5_prevalencia_lenguaje_highballer.png`:** Gráfica de barras comparativa del porcentaje de anuncios con frases de precio firme/intransigente en cada segmento.
6. **`grafica_6_impacto_precio_variables_omitidas.png`:** Comparativa visual de precios medios con y sin frases clave de valor en la descripción.
