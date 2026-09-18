# Práctica 4: Pruebas Estadísticas

## 1. Introducción y Selección de Pruebas

En esta práctica evaluamos de manera formal si las diferencias de precio observadas en las prácticas anteriores (Estadística Descriptiva y Visualización) son estadísticamente reales o si pudieron deberse al azar, utilizando una muestra de **5,703 vehículos**.

### ¿Qué pruebas utilizamos y por qué?
1. **ANOVA (Comparación de 3 o más grupos):**
   - La usamos para comparar precios entre **fabricantes** (Chevrolet, Ford, Pontiac, Dodge, Buick, Mercury) y entre **décadas históricas**.
   - *¿Por qué no hacer muchas pruebas t por parejas?* Porque al hacer muchas pruebas separadas, el riesgo de cometer un falso positivo (encontrar una diferencia que no existe) se acumula. El ANOVA evalúa primero si existe alguna diferencia global entre todos los grupos.
2. **Prueba $t$ (Comparación de 2 grupos):**
   - La usamos para contrastes directos entre dos categorías:
     - Título vehicular: `Clean` vs. `Salvage/Rebuilt`.
     - Condición del auto: `Condición Alta` (excelente/bueno/como nuevo) vs. `Condición Baja` (regular/salvage).
     - Tipo de carrocería: `Convertible` vs. `Coupe` (retomando la señal de la Práctica 2).
3. **Nivel de significancia ($\alpha = 0.05$):**
   - Fijamos un umbral del **5%**. Si el $p\text{-value}$ es menor a $0.05$, rechazamos la hipótesis nula ($H_0$) y concluimos que la diferencia observada es estadísticamente significativa con un 95% de confianza.

---

## 2. Verificación Visual de Supuestos y Enfoque Robusto

Para aplicar estas pruebas de forma confiable ante un profesor o supervisor, revisamos dos aspectos clave:

### 2.1 Normalidad (Inspección Visual y Teorema del Límite Central)
- **Inspección visual:** Al revisar los histogramas (vistos en la Práctica 3) y los gráficos Q-Q generados (`grafica_6_diagnostico_supuestos_qqplots.png`), los precios muestran una forma con cola larga hacia la derecha (asimetría positiva), típica de los mercados de autos donde existen piezas raras de colección muy caras.
- **Justificación teórica:** Gracias al **Teorema del Límite Central**, cuando trabajamos con muestras grandes ($n > 30$ por grupo, y aquí tenemos cientos o miles por categoría), la distribución de los promedios muestrales se comporta de manera normal. Por lo tanto, las pruebas paramétricas basadas en promedios son completamente válidas y confiables.

### 2.2 Varianzas Desiguales y Uso de Variantes Robustas (Welch)
- En datos reales de precios de autos clásicos, unos grupos son mucho más dispersos que otros (por ejemplo, los autos clásicos de los 60s varían entre \$500 y \$200,000, mientras que los de los 80s están más concentrados).
- **Decisión metodológica:** En lugar de asumir que todos los grupos tienen la misma varianza (homocedasticidad), utilizamos directamente las **variantes robustas de Welch**:
  - **Prueba $t$ de Welch (`equal_var=False`):** Ajusta los grados de libertad matemáticamente según la dispersión de cada grupo.
  - **ANOVA de Welch:** Realiza una comparación ponderada que no se distorsiona por las diferencias de varianza entre grupos.

---

## 3. Tabla Resumen de Resultados

A continuación se resumen las 5 pruebas realizadas. En todas se evaluó la variable de **Precio (`price`)**:

| N° | Comparación Realizada | Tipo de Prueba | Estadístico de Prueba | Grados de Libertad ($df$) | $p\text{-value}$ | Decisión ($\alpha = 0.05$) | Tamaño del Efecto | Interpretación del Efecto |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **1** | **Por Fabricante** (6 marcas principales) | ANOVA de Welch | $F = 26.13$ | $(5, 357.3)$ | $1.69 \times 10^{-22}$ | **Rechazar $H_0$** | $\eta^2 = 0.0184$ | Pequeño (explica el 1.84% del precio) |
| **2** | **Por Época Histórica** (5 décadas) | ANOVA de Welch | $F = 195.77$ | $(4, 1438.9)$ | $4.37 \times 10^{-134}$ | **Rechazar $H_0$** | $\eta^2 = 0.1761$ | **Grande** (explica el 17.61% del precio) |
| **3** | **Título:** Clean vs. Salvage/Rebuilt | Prueba $t$ de Welch | $t = 11.36$ | $134.2$ | $2.34 \times 10^{-21}$ | **Rechazar $H_0$** | $d = 0.5274$ | Mediano (Castigo de -\$8,322 USD) |
| **4** | **Condición:** Alta vs. Baja | Prueba $t$ de Welch | $t = 32.76$ | $1552.6$ | $2.16 \times 10^{-179}$ | **Rechazar $H_0$** | $d = 0.7719$ | Mediano a Grande (Brecha de -\$11,367 USD) |
| **5** | **Carrocería:** Convertible vs. Coupe | Prueba $t$ de Welch | $t = 5.43$ | $1676.2$ | $6.64 \times 10^{-08}$ | **Rechazar $H_0$** | $d = 0.2320$ | Pequeño (Prima de +\$3,210 USD) |

---

## 4. Interpretación Detallada de Cada Prueba

### 4.1 Prueba 1: Precios entre Fabricantes
- **Hipótesis:**
  - $H_0:$ El precio promedio es igual en todas las marcas ($\mu_{\text{chevy}} = \mu_{\text{ford}} = \dots = \mu_{\text{mercury}}$).
  - $H_1:$ Al menos una marca tiene un precio promedio diferente.
- **Resultado:** $p\text{-value} = 1.69 \times 10^{-22} < 0.05 \rightarrow$ Se rechaza $H_0$.
- **Comparación por parejas (Tukey HSD):**
  - **Dodge** ($\bar{x} = \$20,197$) y **Chevrolet** ($\bar{x} = \$17,801$) tienen precios significativamente más altos que Ford ($\bar{x} = \$13,932$), Pontiac ($\bar{x} = \$15,055$) y Mercury ($\bar{x} = \$9,819$).
  - Entre Ford y Pontiac no hay diferencia significativa ($p = 0.7212$), ni entre Chevrolet y Dodge ($p = 0.5014$).
- **Tamaño del efecto:** $\eta^2 = 0.0184$. Aunque la diferencia es real, la marca solo explica el **1.84%** de la variación del precio. El modelo concreto, su estado y el año pesan mucho más.

---

### 4.2 Prueba 2: Precios entre Décadas Históricas
- **Hipótesis:**
  - $H_0:$ El precio promedio es el mismo en todas las épocas históricas.
  - $H_1:$ Al menos una época tiene un precio promedio significativamente distinto.
- **Resultado:** $p\text{-value} = 4.37 \times 10^{-134} < 0.05 \rightarrow$ Se rechaza $H_0$.
- **Comparación por parejas (Tukey HSD):**
  - Se comprueba el **quiebre histórico de 1974**: La era clásica 1960–1969 ($\bar{x} = \$26,049$) y 1970–1974 ($\bar{x} = \$22,968$) superan por amplio margen a los autos de 1975–1979 ($\bar{x} = \$13,989$), con una caída estadísticamente significativa de casi \$9,000 USD ($p < 0.0001$).
  - **Estancamiento de la depreciación:** Los autos de 1980–1988 ($\bar{x} = \$11,054$) y los de 1989–2004 ($\bar{x} = \$11,016$) tienen una diferencia promedio de apenas **\$38.69 USD**, la cual **no es significativa** ($p = 1.0000$).
- **Tamaño del efecto:** $\eta^2 = 0.1761$. La época histórica explica el **17.61%** de la varianza del precio (un efecto grande).

---

### 4.3 Prueba 3: Título Limpio (`Clean`) vs. Dañado (`Salvage/Rebuilt`)
- **Hipótesis:**
  - $H_0:$ No hay diferencia de precio entre autos con título limpio y con título de siniestro/reconstruido ($\mu_{\text{clean}} = \mu_{\text{salvage}}$).
  - $H_1:$ Existe diferencia en el precio promedio entre ambos grupos.
- **Resultado:** $t(134.2) = 11.36, p\text{-value} = 2.34 \times 10^{-21} < 0.05 \rightarrow$ Se rechaza $H_0$.
- **Valores observados:** Promedio Clean = **\$16,302.51** vs. Promedio Salvage/Rebuilt = **\$7,980.35**.
- **Interpretación:** Un título con historial de daño castiga el precio en promedio en **-\$8,322.16 USD** (una reducción del 51%), con un tamaño de efecto mediano ($d = 0.5274$).

---

### 4.4 Prueba 4: Condición Conservada (`Alta`) vs. Proyecto/Dañado (`Baja`)
- **Hipótesis:**
  - $H_0:$ Los autos en buena condición tienen el mismo precio promedio que los de baja condición.
  - $H_1:$ Los autos en buena condición tienen un precio promedio diferente.
- **Resultado:** $t(1552.6) = 32.76, p\text{-value} = 2.16 \times 10^{-179} < 0.05 \rightarrow$ Se rechaza $H_0$.
- **Valores observados:** Promedio Condición Alta = **\$16,246.03** vs. Condición Baja = **\$4,879.05**.
- **Interpretación:** La diferencia media es de **\$11,366.98 USD** a favor de los vehículos bien conservados, con un tamaño de efecto considerable ($d = 0.7719$, cercano a efecto grande).

---

### 4.5 Prueba 5: Variantes Descapotables (`Convertible`) vs. Techo Cerrado (`Coupe`)
- **Hipótesis:**
  - $H_0:$ Los autos descapotables y coupés tienen el mismo precio promedio.
  - $H_1:$ Los convertibles tienen un precio promedio diferente a los coupés.
- **Resultado:** $t(1676.2) = 5.43, p\text{-value} = 6.64 \times 10^{-08} < 0.05 \rightarrow$ Se rechaza $H_0$.
- **Valores observados:** Promedio Convertible = **\$18,421.10** vs. Promedio Coupe = **\$15,211.08**.
- **Interpretación:** Se confirma la señal de la Práctica 2: existe una **prima de colección de +\$3,210.02 USD** (+21.1%) para los descapotables frente a las versiones cerradas, con un tamaño de efecto $d = 0.2320$.

---

## 5. Gráficas Generadas en la Práctica

Todas las visualizaciones de apoyo se encuentran guardadas en la carpeta `graficas/` a 300 DPI:

1. **`grafica_1_anova_price_por_fabricante.png`:** Diagrama de cajas por marca con sus medias aritméticas (rombos rojos) y el resultado del ANOVA de Welch.
2. **`grafica_2_anova_price_por_decada.png`:** Diagrama de cajas que ilustra el escalón de precios entre las diferentes décadas históricas.
3. **`grafica_3_ttest_price_por_title_status.png`:** Comparación visual del precio entre títulos Clean y Salvage/Rebuilt.
4. **`grafica_4_ttest_price_por_condicion.png`:** Comparación del precio entre condición alta y baja.
5. **`grafica_5_ttest_price_convertible_vs_coupe.png`:** Comparación del precio entre carrocerías Convertible y Coupe.
6. **`grafica_6_diagnostico_supuestos_qqplots.png`:** Panel de gráficos Q-Q para la inspección visual de las distribuciones.
