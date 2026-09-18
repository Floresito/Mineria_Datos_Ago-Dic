# Señales Comprobadas y Preguntas Guía para la Narrativa del PIA (Práctica 4)

### Señales Comprobadas Estadísticamente en la Práctica 4

- **Confirmación del Quiebre Estructural Post-1974:**
  - El ANOVA de épocas ($F = 304.41, p < 10^{-237}, \eta^2 = 0.1761$) y la prueba post-hoc de Tukey HSD confirmaron que la caída de precio medio entre 1970–1974 y 1975–1979 es de **\$8,978.98 USD** ($p < 0.0001$), demostrando que el escalón de cotización es un efecto real y no fruto del azar muestral.
  - Se confirmó que no existe diferencia estadística ($p = 1.0000$, diferencia de apenas **\$38.69 USD**) entre la década de 1980–1988 y 1989–2004, ratificando la anulación de la depreciación temporal en vehículos de más de 20 años de antigüedad.

- **Diferenciación de Marcas:**
  - El ANOVA por fabricante ($F = 17.41, p < 10^{-16}, \eta^2 = 0.0184$) demostró que Dodge y Chevrolet comandan precios medios significativamente superiores a Ford y Pontiac, pero con un tamaño de efecto pequeño ($\eta^2 < 0.02$), lo que indica que el fabricante por sí solo no explica la mayor parte del precio: son el modelo específico, el año y el estado mecánico los que determinan el valor.

- **Cuantificación del Castigo por Título y Condición:**
  - La prueba $t$ de Welch confirmó un castigo medio de **-\$8,322.16 USD** ($d = 0.5274$) en vehículos con título 'salvage/rebuilt', y una brecha de **-\$11,366.98 USD** ($d = 0.7719$) entre autos en condición de proyecto vs autos conservados.

- **Prima de Colección en Carrocerías Descapotables:**
  - La prueba $t$ de Student confirmó la existencia de una prima estadísticamente significativa de **+\$3,210.02 USD** ($t = 5.74, p < 10^{-7}, d = 0.2320$) para los modelos `convertible` frente a los `coupe`.

---

### Preguntas Guía para las Siguientes Prácticas (Modelos Lineales, Clustering y Clasificación)

- ¿Cómo interactúa el efecto temporal (año de fabricación) con el tipo de carrocería (convertible vs coupe) en un modelo de regresión multivariada: la prima de descapotable aumenta exponencialmente a mayor antigüedad?

- Si el fabricante solo explica el $1.84\%$ de la varianza en precio, ¿cuánto porcentaje adicional de varianza ($R^2$) aportan variables continuas como el odómetro y variables categóricas como modelo e historial de título al combinarse en un modelo lineal múltiple (Práctica 5)?

- ¿Podrá un algoritmo de agrupamiento no supervisado (Clustering en Práctica 7) segmentar de forma natural el mercado entre "autos de proyecto para restaurar" y "piezas de exhibición restauradas", superando la limitación de la asimetría bimodal observada en los años 60?
