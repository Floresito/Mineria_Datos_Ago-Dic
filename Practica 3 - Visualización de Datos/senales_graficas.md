# Señales Visuales y Hallazgos Cuantitativos (Práctica 3)

**Materia:** Minería de Datos  
**Institución:** Universidad Autónoma de Nuevo León (UANL) - Facultad de Ciencias Físico-Matemáticas (FCFM)  
**Semestre:** 7mo Semestre, Licenciatura en Ciencias de la Computación  
**Dataset Base:** `muscle_cars_clean.csv` (5,703 registros)  
**Propósito:** Registro formal de patrones visuales atípicos, dispersiones observadas y datos duros de soporte para la formulación de hipótesis en el Reporte Final (Semana 15).  
**Preguntas de Investigación:** Las preguntas guía detonadoras se encuentran desacopladas en [pendientes_investigacion_pia.md](file:///c:/dev/facu/Mineria_Datos_Ago-Dic/Practica%203%20-%20Visualizaci%C3%B3n%20de%20Datos/pendientes_investigacion_pia.md).

---

### Señal 1: Morfología de Distribución en Histogramas y Facetado Temporal
- **Dato Duro Visual (Gráficas 1A y 1B):**
  - El histograma univariado global de `price` exhibe una concentración pronunciada en el intervalo de \$3,000 a \$15,000, con una cola derecha asimétrica que se prolonga hasta \$225,000.
  - Al observar el histograma facetado por década (Gráfica 1B), en el estrato **1960–1969** la masa de datos se dispersa ampliamente hacia la derecha con una curva KDE aplanada y una mediana de \$21,500. En contraste, en el estrato **1989–2004** la distribución colapsa en un pico agudo y concentrado por debajo de los \$10,000 (mediana de \$8,591).

---

### Señal 2: Patrón de Outliers por Fabricante en Diagramas de Caja
- **Dato Duro Visual (Gráfica 2A):**
  - `dodge` presenta la mediana más alta entre los fabricantes principales (\$16,500) y un rango intercuartílico amplio (IQR = \$19,037.50).
  - `chevrolet` concentra la mayor densidad y dispersión de valores atípicos superiores (*outliers*), con registros que superan holgadamente el límite del bigote superior (\$45,812.50) y alcanzan cotizaciones entre \$80,000 y \$225,000.

---

### Señal 3: Dispersión en la Categoría 'Unknown' de Condición
- **Dato Duro Visual (Gráfica 2B):**
  - Aunque la categoría `excellent` exhibe una mediana superior a `good` y `fair`, la categoría `unknown` (vehículos donde el vendedor no especificó condición en Craigslist) abarca un rango de precios que va desde el mínimo del dataset (\$500) hasta valores atípicos superiores a los \$100,000, con un IQR de \$17,500 comparable al de las categorías intermedias.

---

### Señal 4: Dispersión Vertical y No Linealidad en Precio vs. Odómetro
- **Dato Duro Visual (Gráfica 3A):**
  - La recta de regresión de tendencia exhibe una pendiente negativa suave a lo largo del eje de kilometraje. Sin embargo, la dispersión vertical de los puntos es sumamente alta: se registran vehículos con más de 100,000 millas cotizados por encima de \$30,000–\$50,000, así como vehículos con lecturas bajas (< 30,000 millas) cotizados por debajo de los \$10,000.

---

### Señal 5: Estructura de Inflexión No Lineal en Precio vs. Año de Fabricación
- **Dato Duro Visual (Gráfica 3B):**
  - El diagrama de dispersión de `price` vs. `year` muestra una silueta en forma de curva o quiebre estructural: cotizaciones elevadas y altamente dispersas entre 1960 y 1972, una caída acelerada entre 1973 y 1978, un piso de precios plano y compacto entre 1980 y 1996, y una ligera reapertura de dispersión hacia 2002–2004.

---

### Señal 6: Trayectoria Sincronizada en Líneas de Tendencia Temporal Interanual
- **Dato Duro Visual (Gráfica 5):**
  - El gráfico longitudinal de líneas revela que entre 1964 y 1970, Chevrolet, Ford y Pontiac presentan picos medios sincronizados en el rango de \$25,000 a \$35,000 USD con bandas de confianza amplias. A partir de 1973–1974, las curvas de las tres marcas caen de manera paralela y abrupta hacia el umbral de \$10,000–\$15,000, convergiendo en una banda estrecha a lo largo de las décadas de 1980 y 1990.

---

### Señal 7: Densidad Probabilística y Bimodalidad en Diagramas de Violín
- **Dato Duro Visual (Gráfica 6):**
  - El violín correspondiente a la década **1960–1969** exhibe un cuerpo vertical alargado con dos ensanchamientos notables en su densidad KDE (uno en la zona media-baja de \$15,000 y otro en la zona superior de \$30,000–\$45,000).
  - En contraste, los violines de **1980–1988** y **1989–2004** presentan una base ancha y bulbosa fuertemente concentrada cerca de los \$6,000–\$8,000 con una cúspide muy estrecha.
