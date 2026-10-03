## Objetivo y Planteamiento del Problema

El objetivo de esta práctica es evaluar la capacidad de un clasificador **K-Nearest Neighbors (KNN)** para predecir la **condición reportada del vehículo (`condition`)** utilizando exclusivamente sus atributos cuantitativos continuos y discretos:
$$\text{Features: } X = [\text{Price, Odometer, Year, Cylinders}]$$

### Lógica de Negocio y Conexión con la Investigación
Si un comprador en el mercado digital solo dispusiera de los datos numéricos tabulares (sin acceso a fotografías ni al texto libre del anuncio), ¿qué tan bien puede inferir el estado físico real del automóvil? 

Este experimento se conecta directamente con la hipótesis de la Práctica 5: si las variables tabulares tradicionales (`price`, `year`, `odometer`) no logran discriminar limpiamente la condición del auto, se aporta evidencia cuantitativa de que el precio pedido (*asking price*) no refleja mecánicamente el estado del vehículo y que los determinantes cualitativos clave residen en el texto de la descripción (puente hacia la **Práctica 9: NLP**).

---

## Preparación de la Variable Objetivo y Partición de Datos

### Estructuración de la Variable Objetivo (3 Clases)
Para garantizar un soporte muestral robusto y representativo en cada clase, se estructuraron las etiquetas conocidas ($n = 4,091$, excluyendo los 1,612 registros `'unknown'`) en **3 clases ordinales**:

1. **`Excelente`** (`new`, `like new`, `excellent`): $n = 2,162$ ($52.85\%$)
2. **`Buena`** (`good`): $n = 1,542$ ($37.69\%$)
3. **`Regular / Proyecto`** (`fair`, `salvage`): $n = 387$ ($9.46\%$)

### Partición Estratificada Train / Test (80 / 20)
Para preservar con exactitud la proporción de clases en ambos subconjuntos, se aplicó una partición estratificada:
- **Conjunto de Entrenamiento (*Train*):** $3,272$ observaciones ($80\%$).
- **Conjunto de Prueba (*Test*):** $819$ observaciones ($20\%$).

---

## Justificación Teórica: Estandarización y Ponderación por Distancia

### Estandarización de Features (`StandardScaler`)
El algoritmo KNN clasifica una nueva observación calculando la **Distancia Euclidiana** en el espacio multidimensional:

$$d(p, q) = \sqrt{\sum_{j=1}^{m} (p_j - q_j)^2}$$

- Si utilizáramos los datos crudos, una diferencia de $20,000$ millas en `odometer` o $\$10,000$ USD en `price` generaría distancias de millones en la suma de cuadrados, aplastando por completo la variación de `year` ($1960 - 2004$) o de `cylinders` ($4 - 8$).
- Mediante `StandardScaler`, cada característica se transforma a escala $Z$:
  $$Z = \frac{X - \mu}{\sigma}$$
  centrando la media en $0$ y la desviación estándar en $1$, garantizando que todas las dimensiones pesen equitativamente en la métrica de distancia.
- *Prevención de fuga de datos (Data Leakage):* El escalador se ajusta (`fit`) exclusivamente sobre el conjunto de entrenamiento y se aplica (`transform`) sobre el conjunto de prueba.

### Ponderación por el Inverso de la Distancia (`weights='distance'`)
En lugar del esquema tradicional de votación uniforme (donde cada uno de los $K$ vecinos emite un voto idéntico sin importar su cercanía), se implementó una ponderación inversamente proporcional a la distancia:

$$w_i = \frac{1}{d(p, q_i)}$$

- Mitiga el sesgo hacia las clases mayoritarias en regiones de densidad mixta, incrementando la precisión general de $64.47\%$ a $70.45\%$ y mejorando la sensibilidad sobre la clase minoritaria.
- En un inicio se probó la votación uniforme (sin ponderación), obteniendo un accuracy de 64.47\%, pero al aplicar la ponderación por distancia se obtuvo un accuracy de 70.45\%, asi que se decidió quedarse con la ponderacion por distancia.

---

## Selección del Hiperparámetro $K$ y Curva de Rendimiento

Se evaluó el clasificador en un rango exhaustivo de **$K = 1$ hasta $K = 25$** (`grafica_1_curva_k_accuracy.png`):

### Trade-off Sesgo-Varianza en KNN
- **$K$ muy bajo ($K = 1$):** El modelo tiene baja polarización (*bias*) pero altísima varianza. Memoriza el ruido local y los valores atípicos del conjunto de entrenamiento ($Train\text{ Accuracy} = 100.0\%$), pero generaliza con menor precisión sobre datos no vistos.
- **$K$ muy alto ($K > 20$):** Aumenta el sesgo al suavizar en exceso las fronteras de decisión y capturar vecinos lejanos no representativos.
- **$K$ Óptimo:** La curva de validación sobre el conjunto de prueba alcanza su punto máximo en **$K = 15$**, con una **Precisión Global (*Accuracy*) del $70.45\%$** (superando el $64.47\%$ obtenido con votación uniforme).

---

## Evaluación del Modelo Final ($K = 15$ con Ponderación por Distancia)

### Reporte de Métricas por Clase

| Clase Objetivo | Precisión (*Precision*) | Sensibilidad (*Recall*) | Puntuación F1 (*F1-Score*) | Soporte en Test ($n$) |
| :--- | :---: | :---: | :---: | :---: |
| **`Excelente`** | `0.7702` ($77.02\%$) | `0.8360` ($83.60\%$) | **`0.8018`** | $433$ |
| **`Buena`** | `0.6215` ($62.15\%$) | `0.6375` ($63.75\%$) | **`0.6294`** | $309$ |
| **`Regular / Proyecto`** | `0.5625` ($56.25\%$) | `0.2338` ($23.38\%$) | **`0.3303`** | $77$ |
| **Promedio Global / Macro** | **`0.6514`** | **`0.5691`** | **`0.5871`** | **$819$** |
| **Promedio Ponderado (*Weighted*)** | **`0.6946`** | **`0.7045`** | **`0.6924`** | **$819$** |

### Diagnóstico de la Matriz de Confusión

La matriz de confusión (`grafica_2_matriz_confusion_knn.png`) muestra el desglose exacto de aciertos y errores:

$$\begin{pmatrix} 
\text{Real \ Predicha} & \textbf{Excelente} & \textbf{Buena} & \textbf{Regular / Proyecto} \\
\textbf{Excelente (433)} & \mathbf{362 \ (83.6\%)} & 69 \ (15.9\%) & 2 \ (0.5\%) \\
\textbf{Buena (309)} & 100 \ (32.4\%) & \mathbf{197 \ (63.8\%)} & 12 \ (3.9\%) \\
\textbf{Regular / Proyecto (77)} & 8 \ (10.4\%) & 51 \ (66.2\%) & \mathbf{18 \ (23.4\%)}
\end{pmatrix}$$

#### Datos Duros de los Errores de Clasificación:
1. **Confusión Cruzada entre `Excelente` y `Buena`:** El modelo confunde $100$ autos en condición `Buena` clasificándolos como `Excelente` ($32.4\%$), y $69$ autos `Excelente` clasificándolos como `Buena` ($15.9\%$).
2. **Dificultad Persistente en `Regular / Proyecto`:** Aunque el recall mejoró del $6.5\%$ al $23.4\%$ gracias a la ponderación por distancia, de los $77$ vehículos en condición regular o proyecto en el conjunto de prueba, el modelo **solo identificó correctamente a 18** ($Recall = 23.38\%$). La mayoría ($51$ autos, $66.2\%$) siguió siendo clasificada erróneamente como `Buena` y $8$ ($10.4\%$) como `Excelente`.

---

## Casos de Desacuerdo con Alta Confianza del Modelo

Se identificaron casos representativos donde el modelo KNN emitió una predicción con **confianza del $100\%$** basada en la proximidad inmediata de sus vecinos tabulares, pero la condición real reportada en el anuncio difería radicalmente:

| Modelo de Auto | Año | Precio (USD) | Odómetro (mi) | Cilindros | Condición Real | Condición Predicha por KNN | Confianza del Modelo |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- | :---: |
| **Ford Thunderbird** | 1965 | \$4,500 | 77,000 | 8.0 | `Regular / Proyecto` | **`Buena`** | **100.0%** |
| **Ford Mustang SVT Cobra** | 1996 | \$12,500 | 54,000 | 8.0 | `Buena` | **`Excelente`** | **100.0%** |
| **Chevrolet Camaro** | 1969 | \$36,500 | 6,000 | 8.0 | `Buena` | **`Excelente`** | **100.0%** |
| **Chevrolet Camaro Z28** | 1998 | \$14,900 | 30,000 | 8.0 | `Buena` | **`Excelente`** | **100.0%** |
| **Pontiac Firebird** | 1998 | \$26,995 | 33,830 | 8.0 | `Buena` | **`Excelente`** | **100.0%** |
| **Chevrolet Corvette** | 2004 | \$27,975 | 7,870 | 8.0 | `Buena` | **`Excelente`** | **100.0%** |
| **Chevrolet Corvette** | 1977 | \$18,900 | 36,000 | 8.0 | `Buena` | **`Excelente`** | **100.0%** |

---

## Catálogo de Gráficas Exportadas

1. **`grafica_1_curva_k_accuracy.png`:** Curva comparativa de precisión (Train vs. Test) para valores de $K$ entre $1$ y $25$ con ponderación por distancia, señalando el $K=15$ óptimo.
2. **`grafica_2_matriz_confusion_knn.png`:** Mapa de calor de la matriz de confusión normalizada y con conteos absolutos.
