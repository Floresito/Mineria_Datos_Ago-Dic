# Diagnóstico Inicial del Dataset y Decisiones de Limpieza

## 1. Diagnóstico Inicial del Dataset Completo (Pre-Filtrado)

### 1.1 Estructura y Memoria
- **Total de registros (filas):** `426,880`
- **Total de atributos (columnas):** `26`
- **Uso en memoria RAM (crudo en Pandas):** `~4,255.72 MB`

### 1.2 Integridad y Tipos de Datos Iniciales

| Columna | Tipo de Dato Original | Tipo de Dato Objetivo | Registros No Nulos | Registros Nulos | % de Nulos |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `id` | `int64` | `int64` | 426,880 | 0 | 0.00% |
| `url` | `object` | *Eliminar* | 426,880 | 0 | 0.00% |
| `region` | `object` | `string` / `category` | 426,880 | 0 | 0.00% |
| `region_url` | `object` | *Eliminar* | 426,880 | 0 | 0.00% |
| `price` | `int64` | `float64` | 426,880 | 0 | 0.00% |
| `year` | `float64` | `int64` | 425,675 | 1,205 | 0.28% |
| `manufacturer`| `object` | `category` | 409,234 | 17,646 | 4.13% |
| `model` | `object` | `string` | 421,603 | 5,277 | 1.24% |
| `condition` | `object` | `category` | 252,776 | 174,104 | 40.79% |
| `cylinders` | `object` | `category` | 249,202 | 177,678 | 41.62% |
| `fuel` | `object` | *Eliminar* | 423,867 | 3,013 | 0.71% |
| `odometer` | `float64` | `float64` | 422,480 | 4,400 | 1.03% |
| `title_status`| `object` | `category` | 418,638 | 8,242 | 1.93% |
| `transmission`| `object` | `category` | 424,324 | 2,556 | 0.60% |
| `VIN` | `object` | *Transformar a `is_vin_missing`* | 265,838 | 161,042 | 37.73% |
| `drive` | `object` | `category` | 296,313 | 130,567 | 30.59% |
| `size` | `object` | `category` | 120,519 | 306,361 | 71.77% |
| `type` | `object` | `category` | 334,022 | 92,858 | 21.75% |
| `paint_color` | `object` | `category` | 296,677 | 130,203 | 30.50% |
| `image_url` | `object` | *Eliminar* | 426,812 | 68 | 0.02% |
| `description` | `object` | `string` | 426,810 | 70 | 0.02% |
| `county` | `float64` | *Eliminar* | 0 | 426,880 | 100.00% |
| `state` | `object` | `category` | 426,880 | 0 | 0.00% |
| `lat` | `float64` | `float64` | 420,331 | 6,549 | 1.53% |
| `long` | `float64` | `float64` | 420,331 | 6,549 | 1.53% |
| `posting_date`| `object` | `datetime64[ns, UTC]` | 426,812 | 68 | 0.02% |

### 1.3 Duplicados
- **Duplicados por ID (`id`):** `0` (las claves primarias asignadas por Craigslist son únicas).
- **Duplicados por contenido exacto (excluyendo metadatos de URL/ID):** `20` registros en el dataset global.

### 1.4 Distribución Numérica Cruda y Outliers de Captura
- **`price`:** Rango observado de \$0 a \$3,736,928,711. El 5% inferior registra \$0 (técnicas de captación de vendedores en Craigslist como precios de enganche), mientras que los máximos corresponden a errores tipográficos sistemáticos.
- **`odometer`:** Rango observado de 0.0 a 10,000,000.0 millas.
- **`year`:** Rango de 1900 a 2022.

### 1.5 Cobertura Temporal de `posting_date`
- **Formato:** ISO 8601 con zona horaria (`YYYY-MM-DDTHH:MM:SS-offset`).
- **Fecha Mínima:** 2021-04-04 07:00:25 UTC.
- **Fecha Máxima:** 2021-05-05 04:24:09 UTC.
- **Cobertura:** 32 días continuos de publicaciones diarias (Abril 2021: 313,144 registros; Mayo 2021: 113,668 registros).

---

## 2. Filtrado Temático a Muscle Cars (Decisión de Alcance)

### Decisión Tomada: Ampliación de Rango Temporal (1960 - 2004) y Filtrado por Taxonomía de Modelos
Para garantizar la representatividad estadística y superar el umbral mínimo establecido de **5,000 registros**, se implementó la **Opción A**:
- **Rango de Años:** `1960 <= year <= 2004`.
  - Justificación técnica: Este intervalo abarca la época dorada de los muscle cars clásicos (1964–1974), la etapa de transición y resurgimiento automotriz (1975–1988, incorporando Fox-body Mustangs, 3ra generación de Camaros/Firebirds y el Buick Grand National/GNX) y la 4ta generación / SN95 (hasta 2004, antes del inicio del diseño retro moderno en 2005).
- **Taxonomía de Búsqueda de Modelos:**
  Debido a que el scraper original de Craigslist solo disponía de fabricantes preestablecidos (dejando marcas clave como Plymouth, Oldsmobile o AMC fuera de la columna `manufacturer`), se aplicó una expresión regular exhaustiva sobre el campo de texto `model`:
  `mustang|camaro|corvette|chevelle|charger|challenger|gto|firebird|trans am|road runner|barracuda|cuda|cutlass|442|grand national|javelin|amx|torino|cougar|nova|monza|dart|demon|super bee|fury|bel air|impala|monte carlo|el camino|riviera|skylark|grand prix|galaxie|thunderbird|plymouth|oldsmobile|amc|lemans|tempest|corvair|cyclone|fairlane`
- **Total de registros obtenidos tras filtrado:** **`5,911 filas`**.

---

## 3. Tratamiento de Valores Faltantes, Outliers y Transformaciones

### 3.1 Tratamiento por Variable

1. **`price` (Variable Objetivo):**
   - *Diagnóstico Inicial:* En el dataset crudo y en el subconjunto filtrado **no existen valores nulos reales (`NaN = 0`)** debido a que Craigslist almacena el campo como entero. Sin embargo, se identificaron **208 registros con errores evidentes de captura / valores anómalos** (117 con `price == $0`, 87 con `0 < price < $500` correspondientes a precios señuelo o de enganche, y 4 con `price > $500,000` correspondientes a errores tipográficos).
   - *Tratamiento:* **Eliminación directa de registros con `price < $500` o `price > $500,000`**.
   - Al tratarse de la variable objetivo dependiente ($Y$), imputar valores sintéticos (como la mediana jerárquica)introducción un riesgo inherente de **circularidad y reducción artificial de la varianza**, afectando directamente los modelos predictivos posteriores (regresión lineal y KNN). Dado que tras descartar estos 208 registros con error de captura el dataset aún conserva **5,703 observaciones válidas** (superando con holgura el umbral metodológico de 5,000 filas), por lo que se decidió por simplemente borrar esos registros.

2. **`odometer`:**
   - *Tratamiento:* **Creación de indicador booleano `odometer_is_missing` + Imputación por mediana agrupada por `year`**.
   - Los valores nulos, `<= 0` o `> 500,000` millas (140 registros en el subset) representan anomalías de odómetro. Se preservó la señal informacional mediante una variable indicadora binaria ($1$ si fue anómalo/nulo, $0$ si fue válido) y se imputó el valor continuo con la mediana correspondiente a su año de fabricación (`year`).

3. **`condition`:**
   - *Tratamiento:* **Asignación de categoría explícita `'unknown'`**.
   - Al ser un atributo cualitativo ordinal con alta tasa de omisión en el origen (~40%), imputar con la moda sobre-representaría clases como *excellent* o *good*. La categoría explícita conserva la totalidad de las filas sin introducir sesgo subjetivo.

4. **`cylinders`:**
   - *Tratamiento:* **Inferencia basada en la moda del modelo / Arquitectura motriz**.
   - Se imputaron los valores faltantes utilizando la moda de cilindrada observada para cada modelo específico (`model`). Para modelos residuales sin registros suficientes, se asignó `'8 cylinders'` acorde al estándar mecánico predominante en la plataforma de muscle cars americanos clásicos.

5. **`fuel`:**
   - *Tratamiento:* **Eliminación de la columna**.
   - En el segmento de muscle cars americanos (1960–2004), la totalidad de los vehículos utiliza gasolina (`gas`). Al carecer de varianza estadística, la variable no aporta poder discriminativo ni predictivo.

6. **`title_status`, `transmission`, `drive`:**
   - *Tratamiento:* **Asignación de categoría explícita `'unknown'`**.
   - Se estandarizan los nulos en una categoría nula explícita, preservando el volumen de registros sin asumir configuraciones mecánicas (automática vs. manual) o estatus legal del título.

7. **`type`, `paint_color`, `size`:**
   - *Tratamiento:* **Asignación de categoría explícita `'unknown'`**.
   - Permite que los algoritmos basados en distancias (KNN, K-Means) o árboles de decisión manejen la ausencia del dato sin fabricar características estéticas o de carrocería.

8. **`description`:**
   - *Tratamiento:* **Relleno con cadena vacía `""`**.
   - Prepara la columna textual para el procesamiento de lenguaje natural (NLP) y extracción de términos clave en la Práctica 9, evitando excepciones por tipos de datos mixtos (`float` / `str`).

9. **`VIN`:**
   - *Tratamiento:* **Transformación a variable indicadora `is_vin_missing` y eliminación del texto original**.
   - El número de serie crudo es un identificador de alta cardinalidad no utilizable directamente en modelos estadísticos, pero la presencia o ausencia del VIN ($1$ = ausente, $0$ = presente) funciona como una característica proxy de la confiabilidad y transparencia del anuncio.

10. **`lat` y `long`:**
    - *Tratamiento:* **Imputación geoespacial con la mediana del estado (`state`)**.
    - Para el 0.24% de registros con coordenadas nulas, se imputó el centroide aproximado mediante la mediana de latitud y longitud correspondiente a su estado federado.

11. **Columnas Eliminadas:**
    - `county`: 100% de valores nulos en el origen.
    - `url`, `image_url`, `region_url`: Metadatos de navegación de Craigslist sin valor analítico.
    - `fuel`: Variable constante sin varianza.
    - `VIN`: Reemplazada por la variable ingenieril `is_vin_missing`.

---

## 4. Resumen Técnico Final de la Práctica 1

- **Filas iniciales (Dataset Crudo):** `426,880`
- **Filas tras filtrado temático (Muscle Cars 1960 - 2004):** `5,911`
- **Filas eliminadas por error de captura en variable objetivo (`price < $500` o `> $500,000`):** `208`
- **Filas finales en dataset limpio (`muscle_cars_clean.csv`):** `5,703`
- **Total de atributos finales:** `22`
- **Porcentaje de valores nulos resultante en todas las columnas:** `0.00%`
- **Rango temporal cubierto (`posting_date`):** `2021-04-04` a `2021-05-05` (30 días continuos en el subconjunto).
- **Archivos generados en el repositorio:**
  - `Practica 1 - Limpieza de Datos/first_inmmersion.py`: Script exploratorio y diagnóstico del dataset completo.
  - `Practica 1 - Limpieza de Datos/limpieza_datos.py`: Pipeline automatizado de limpieza, depuración y exportación.
  - `Practica 1 - Limpieza de Datos/muscle_cars_clean.csv`: Dataset final limpio y libre de nulos.
  - `Practica 1 - Limpieza de Datos/diagnostico_inicial.md`: Documento formal de diagnóstico y fundamentación técnica.
