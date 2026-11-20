**Escuela Politécnica Nacional**

# Recuperación de información financiera en FiQA

Comparación de Jaccard, TF-IDF, BM25 y MiniLM con búsqueda exacta

**Integrantes.** Ismael Freire, Johann Pasquel y Jorge Torres.

Recuperación de Información · Prof. Iván Carrera · 19 de noviembre de 2026

**Resumen.** Evaluamos Jaccard, TF-IDF, BM25 y MiniLM con las mismas 648 consultas de prueba de FiQA. MiniLM alcanzó un MAP@1000 de 0,311586. Entre los métodos clásicos, el mejor resultado fue el de BM25 con Porter, con 0,206219. El análisis de dos consultas muestra cómo cambia el rendimiento de estos modelos según la pregunta.

## 1. Descripción del corpus utilizado

Usamos la versión de FiQA que distribuye BEIR, una colección de preguntas y respuestas financieras en inglés [1, 2, 3]. El cargador combina el título y el texto de cada documento. Tanto `doc_id` como el identificador de consulta se guardan como cadenas.

**Tabla 1.** Documentos, consultas y juicios de relevancia de FiQA

| Elemento | Cantidad | Interpretación |
| --- | ---: | --- |
| Documentos | 57 638 | Corpus completo, incluidos textos vacíos |
| Consultas globales | 6 648 | Train, validación y test disjuntos |
| Consultas test | 648 | Mismo conjunto para las siete configuraciones |
| Pares de juicios test | 1 706 | Todos con score de relevancia igual a 1 |
| Documentos relevantes test distintos | 1 706 | No equivale al número de consultas |

El conjunto de entrenamiento tiene 5 500 consultas y el de validación, 500. Para evaluar los modelos usamos el conjunto de prueba. El cargador fija las revisiones del corpus y de los qrels y comprueba que sus identificadores correspondan.

Tras corregir la tokenización el 16 de noviembre, el análisis del corpus confirmó que todos los títulos estaban vacíos y que 38 documentos carecían de texto. Los mantuvimos para conservar la colección y sus juicios de relevancia. El documento `117276`, juzgado relevante para `qid 5206`, es uno de esos casos.

El preprocesamiento clásico sin Porter produce 4 099 045 apariciones de términos. La longitud documental media es de 71,12 tokens, la mediana de 48 y el percentil 99 de 340. Las consultas de prueba tienen una media de 6,39 tokens. El cálculo de longitudes documentales incluye los documentos vacíos.

Al aplicar el algoritmo de Porter [4], el vocabulario documental pasó de 78 115 a 57 620 formas (26,24 % de reducción) sobre el mismo número de apariciones. En la sección 4 examinamos el efecto en las métricas.

Los qrels recogen los juicios disponibles. Cada consulta de prueba tiene entre 1 y 15 documentos juzgados relevantes, con media 2,633 y mediana 2. En 404 consultas hay dos relevantes o menos. Con tan pocos documentos relevantes por consulta, recuperarlos todos puede dar un recall alto incluso si la P@10 es baja [5].

<!-- pagebreak -->

## 2. Explicación de las decisiones de diseño

Implementamos los modelos y el índice invertido en Python 3.12, usamos MiniLM para la búsqueda densa y Streamlit para la interfaz. El repositorio documenta las fórmulas y los cortes empleados en la evaluación.

### 2.1. Representación y modelos clásicos

Aplicamos el mismo preprocesamiento al corpus y a las consultas. Pasamos el texto a minúsculas, lo tokenizamos con NLTK y filtramos las palabras vacías en inglés. La reducción de sufijos con Porter es opcional [4]. El tokenizador conserva formas como `401k` y separa `goog-stock` en `goog` y `stock`. También divide `3.5` en `3` y `5`, por lo que pierde el significado del decimal.

Nuestro índice invertido registra para cada término sus frecuencias por documento (postings). El archivo del índice guarda también la frecuencia documental `df`, las longitudes, el tamaño `N` y la configuración de Porter.

**Jaccard binario.** Compara los conjuntos de términos únicos de la consulta y el documento mediante `|intersección(Q,D)| / |unión(Q,D)|` [5]. En nuestra implementación, los términos desconocidos de la consulta cuentan en la unión.

**TF-IDF.** Pondera los términos de la consulta y del documento, normaliza ambos vectores con L2 y calcula su coseno [5]. Usamos `tf = 1 + ln(f)` para frecuencia positiva e `idf = ln(N/df)`, sin suavizado. Una consulta de norma cero produce un ranking vacío.

**BM25.** La puntuación combina la frecuencia del término con un ajuste por la longitud del documento [6]. Sumamos sobre los términos únicos de la consulta. Cada término aporta `idf = ln(1 + (N - df + 0,5)/(df + 0,5))` multiplicado por `f × (k1 + 1) / (f + k1 × (1 - b + b × dl/avgdl))`. Usamos k1 = 1,5 y b = 0,75. La longitud `dl` cuenta apariciones y el promedio `avgdl` incluye los documentos vacíos.

### 2.2. Recuperación densa y evaluación separada

El modelo `all-MiniLM-L6-v2` representa el texto en 384 dimensiones [7]. Usamos el texto original, con un límite de 256 tokens incluidos los especiales, y normalizamos los vectores float32 con L2. Con esos vectores, FAISS IndexFlatIP hace una búsqueda exacta por producto interno, que equivale al coseno [8]. Los reportes fijan la revisión del modelo.

Ordenamos cada ranking por puntuación descendente y resolvemos los empates por `doc_id`. Cada modelo usa su propia escala. La evaluación incluye los documentos vacíos y la vista Buscar los omite al presentar el ranking. Buscar incluye escritura asistida y un grafo 3D opcional para explorar los resultados. En Métricas se pueden comparar variantes de Porter y consultar el detalle por qid a partir de los archivos validados de `reports/`.

### 2.3. Despliegue realizado

Publicamos la aplicación de #16 en [Streamlit Community Cloud](https://jab-retrieval-information-system.streamlit.app/). Hugging Face Hub aloja los seis artefactos en el dataset público `VLADIMIRJON/jab-fiqa-artifacts`. En conjunto suman 261 655 898 bytes. La aplicación verifica su tamaño y SHA-256 con `deploy/artifacts.json` y reutiliza los archivos que pasan la comprobación.

En la prueba local de #16, la interfaz estuvo disponible a los 32,0 s y MiniLM a los 68,3 s. El 17 de noviembre se probaron las búsquedas y Métricas en el servicio público y se midieron 977 MiB de RSS tras la carga inicial. Quedaron por comprobar el pico de memoria, el arranque tras reposo y la vista Comparar (#14).

<!-- pagebreak -->

## 3. Ejemplos de consultas y resultados

Entre las 648 consultas, elegimos las dos que dieron el mayor y el menor valor de `AP@1000(denso) - AP@1000(BM25/Porter)`. En caso de empate, aplicamos el orden lexicográfico de `qid`.

### 3.1. Mejora del denso: qid 2076

**Consulta original:** “Can vet / veterinary bills be considered deductions (tax-deductible) for Income Tax purposes [Canada]?”

**Tabla 2.** Posición de los documentos relevantes para la consulta 2076

| Documento relevante (qrel = 1) | Posición denso | Posición BM25/Porter |
| --- | ---: | --- |
| `278824` | 1 | No recuperado en top-1000 |
| `184646` | 2 | 796 |

**Tabla 3.** Métricas de recuperación para la consulta 2076

| Configuración | P@10 | R@10 | AP@1000 |
| --- | ---: | ---: | ---: |
| Denso | 0,200000 | 1,000000 | 1,000000 |
| BM25 con Porter | 0,000000 | 0,000000 | 0,000628 |

Los dos documentos relevantes relacionan animales con gastos médicos sin usar `vet` ni `veterinary`. El primer resultado de BM25, `541809`, trata de gastos empresariales y carece de juicio para esta consulta. El denso obtiene una ventaja de +0,999372 en AP. El texto de `278824` se refiere a EE. UU. y su juicio de relevancia deja abierta su aplicabilidad fiscal a Canadá.

### 3.2. Empeoramiento del denso: qid 1310

**Consulta original:** “Is is possible to take a mortgage using Bitcoin as collateral?”

**Tabla 4.** Posición del documento relevante para la consulta 1310

| Documento relevante (qrel = 1) | Posición denso | Posición BM25/Porter |
| --- | ---: | --- |
| `535651` | 113 | 1 |

**Tabla 5.** Métricas de recuperación para la consulta 1310

| Configuración | P@10 | R@10 | AP@1000 |
| --- | ---: | ---: | ---: |
| Denso | 0,000000 | 0,000000 | 0,008850 |
| BM25 con Porter | 0,100000 | 1,000000 | 1,000000 |

BM25 sitúa en primer lugar la respuesta que menciona bitcoins como garantía. MiniLM la recupera en la posición 113 y su AP queda 0,991150 por debajo de BM25. El primer resultado de MiniLM, `101103`, define la hipoteca y carece de juicio para esta consulta.

Recalculamos las métricas de ambos casos a partir de las posiciones guardadas. Los valores coinciden con el CSV. Si un documento carece de juicio, su relevancia queda indeterminada.

<!-- pagebreak -->

## 4. Análisis de métricas de evaluación

### 4.1. Protocolo y resultados globales

Evaluamos 648 consultas de prueba con relevancia binaria (`score > 0`) y rankings de hasta 1000 documentos. Probamos siete configuraciones, correspondientes a los tres métodos clásicos con y sin Porter y al modelo denso. Fijamos los parámetros antes de evaluar el conjunto de prueba. Al verificar los resultados, contrastamos las medias del CSV y el JSON con una tolerancia absoluta de 1e-12.

**P@10** es la proporción de documentos relevantes en las diez primeras posiciones, con denominador fijo de 10. **R@10** mide qué proporción de todos los documentos relevantes de la consulta aparece en esas posiciones. **AP@1000** suma la precisión en cada posición relevante hasta 1000 y divide por el total de relevantes, incluidos los que quedaron fuera del ranking. **MAP@1000** promedia la AP de las 648 consultas, incluidos los ceros. Las P y R globales también son medias por consulta [5].

**Tabla 6.** Medias sobre las 648 consultas de prueba

| Modelo | Porter | P@10 | R@10 | MAP@1000 |
| --- | --- | ---: | ---: | ---: |
| Jaccard | No | 0,015432 | 0,063079 | 0,038520 |
| Jaccard | Sí | 0,015123 | 0,060502 | 0,038461 |
| TF-IDF | No | 0,049074 | 0,217460 | 0,133028 |
| TF-IDF | Sí | 0,050463 | 0,219553 | 0,139718 |
| BM25 | No | 0,063735 | 0,284832 | 0,180993 |
| BM25 | Sí | 0,069444 | 0,313021 | 0,206219 |
| Denso | No aplica | 0,104475 | 0,441306 | 0,311586 |

Frente a BM25 con Porter, MiniLM obtiene 0,105367 más en MAP@1000 (un 51,09 % respecto al valor de BM25), 0,035031 más en P@10 y 0,128284 más en R@10. La AP sube en 443 consultas, baja en 156 y empata en 49. Tanto en P@10 como en R@10 hubo 219 mejoras, 60 pérdidas y 369 empates.

Con Porter, el cambio de MAP fue de -0,000058 en Jaccard, +0,006689 en TF-IDF y +0,025227 en BM25. Respecto a la variante sin Porter, estos cambios representan -0,15 %, +5,03 % y +13,94 %, respectivamente. En BM25, la AP subió en 287 consultas, cayó en 265 y se mantuvo en 96.

En los reportes de la evaluación final verificamos 4 536 filas, siete configuraciones y 49 medias por grupo. Los hashes de los artefactos que constan en esos reportes coinciden con los del manifiesto de despliegue.

<!-- pagebreak -->

### 4.2. Tipos de consulta observables

Agrupamos las consultas según tres criterios independientes. Medimos la longitud tras el preprocesamiento sin Porter y registramos la presencia de dígitos y de secuencias ASCII de dos o más mayúsculas. La Tabla 7 reúne los resultados por longitud.

**Tabla 7.** MAP@1000 por longitud de consulta

| Longitud de consulta | n | MAP BM25/Porter | MAP denso |
| --- | ---: | ---: | ---: |
| 0-3 tokens | 62 | 0,105802 | 0,192947 |
| 4-7 tokens | 402 | 0,214012 | 0,308552 |
| 8 o más tokens | 184 | 0,223031 | 0,358192 |

El denso obtuvo el mayor MAP en los siete grupos. De las consultas, 73 contienen dígitos y 575 carecen de ellos. Las secuencias de mayúsculas aparecen en 118 consultas y faltan en 530. Con Porter, el MAP de TF-IDF en consultas cortas bajó de 0,072929 a 0,056409, pese a mejorar en el conjunto completo. Los grupos pueden diferir en longitud, tema y dificultad. Las diferencias son descriptivas, sin pruebas de significancia.

### 4.3. Alcance de la evidencia y limitaciones

**Limitaciones.** Comprobamos los índices clásicos en una muestra de 500 documentos. Los juicios pueden ser incompletos. Al tokenizar se pierden los decimales y MiniLM trunca los textos largos. Los resultados corresponden a FiQA.

**Conclusión.** MiniLM obtuvo los mejores promedios en FiQA. Entre los modelos clásicos, BM25 con Porter ocupó el primer lugar. En la consulta sobre Bitcoin, BM25 situó primero el documento relevante que contenía los términos decisivos. El promedio resume el desempeño global, y los casos concretos muestran dónde cambia el orden de los documentos relevantes y qué juicios respaldan esa comparación.

### Fuentes y trazabilidad

Las cifras de las tablas proceden de los reportes de evaluación. El registro de verificación adjunto permite reproducir su comprobación.

[1] Maia, M., Handschuh, S., Freitas, A., Davis, B., McDermott, R., Zarrouk, M. y Balahur, A. (2018). [WWW'18 Open Challenge: Financial Opinion Mining and Question Answering](https://andrefreitas.org/papers/www_fiqa_2018.pdf). *Companion Proceedings of The Web Conference 2018*, 1941-1942.

[2] Thakur, N., Reimers, N., Rücklé, A., Srivastava, A. y Gurevych, I. (2021). [BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval Models](https://datasets-benchmarks-proceedings.neurips.cc/paper_files/paper/2021/hash/65b9eea6e1cc6bb9f0cd2a47751a186f-Abstract-round2.html). *NeurIPS Datasets and Benchmarks*.

[3] Material de la asignatura facilitado por Iván Carrera. BEIR, [corpus FiQA](https://huggingface.co/datasets/BeIR/fiqa) y [juicios de relevancia](https://huggingface.co/datasets/BeIR/fiqa-qrels).

[4] Porter, M. F. (1980). [An algorithm for suffix stripping](https://tartarus.org/martin/PorterStemmer/def.txt). *Program*, 14(3), 130-137.

[5] Manning, C. D., Raghavan, P. y Schütze, H. (2008). [Introduction to Information Retrieval](https://nlp.stanford.edu/IR-book/information-retrieval-book.html). Cambridge University Press.

[6] Robertson, S. y Zaragoza, H. (2009). [The Probabilistic Relevance Framework: BM25 and Beyond](https://www.ccs.neu.edu/home/vip/teach/IRcourse/IR_surveys/robertson_foundations.pdf). *Foundations and Trends in Information Retrieval*, 3(4), 333-389.

[7] Sentence Transformers (s. f.). [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2). Ficha del modelo.

[8] Faiss (s. f.). [Faiss indexes](https://github.com/facebookresearch/faiss/wiki/Faiss-indexes). Documentación de `IndexFlatIP`.
