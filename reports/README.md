# Resultados de evaluación (issue #11)

Resultados pequeños y reproducibles de `scripts/evaluate.py` sobre las **648 consultas test** de FiQA (qrels de BEIR, `test.tsv`). Los consume la app (#15) y el informe (#12). Los rankings completos, índices y embeddings **no** se versionan.

| Archivo | Contenido |
| --- | --- |
| `metrics_per_query.csv` | Una fila por (modelo, stem, consulta): P@k, R@k y AP@1000. |
| `metrics_summary.json` | Configuración, conteos, procedencia, versiones y promedios por modelo/stem. |
| `eda_summary.json` | Resultado del EDA (#4), ajeno a esta issue. |

## Cómo regenerar

Requiere los artefactos de `scripts/build_artifacts.py` (fuera de Git, en `artifacts/`): `index_no_stem.json`, `index_stem.json` y `dense/` (ver `AGENTS.md`). Con el entorno de `AGENTS.md` activo y FiQA y MiniLM ya en la caché de Hugging Face:

```powershell
python scripts/evaluate.py
```

Opciones (todas opcionales):

| Opción | Por defecto | Efecto |
| --- | --- | --- |
| `--k N` | `10` | Corte de P@k y R@k. Entero positivo, `<= --depth`. |
| `--depth N` | `1000` | Profundidad del ranking recuperado por consulta. Debe ser `>= 1000`: AP y MAP se cortan siempre en 1000 (`ap_at_1000`). |
| `--artifacts-dir DIR` | `artifacts` | Dónde buscar los artefactos. |
| `--output-dir DIR` | `reports` | Dónde escribir el CSV y el JSON. |
| `--rankings-dir DIR` | (nada) | Vuelca los rankings completos como JSONL gzip (uno por corrida, `rankings_<modelo>[_stem\|_no_stem].jsonl.gz`). Se rechaza si cae dentro de `reports/`; use `artifacts/` o similar. Nunca se versionan. |

Reejecutar sobrescribe los dos archivos. Se preparan ambos como temporales y solo entonces se reemplazan, primero el CSV y después el JSON (si algo falla antes de reemplazar, quedan los anteriores). Al terminar, el propio script ejecuta `verify_reports` (ver «Consistencia entre el CSV y el JSON»). Ejecuta 7 corridas: Jaccard, TF-IDF y BM25 con y sin Porter (índices distintos) y el denso una sola vez, porque no usa preprocesamiento léxico. Los tiempos y el progreso van solo a la salida estándar, nunca a los archivos.

Errores comprobados: `k` o `depth` no positivos, `k > depth`, `depth < 1000`, artefactos ausentes (el mensaje indica cómo generarlos), índice con configuración de stemming distinta, de otro corpus o cuyo contenido no coincide con el corpus y el preprocesamiento actuales (comprobación por muestra, ver abajo), y artefacto denso de otro corpus (por `corpus_sha256`).

## `metrics_per_query.csv`

UTF-8, saltos de línea `\n`, cabecera y 7 × 648 = 4 536 filas de datos. Cabecera exacta y orden de columnas:

`model,stem,qid,k,precision,recall,ap_at_1000`

| Columna | Tipo | Valores / rango | Significado |
| --- | --- | --- | --- |
| `model` | texto | `jaccard`, `tfidf`, `bm25`, `dense` | Modelo de recuperación. |
| `stem` | texto | `false`, `true`, `na` | `false`/`true`: índice clásico sin/con Porter. `na`: el denso no aplica preprocesamiento léxico (usa el texto original), así que no hay variante con/sin Porter. Comprobado con `pandas.read_csv`: `na` no se interpreta como nulo y la columna queda como texto (`false`/`true`/`na`), no booleana. |
| `qid` | texto | id de consulta de los qrels test | Siempre cadena (como en `ris.data`); `pandas.read_csv` la infiere como entero, use `dtype={"qid": str}` para unirla con `ris.data`. |
| `k` | entero | `--k` (10 por defecto) | Corte de `precision` y `recall`; igual en todas las filas. |
| `precision` | float | `[0, 1]` | P@k = relevantes en el top-k / **k** (un ranking con menos de k documentos se penaliza). |
| `recall` | float | `[0, 1]` | R@k = relevantes en el top-k / **todos** los relevantes de la consulta. |
| `ap_at_1000` | float | `[0, 1]` | AP sobre las primeras 1000 posiciones: suma de P@i en cada posición i con relevante / **todos** los relevantes. Un relevante fuera de las 1000 primeras aporta 0 pero sigue en el denominador. |

Relevante = documento con `score > 0` en los qrels. Una consulta sin ranking (vacía o sin términos conocidos) aporta 0 en las tres métricas. Las fórmulas son las de `ris.evaluation` (issue #10), que se usan directamente.

**Orden de filas** (determinista): `model` en el orden `jaccard`, `tfidf`, `bm25`, `dense`; dentro de cada modelo `stem` en el orden `false`, `true`, `na`; dentro de cada uno `qid` ascendente **como cadena** (`"1"`, `"10"`, `"100"`, ..., no numérico).

**Floats**: `repr` completo de Python (sin redondear), p. ej. `0.3333333333333333`; así dos corridas dan archivos idénticos byte a byte.

## `metrics_summary.json`

UTF-8, `indent=2`, claves ordenadas alfabéticamente, `\n`, sin timestamps ni valores no deterministas ni rutas absolutas. Claves de primer nivel:

| Clave | Tipo | Contenido |
| --- | --- | --- |
| `schema_version` | entero | `1`. Subirlo si cambia el esquema de forma incompatible. |
| `config` | objeto | Parámetros de la ejecución y de los modelos (abajo). |
| `counts` | objeto | Conteos verificables de datos y consultas (abajo). |
| `provenance` | objeto | Revisiones y SHA-256 de datos y artefactos usados (abajo). |
| `versions` | objeto | Versiones: `python` y paquetes por nombre de distribución (`faiss-cpu`, `huggingface_hub`, `nltk`, `numpy`, `pyarrow`, `sentence-transformers`, `torch`, `retrieval-information-system`); `null` si no está instalado. |
| `results` | lista de objetos | Una entrada por (modelo, stem), en el orden del CSV (7 entradas). |

### `config`

| Clave | Tipo | Contenido |
| --- | --- | --- |
| `split` | texto | `"test"`. |
| `k`, `depth`, `ap_cutoff` | enteros | Corte de P@k/R@k, profundidad recuperada y corte de AP (`1000`). |
| `models` | lista | `["jaccard", "tfidf", "bm25", "dense"]`. |
| `stem_values` | objeto | Valores de `stem` evaluados por modelo. |
| `query_order`, `row_order`, `ranking_order`, `relevance` | texto | Convenciones de orden y de relevancia (ver arriba). |
| `metrics` | objeto | Definición de `precision`, `recall`, `ap_at_1000` y `map_at_1000`. |
| `preprocessing` | objeto | Preprocesamiento clásico y denso, y `known_limitation` (ver más abajo). |
| `hyperparameters` | objeto | `jaccard`, `tfidf`, `bm25` (con `k1` = 1.5 y `b` = 0.75 leídos del retriever) y `dense` (modelo, revisión, dimensión, normalización, `IndexFlatIP`, `max_seq_length`, texto del documento, leídos de los metadatos del artefacto). |

### `counts`

| Clave | Contenido |
| --- | --- |
| `n_queries` | Consultas evaluadas (consultas de los qrels test). |
| `n_queries_without_relevant` | Consultas sin ningún documento con `score > 0`. |
| `n_qrel_pairs` | Pares (consulta, documento) del archivo de qrels. |
| `n_relevant_pairs` | De ellos, con `score > 0`. |
| `n_relevant_docs_distinct` | Documentos relevantes distintos. |
| `n_relevant_docs_not_in_corpus` | Relevantes que no están en el corpus. |
| `n_corpus_docs` | Documentos del corpus. |
| `n_rows` | Filas de datos del CSV (sin cabecera); `verify_reports` la contrasta con el archivo. |

### `provenance`

| Clave | Contenido |
| --- | --- |
| `dataset` | Lista de `{repo, revision, file, sha256}` de los archivos de FiQA y de los qrels test, con las revisiones fijadas en `ris.data`. |
| `artifacts_dir_files` | Por archivo, con ruta **relativa a `--artifacts-dir`**: `index_no_stem.json` e `index_stem.json` (`sha256`, `format_version`, `stem`, `n_docs`, `n_terms`) y `dense/metadata.json` (`sha256`, `model_revision`, `corpus_sha256`, `doc_ids_sha256`, `format_version`, `n_docs`, `file_sha256` de `embeddings.npy` e `index.faiss`, `build_batch_size`, `build_threads`, `build_library_versions`). El artefacto denso se verifica contra el hash del corpus cargado antes de evaluar. |
| `metrics_per_query_sha256` | SHA-256 de los bytes exactos de `metrics_per_query.csv` que acompañan a este JSON. |
| `nltk_resources` | `{version, files}`: versión de NLTK y SHA-256 de los recursos que usa `preprocess` (`tokenizers/punkt_tab/english/{abbrev_types.txt, collocations.tab, ortho_context.tab, sent_starters.txt}` y `corpora/stopwords/english`), con nombres relativos al directorio de datos de NLTK, nunca rutas absolutas. |
| `query_ids_sha256` | Hash de la lista ordenada de ids de consulta evaluados. |

### `results[]`

| Clave | Tipo | Contenido |
| --- | --- | --- |
| `model`, `stem` | texto | Como en el CSV. |
| `n_queries` | entero | Consultas evaluadas. |
| `n_empty_rankings` | entero | Consultas cuyo ranking quedó vacío. |
| `mean_precision_at_k` | float | Media de `precision` sobre todas las consultas (k = `config.k`). |
| `mean_recall_at_k` | float | Media de `recall`. |
| `map_at_1000` | float | MAP@1000: `ris.evaluation.mean_average_precision` sobre todas las consultas de los qrels. |
| `rankings_sha256` | texto | SHA-256 de todos los rankings de profundidad `depth` de esa corrida (ver abajo). |

## Consistencia entre el CSV y el JSON

Escribir dos archivos no es atómico en conjunto: si falla el segundo reemplazo, quedaría un CSV nuevo con un JSON viejo. Por eso el JSON guarda `provenance.metrics_per_query_sha256`, y `verify_reports(output_dir)` (en `scripts/evaluate.py`) relee ambos archivos y lanza `ValueError` si el hash o `counts.n_rows` no coinciden con el CSV; el script la ejecuta al terminar cada corrida.

- **La vista #15 debe validar antes de mostrar:** calcular el SHA-256 de los bytes de `metrics_per_query.csv` y compararlo con `provenance.metrics_per_query_sha256`; si no coinciden, no mostrar los resultados y avisar.
- **Cómo recuperarse:** ejecutar de nuevo `python scripts/evaluate.py` (idempotente; vuelve a escribir ambos archivos), o restaurar los dos desde Git.

## Qué comprueba y qué no comprueba el script sobre los índices

Los índices clásicos no guardan el hash del corpus ni del preprocesamiento. Tras cargar cada uno, `verify_index_sample` toma una muestra determinista de 500 documentos (ids ordenados, paso uniforme, incluidos el primero y el último), recalcula `preprocess(título + ' ' + texto, stem=…)` con la misma entrada que `InvertedIndex.build` y exige que la longitud y todas las frecuencias de cada documento coincidan; si no, aborta y pide regenerar los índices. Es una comprobación **por muestra, no total**: detecta índices de otro contenido o preprocesados con otros recursos de NLTK (stopwords, punkt) solo si esos cambios afectan a algún documento de la muestra. Los SHA-256 de `provenance.nltk_resources` documentan los recursos usados en esta evaluación, pero no prueban con qué recursos se construyeron los índices: eso lo cubre únicamente la muestra. El corpus y el denso sí se comparan completos (`corpus_sha256`).

## Reproducibilidad

`rankings_sha256` resume una serialización canónica de **todos** los rankings de la corrida (por consulta en orden ascendente: `qid`, y cada `doc_id` con su score en `repr`). Con los mismos artefactos y `--depth`, dos ejecuciones deben dar el mismo hash, los mismos CSV y JSON byte a byte, sin guardar rankings pesados. Cambiar `--depth` cambia el hash, no las métricas (mientras `depth >= 1000`). La igualdad bit a bit del denso depende de la misma máquina, hilos y versiones de `torch`/`faiss`; comparar `versions` y `provenance` antes de comparar hashes entre equipos.

## Qué no se versiona

Datos de FiQA, índices, embeddings, cachés y rankings completos (`--rankings-dir`): se generan localmente o se publican en el Hub del equipo. Solo estos CSV/JSON pequeños viven en `reports/`.

## Limitación conocida de la tokenización clásica

Hasta el 16 de noviembre, `ris.preprocessing.preprocess` descartaba entero todo token no alfanumérico (guiones, decimales, barras): 112 de las 648 consultas test perdían alguna palabra. Desde entonces cada token se parte en sus trozos alfanuméricos (`goog-stock` → `goog`, `stock`; `1.000` → `1`, `000`) y solo se descarta `n't`, igual en documentos y consultas. Con ese único cambio, MAP@1000 subió en las seis configuraciones clásicas (BM25 con Porter: 0,198736 → 0,206219). Limitación que queda: un decimal se parte en dos números (`3.5` → `3`, `5`) y puede coincidir con documentos que solo tengan esos números por separado. El denso no se ve afectado porque usa el texto original.

## Fuera de alcance

nDCG@10 no está implementado: es opcional y exige una fuente primaria verificada para las referencias de BEIR.

## Análisis comparativo (#12)

`comparison_summary.json` (esquema 1) es un derivado de los dos archivos de #11;
no los reemplaza. Lo generan el notebook `12_analisis_modelos.ipynb` o
`python -m scripts.analyze_models`; reproducción en `notebooks/README.md`.

| Clave | Contenido |
| --- | --- |
| `counts`, `methodology` | Conteos contrastados con datos reales, comparador BM25/Porter fijo, tolerancia 1e-12, reglas operativas, selección por extremos de AP y ausencia de aleatoriedad. |
| `overall` | Siete filas: `model`, `stem`, `n`, `precision`, `recall`, `ap_at_1000`. Aquí `ap_at_1000` es la **media** de AP, es decir MAP@1000. |
| `porter`, `versus_bm25_porter` | Pares `target`/`reference`, `n` y, para cada métrica, delta medio absoluto, relativo (fracción; `null` si referencia cero), conteos improved/worsened/tied y mínimo/cuartiles lineales/máximo. |
| `groups` | Por grupo: `n`, lista de `qids` como cadenas y `means` con las siete configuraciones. Cada familia es exhaustiva; grupos de familias distintas se solapan. |
| `cases` | `dense_improves` y `dense_worsens`: `qid`, consulta original, ΔAP, todos los relevantes y evidencia de ambos rankings. Serían `null` si no hubiera casos de ese signo. |
| `cases.*.rankings` | Métricas por consulta, tamaño/hash del ranking local y solo documentos del top-10 más todos los relevantes: `doc_id`, posición (o `null`: «no recuperado en top-1000»), score del modelo, juicio/score qrel, título y fragmento fiel con offsets y longitud original. |
| `provenance` | Hashes de reportes de entrada, revisiones/hashes de datos, recursos NLTK, configuración/versión original y local, commit de código de recuperación/evaluación, hashes de fuentes y artefactos, diferencias de bytes y validación de casos. |

Todos los ids siguen siendo texto; relevancia significa `score > 0`. Un juicio
ausente se etiqueta «no juzgado», nunca «juzgado no relevante». Los scores de
modelos distintos no están calibrados. Los cuatro rankings de casos completos y
las figuras auxiliares quedan en `artifacts/analysis/`, fuera de Git.

Antes de agregar se valida el SHA-256 exacto del CSV, esquema, tipos, split/cortes,
4 536 filas, siete conjuntos idénticos de 648 qids sin duplicados y medias con
`atol=1e-12, rtol=0`. `.gitattributes` mantiene LF en el CSV incluso en Windows;
no cambian sus valores ni su hash original. La ejecución falla si hay una
inconsistencia: no modifica ni regenera los reportes de #11.
