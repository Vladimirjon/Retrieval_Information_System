# Dataset: FiQA (BEIR)

Materiales indicados por el profesor:

- Corpus y consultas: https://huggingface.co/datasets/BeIR/fiqa
- Juicios de relevancia (qrels): https://huggingface.co/datasets/BeIR/fiqa-qrels

## Revisiones utilizadas

El cargador fija estas revisiones de Hugging Face para que una ejecución no cambie si se actualiza `main`:

| Repositorio | Revisión | Archivo |
|---|---|---|
| [`BeIR/fiqa`](https://huggingface.co/datasets/BeIR/fiqa/tree/979c07a7cb5ccc6ca009792241fa1250b98055dd) | `979c07a7cb5ccc6ca009792241fa1250b98055dd` | `corpus/corpus-00000-of-00001.parquet`, `queries/queries-00000-of-00001.parquet` |
| [`BeIR/fiqa-qrels`](https://huggingface.co/datasets/BeIR/fiqa-qrels/tree/252958f2d646e22cab6d0c72dd3f0d5de6d0655a) | `252958f2d646e22cab6d0c72dd3f0d5de6d0655a` | `train.tsv`, `dev.tsv`, `test.tsv` |

El código descarga estos archivos con `huggingface_hub` y lee los Parquet con PyArrow. Los archivos quedan en la caché local de Hugging Face, fuera del repositorio.

## Verificación de los archivos fijados

Se descargaron los archivos upstream de esas revisiones el 11 de noviembre de 2026. Se leyó el corpus completo con PyArrow 26.0.1; los TSV se procesaron con `csv.DictReader`. Las sumas SHA-256 calculadas para los Parquet coinciden con sus metadatos LFS.

| Archivo | Filas | IDs únicos | SHA-256 |
|---|---:|---:|---|
| `corpus/corpus-00000-of-00001.parquet` | 57 638 | 57 638 documentos | `b36bb07dab3bfad488190818828e749a52faf5ba391c1d51dac8b06064576f99` |
| `queries/queries-00000-of-00001.parquet` | 6 648 | 6 648 consultas | `e9b3863724ffd538d8efff19684ef8ae495bdbddd96b35e81bdc39e3e22860d6` |
| `train.tsv` | 14 166 juicios | 5 500 consultas | `e46a99529aa61b12e086a5c057e4e0ecaeda9502e53d5fe855efce778f1ec59a` |
| `dev.tsv` (`validation`) | 1 238 juicios | 500 consultas | `03b27e547cd29dc721c3b93ce8ddd70df6b2d0ce954d8ef3659d386e8cd72c11` |
| `test.tsv` | 1 706 juicios | 648 consultas | `6adc2a640dcdd22bb8b3858f89107adef2a7c3db20a63550dfa7a0f71e379e44` |

Comprobaciones sobre esos archivos:

- Los Parquet tienen las columnas `_id`, `title` y `text` de tipo string, sin nulos. Los IDs de documentos y consultas son únicos.
- Los 57 638 títulos de documentos están vacíos. También están vacíos los títulos de las 6 648 consultas. **Hay 38 documentos con `text` vacío**; se conservan para mantener el corpus completo.
- Los qrels tienen columnas `query-id`, `corpus-id` y `score`. Todos los scores valen 1 y no hay pares `(query-id, corpus-id)` duplicados.
- Todos los IDs de consulta de los qrels existen en `queries`; los splits son disjuntos y reúnen las 6 648 consultas.
- Todos los `corpus-id` citados en los qrels existen en el corpus. Los splits referencian 14 166 documentos de train, 1 238 de validation y 1 706 de test; su unión contiene 17 110 documentos. Los otros 40 528 documentos no tienen qrel.
- Los IDs de qrels son valores numéricos escritos en los TSV; `csv.DictReader` los recibe como cadenas y el cargador los convierte a entero y luego a `str` para hacerlos coincidir con los IDs string de los Parquet.

El split `dev.tsv` se expone como `validation` en la interfaz de carga para mantener el nombre usado por el dataset de Hugging Face. También se acepta `dev` como alias.

## Pendiente

- Verificar los valores de referencia de BEIR para FiQA (nDCG@10 de BM25 y modelos densos) en el paper antes de citarlos en el informe.
