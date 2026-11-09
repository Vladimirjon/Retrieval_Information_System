# Dataset: FiQA (BEIR)

Materiales indicados por el profesor:

- Corpus y consultas: https://huggingface.co/datasets/BeIR/fiqa
- Juicios de relevancia (qrels): https://huggingface.co/datasets/BeIR/fiqa-qrels

## Datos verificados

Fuente: API pública de Hugging Face (`datasets-server.huggingface.co`, endpoints `size`, `first-rows` y `statistics`), consultada el 9 de noviembre de 2026.

| Repositorio | Config | Split | Filas | Columnas |
|---|---|---|---|---|
| `BeIR/fiqa` | `corpus` | `corpus` | 57 638 | `_id` (string), `title` (string), `text` (string) |
| `BeIR/fiqa` | `queries` | `queries` | 6 648 | `_id` (string), `title` (string), `text` (string) |
| `BeIR/fiqa-qrels` | `default` | `train` | 14 166 | `query-id` (int64), `corpus-id` (int64), `score` (int64) |
| `BeIR/fiqa-qrels` | `default` | `validation` | 1 238 | ídem |
| `BeIR/fiqa-qrels` | `default` | `test` | 1 706 | ídem |

Observaciones verificadas:

- El corpus está en inglés y es de dominio financiero (preguntas y respuestas de foros). En las primeras filas `title` viene vacío: el texto útil está en `text`.
- Las consultas no vienen separadas por split: el conjunto de test se obtiene filtrando las 6 648 consultas por los `query-id` que aparecen en el split `test` de los qrels.
- En el split `test` de los qrels, `score` vale siempre 1 (mínimo 1, máximo 1): la relevancia es binaria.
- **Tipos distintos:** en los qrels los ids son `int64`; en el corpus y en las consultas `_id` es `string`. Hay que convertir los ids de los qrels a `str` al cargarlos, o ninguna consulta encontrará sus relevantes.

## Datos sin verificar todavía

- Número de consultas distintas en el split `test`: 648 según el paper de BEIR (Thakur et al., 2021). Se confirma en la issue de carga de datos contando los `query-id` únicos.
- Valores de referencia de BEIR para FiQA (nDCG@10 de BM25 y modelos densos): consultar el paper antes de citarlos en el informe.
