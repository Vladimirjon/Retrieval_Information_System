# Estado del proyecto

Actualizado: miércoles 11 de noviembre de 2026 (#1 integrada; #2 implementada localmente, revisión pendiente).

## Ubicación y alcance

La planificación está integrada en `main` (PR #19). #1 está integrada en `main` (PR #21, merge `4257135`) y la issue cerrada. El plazo interno es el martes 17 de noviembre; del 18 al 23 queda contingencia.

## #1 hecha (estructura, entorno y CI)

- **Probado localmente** (`.venv` con Python 3.12.10 sobre Windows): `python -m pytest` pasa 1 prueba, `python -m ruff check .` sin errores, `import ris` funciona y `pip check` sin conflictos.
- **CI**: el job "Lint and Test" del PR #21 pasó en Ubuntu con Python 3.12 (1m17s), lo que confirma la instalación limpia con torch CPU.
- Entregado: `pyproject.toml`, `requirements.txt` con versiones exactas, `src/ris/`, `tests/`, `app/`, `scripts/`, `notebooks/`, `reports/`, `docs/informe/`, `.github/workflows/ci.yml` y los comandos de instalación en `readme.md` y AGENTS.md.
- No probado: Linux/macOS en local; solo se ha comprobado Windows y el CI de Ubuntu.

## #2 en curso (implementación local)

- Rama: `feat/02-carga-fiqa`; el árbol estaba limpio al comenzar esta tarea.
- `src/ris/data.py` implementa `load_corpus()`, `load_queries(split)` y `load_qrels(split)`. Lee Parquet con PyArrow, TSV con `csv` y fija las revisiones upstream de `BeIR/fiqa` y `BeIR/fiqa-qrels`. Añadidos pins directos `huggingface_hub==2.4.0` y `pyarrow==26.0.1`.
- Verificación directa de los archivos fijados: 57 638 documentos con IDs únicos; todos los títulos vacíos; 38 textos vacíos que se conservan. Hay 6 648 consultas globales únicas. Train/validation/test tienen 14 166/1 238/1 706 juicios y 5 500/500/648 consultas únicas. Todos los scores valen 1; no faltan IDs de consultas ni documentos referenciados por qrels.
- Evidencia local: `pytest --basetemp .pytest_cache\issue-2-tmp` pasa 7 pruebas; `ruff check .` sin errores. El cargador, en modo offline después de descargar una vez los archivos fijados, reprodujo los conteos y las correspondencias de los tres splits. Los SHA-256 locales de ambos Parquet y de los tres TSV coinciden con lo documentado.
- Caché descargada en `data/hf_home/`, excluida de Git. En Windows Hugging Face avisó que usará copias de caché en vez de symlinks; la carga se completó.
- Revisión cruzada final sin hallazgos pendientes. Corrigió las observaciones P3 sobre validación de columnas y precisión del formato TSV; se añadió una prueba de rutas, tipo de repo y revisiones fijadas. `load_qrels("test")` lanza ValueError si algún score no es 1 (prueba propia).
- Pendiente: comprobar CI del PR y aprobación/merge.

## GitHub configurado y verificado por API el 9 de noviembre de 2026

- Ruleset `proteger-main` (id 26888439), activo en la rama por defecto: reglas `deletion`, `non_fast_forward`, `update` y `pull_request`; exige 1 aprobación, CODEOWNER, descarta aprobaciones con commits nuevos y requiere resolver hilos. El rol admin puede omitir requisitos solo mediante bypass `pull_request`; no hay push directo.
- Borrado automático de ramas al hacer merge activado.
- Seis milestones M0–M5 creados con las fechas del plan; 12 etiquetas creadas (9 de área y 3 `resp:*`).
- Las 18 issues #1–#18 están creadas con la numeración del plan. Todas asignadas: Vladimirjon (10), Ismael Freire @BrandonFreire (#3, #6, #10, #14) y Jorge Torres @Almejandro04 (#4, #12, #15, #17).

## Decisiones

Python 3.12, `src/ris/`, FiQA inglés, cuatro recuperadores propios/FAISS, evaluación P@10/R@10 y MAP@1000, Streamlit en HF Spaces CPU básico y artefactos fuera de Git; detalle en [PLAN](docs/PLAN.md).

## Próximos pasos

1. Completar revisión/PR de #2; #3 (Ismael) y #10 (Ismael) pueden avanzar en paralelo con sus contratos. #1 se cerró el 11 de noviembre, un día después de la fecha interna del 10.
2. Jorge prepara el esqueleto del informe (#17) y el notebook de #4, que ejecuta al tener #2 y #3.

## Aprendido y pendiente de confirmar

- El equipo tenía solo Python 3.14; #1 exige 3.12 (`requires-python = ">=3.12,<3.13"`). Se instaló 3.12 en paralelo con winget y el `.venv` se crea con `py -3.12`.
- Los qrels almacenan los ids como enteros y los Parquet como strings; `ris.data` convierte los IDs de qrels a `str`.
- Las 648 consultas test únicas fueron contadas directamente en el TSV fijado; son distintas de las 1 706 filas de qrels.
