# Estado del proyecto

Actualizado: jueves 12 de noviembre de 2026 (#2, #3, #4 y #5 integradas; siguiente: #6, #7 y #8).

## Ubicación y alcance

La planificación está integrada en `main` (PR #19). #1 está integrada en `main` (PR #21, merge `4257135`) y la issue cerrada. #2 está integrada mediante PR #23 (merge `a1210e1`, issue cerrada, comprobado por API y Git al preparar #4). #3 está integrada mediante PR #24 (merge `20ea87a`, aprobado por @Vladimirjon tras ejecutar 14 pruebas y ruff en local; CI en verde). El plazo interno es el martes 17 de noviembre; del 18 al 23 queda contingencia.

## #1 hecha (estructura, entorno y CI)

- **Probado localmente** (`.venv` con Python 3.12.10 sobre Windows): `python -m pytest` pasa 1 prueba, `python -m ruff check .` sin errores, `import ris` funciona y `pip check` sin conflictos.
- **CI**: el job "Lint and Test" del PR #21 pasó en Ubuntu con Python 3.12 (1m17s), lo que confirma la instalación limpia con torch CPU.
- Entregado: `pyproject.toml`, `requirements.txt` con versiones exactas, `src/ris/`, `tests/`, `app/`, `scripts/`, `notebooks/`, `reports/`, `docs/informe/`, `.github/workflows/ci.yml` y los comandos de instalación en `readme.md` y AGENTS.md.
- No probado: Linux/macOS en local; solo se ha comprobado Windows y el CI de Ubuntu.

## #2 integrada (registro de su implementación y verificación)

- Rama: `feat/02-carga-fiqa`; el árbol estaba limpio al comenzar esta tarea.
- `src/ris/data.py` implementa `load_corpus()`, `load_queries(split)` y `load_qrels(split)`. Lee Parquet con PyArrow, TSV con `csv` y fija las revisiones upstream de `BeIR/fiqa` y `BeIR/fiqa-qrels`. Añadidos pins directos `huggingface_hub==2.4.0` y `pyarrow==26.0.1`.
- Verificación directa de los archivos fijados: 57 638 documentos con IDs únicos; todos los títulos vacíos; 38 textos vacíos que se conservan. Hay 6 648 consultas globales únicas. Train/validation/test tienen 14 166/1 238/1 706 juicios y 5 500/500/648 consultas únicas. Todos los scores valen 1; no faltan IDs de consultas ni documentos referenciados por qrels.
- Evidencia local: `pytest --basetemp .pytest_cache\issue-2-tmp` pasa 7 pruebas; `ruff check .` sin errores. El cargador, en modo offline después de descargar una vez los archivos fijados, reprodujo los conteos y las correspondencias de los tres splits. Los SHA-256 locales de ambos Parquet y de los tres TSV coinciden con lo documentado.
- Caché descargada en `data/hf_home/`, excluida de Git. En Windows Hugging Face avisó que usará copias de caché en vez de symlinks; la carga se completó.
- Revisión cruzada final sin hallazgos pendientes. Corrigió las observaciones P3 sobre validación de columnas y precisión del formato TSV; se añadió una prueba de rutas, tipo de repo y revisiones fijadas. `load_qrels("test")` lanza ValueError si algún score no es 1 (prueba propia).
- Estado de integración actualizado al preparar #4: PR #23 integrado e issue #2 cerrada. El registro de pruebas anteriores se conserva; esta tarea no vuelve a certificar el CI de #2.

## #4 revisada e integrada con el PR #25

- Estado inicial: `main` en `ff99a03`, árbol limpio y seis commits por detrás de `origin/main`. Tras `git fetch origin --prune`, se creó `docs/04-eda-fiqa` desde `origin/feat/03-preprocesamiento` (`c4d8ef4`), como autorizó el equipo. PR #24 sigue abierto; no se modificaron la rama de Ismael, `ris.data`, `ris.preprocessing` ni las dependencias.
- Entregables: `notebooks/04_eda_fiqa.ipynb`, instrucciones en `notebooks/README.md`, auxiliares en `src/ris/eda.py`, ocho pruebas pequeñas en `tests/test_eda.py`, `reports/eda_summary.json` y `docs/informe/04_descripcion_corpus.md` generado desde los agregados.
- **Probado localmente** con Python 3.12.16 en Windows: `python -m pytest` pasa 22 pruebas; `python -m ruff check .` sin errores; `python -m pip check` sin conflictos. Se instaló el entorno con las versiones exactas de `requirements.txt`, sin cambiarlo. El notebook se ejecutó completo con FiQA real, seis figuras y validaciones de tamaños, IDs, distribuciones, TF/CF/DF y conservación de longitudes por Porter.
- Resultados: 57 638 documentos, 6 648 consultas globales y 648 test; 1 706 pares test y 1 706 documentos relevantes distintos. Todos los títulos están vacíos; se conservan 38 textos vacíos. La representación léxica contiene 3 838 998 tokens (media 66.605, mediana 46, p99 319), 66 796 términos sin Porter y 46 821 con Porter (reducción 29.904%, sin cambiar longitudes). Hay 40 documentos sin tokens léxicos y ninguna consulta test vacía tras el pipeline.
- Relevancia: entre 1 y 15 relevantes por consulta (media 2.633, mediana 2); 404 consultas tienen como máximo dos. El documento vacío `117276` está marcado relevante para la consulta `5206`: se conserva y se registra para revisar su efecto al evaluar. Los techos ideales de métricas del notebook no son resultados de recuperadores.
- Reproducibilidad y revisión visual: dos ejecuciones completas reprodujeron exactamente las estadísticas; la segunda tardó aproximadamente 258 s en este equipo. Las seis figuras finales son idénticas por SHA-256 a las revisadas visualmente. Los hashes de los cinco archivos de FiQA coinciden con `docs/DATASET.md`; JSON, notebook y texto para el informe son consistentes. Se registran también hashes de los recursos NLTK. El JSON ocupa aproximadamente 249 KB y el notebook 286 KB; no se versionan corpus, cachés ni PDF.
- Revisión crítica local: se distinguen tokens, tipos, términos, CF y DF, desviación poblacional y percentiles lineales; las gráficas conservan ceros y colas. Se corrigió un valor esperado manual de varianza y se ordenaron imports. Las interpretaciones separan observaciones, implicaciones e hipótesis; no se atribuye mejora de recuperación a la reducción del vocabulario. Los PDF docentes se consultaron localmente y no se incorporan a Git.
- **GitHub y CI:** [PR #25](https://github.com/Vladimirjon/Retrieval_Information_System/pull/25) abierto por @Almejandro04, desde `docs/04-eda-fiqa` hacia `feat/03-preprocesamiento`, con revisión solicitada a @Vladimirjon. Se comprobaron los ocho archivos publicados por sus hashes de Git, el commit `41fac35`, la referencia desde la issue #4 y la ausencia de conflictos. El [CI del commit funcional](https://github.com/Vladimirjon/Retrieval_Information_System/actions/runs/43146066439) pasó en Ubuntu con Python 3.12: instalación, Ruff y tests. La ejecución completa del notebook y la revisión visual son verificaciones locales, no pasos del CI.
- Revisión de @Vladimirjon (12 de noviembre): 22 pruebas y ruff en local, ocho archivos del diff y CI en verde. Tras integrar el PR #24 se cambió la base del PR a `main` (sigue con los mismos ocho archivos). `Closes #4` cierra la issue al integrar. No hay comprobación de despliegue en esta issue.
- Limitación conocida del preprocesamiento (#3), medida el 12 de noviembre con `preprocess`: descarta tokens que no son estrictamente alfanuméricos tras `word_tokenize`, como palabras con guion (`long-term`), decimales (`3.5`) y contracciones, y parte `S&P500` en `p500`. Es consistente en documentos y consultas; su efecto en P@10/R@10/MAP no está medido. Si se corrige, hay que volver a ejecutar el notebook de #4 y regenerar `reports/eda_summary.json` y `docs/informe/04_descripcion_corpus.md`. Decidirlo antes de ejecutar #11.

## #5 hecha: índice invertido y construcción con progreso

- `scripts/build_artifacts.py` informa cinco fases y, durante la construcción, muestra cada 1 000 documentos el porcentaje, vocabulario, velocidad y tiempo transcurrido.
- `InvertedIndex.build` acepta un callback opcional de progreso sin cambiar el resultado del índice. La prueba específica y la suite completa pasan: `python -m pytest` (27 pruebas); `python -m ruff check .` sin errores.
- Ejecuciones repetidas secuenciales reconstruyen desde el corpus y reemplazan el JSON correspondiente a `--stem`, sin acumular documentos.
- Las dos construcciones se ejecutaron sobre FiQA y el script verificó la equivalencia al recargar. Sin stemming: 57 638 documentos, 66 796 términos, 161,67 s de construcción y 38,98 MiB. Con stemming: 57 638 documentos, 46 821 términos, 257,35 s y 36,38 MiB. La suite completa pasó 27 pruebas en Python 3.12.10. Revisión de Claude Code (12 de noviembre): 27 pruebas y ruff en verde; los tres criterios de aceptación cubiertos; se retiraron espacios finales en `index.py`. Los artefactos de `artifacts/` están fuera de Git. Nota de entorno: dentro de la app de escritorio de Claude, NLTK bloquea `AppData\Roaming\nltk_data` por redirección de ruta (5 pruebas de preprocesamiento fallan); con `NLTK_DATA` apuntando a una copia pasan las 27. No afecta al código ni al CI.

## GitHub configurado y verificado por API el 9 de noviembre de 2026

- Ruleset `proteger-main` (id 26888439), activo en la rama por defecto: reglas `deletion`, `non_fast_forward`, `update` y `pull_request`; exige 1 aprobación, CODEOWNER, descarta aprobaciones con commits nuevos y requiere resolver hilos. El rol admin puede omitir requisitos solo mediante bypass `pull_request`; no hay push directo.
- Borrado automático de ramas al hacer merge activado.
- Seis milestones M0–M5 creados con las fechas del plan; 12 etiquetas creadas (9 de área y 3 `resp:*`).
- Las 18 issues #1–#18 están creadas con la numeración del plan. Todas asignadas: Vladimirjon (10), Ismael Freire @BrandonFreire (#3, #6, #10, #14) y Jorge Torres @Almejandro04 (#4, #12, #15, #17).

## Decisiones

Python 3.12, `src/ris/`, FiQA inglés, cuatro recuperadores propios/FAISS, evaluación P@10/R@10 y MAP@1000, Streamlit en HF Spaces CPU básico y artefactos fuera de Git; detalle en [PLAN](docs/PLAN.md).

## Próximos pasos

1. Empezar #6, #7 y #8 (modelos clásicos sobre `InvertedIndex`) desde `main`; #10 puede avanzar con los contratos existentes.
2. Decidir si se mejora la tokenización de guiones y decimales (ver limitación en #4) antes de #11. El resumen de #4 queda listo para reutilizarse en #17; no se declara terminado el informe completo.

## Aprendido y pendiente de confirmar

- El equipo tenía solo Python 3.14; #1 exige 3.12 (`requires-python = ">=3.12,<3.13"`). Se instaló 3.12 en paralelo con winget y el `.venv` se crea con `py -3.12`.
- Los qrels almacenan los ids como enteros y los Parquet como strings; `ris.data` convierte los IDs de qrels a `str`.
- Las 648 consultas test únicas fueron contadas directamente en el TSV fijado; son distintas de las 1 706 filas de qrels.
