# Estado del proyecto

Actualizado: jueves 12 de noviembre de 2026 (#2, #3, #4, #5 y #6 integradas; #7 y #8 integradas; siguiente: #9).

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

## #6 integrada: Jaccard binario (PR #27, rama `feat/06-jaccard`)

- Nuevo paquete `src/ris/retrievers/`: `base.py` define `Result(doc_id, score)` (dataclass inmutable) y el protocolo `Retriever`; `jaccard.py` implementa `JaccardRetriever(index, stem)`. Rechaza con ValueError un índice con distinta configuración de stemming.
- `InvertedIndex.doc_lengths` cuenta tokens con repeticiones, así que no sirve para |D| en Jaccard. El retriever calcula al iniciarse los términos únicos de cada documento a partir de las posting lists (número de postings que contienen el doc_id), sin cambiar el formato del índice de #5. Los términos de consulta desconocidos cuentan en |Q| y no en la intersección.
- **Probado localmente** (Python 3.12.10, Windows, 12 de noviembre): `python -m pytest tests/test_jaccard.py` pasa 12 pruebas con scores calculados a mano; la suite completa pasa 39; `python -m ruff check .` sin errores.
- Prueba manual con FiQA (índice sin stemming construido en memoria, 50,1 s): el retriever se prepara en 0,17 s; consultas test entre 4 y 16 ms; orden descendente comprobado en 50 consultas y empates ordenados por doc_id. No es una evaluación de métricas (eso es #11).
- Revisión de Claude Code (12 de noviembre): diff leído completo; con `NLTK_DATA` apuntando a una copia, 39 pruebas y ruff en verde; CI "Lint and Test" del PR en verde; cumple los contratos de ris.retrievers (orden por score y doc_id, sin duplicados, consulta vacía, k no positivo). Integrado mediante PR #27 (autor @BrandonFreire).

## #7 hecha: coseno TF-IDF propio

- Rama local `feat/07-tfidf`, creada desde `main` en `6deb642`; se comprobó que coincidía con `main` remoto al comenzar. Integrada mediante el PR #28.
- `TfidfRetriever` reutiliza el índice invertido y aplica a documentos y consultas `tf=1+ln(f)` para `f>0`, `idf=ln(N/df)` y coseno con normalización L2, según el acuerdo en el plan. Ignora términos fuera del vocabulario, devuelve `[]` si la consulta tiene norma cero y devuelve solo documentos con score positivo. El índice no debe modificarse mientras se use el recuperador: IDF y normas se precalculan.
- El ejemplo de `Material_clase/Modelo_vectorial (1).pdf` usa TF lineal e IDF en base 10; también presenta TF sublineal. Una prueba reutiliza su corpus y recalcula los scores con la variante acordada por el equipo, sin atribuir esos valores a la diapositiva.
- **Probado localmente** en Windows con Python 3.12: 57 pruebas, incluidas 18 de TF-IDF; Ruff y `git diff --check` sin errores. Para pytest se usó un `--basetemp` dentro de `.pytest_cache` por permisos del directorio temporal del entorno.
- Comprobación de integración local con los dos índices FiQA existentes: 57 638 documentos y 66 796/46 821 términos sin/con Porter. Dos consultas por índice devolvieron cinco documentos cada una y sus 20 scores se recalcularon de forma independiente desde los postings; una consulta de términos desconocidos devolvió `[]` en ambos. Esto no es una evaluación de relevancia ni una medición de CI.
- Revisión cruzada independiente del código, contrato y pruebas: sin hallazgos de corrección pendientes. Revisión de Claude Code (12 de noviembre): código y pruebas leídos completos, cálculos manuales de las pruebas comprobados y ranking del corpus docente recalculado con un script independiente; con `NLTK_DATA` apuntando a una copia (el directorio de datos de NLTK queda redirigido en el entorno de la app), 57 pruebas y ruff en verde; CI "Lint and Test" del PR #28 en verde. Integrado con el PR #28 (autor @Vladimirjon).

## #8 hecha: BM25 propio

- Rama `feat/08-bm25`, creada desde `main` en `f8374da`. Integrada mediante el PR #29.
- `BM25Retriever` reutiliza el índice invertido con la variante fijada en el plan: k1=1,5, b=0,75, `idf=ln(1+(N-df+0,5)/(df+0,5))` (nunca negativo) y suma sobre términos únicos de la consulta de `idf × tf×(k1+1)/(tf+k1×(1-b+b×dl/avgdl))`. N y avgdl cuentan los documentos vacíos; consulta vacía o avgdl=0 devuelve `[]`; solo se devuelven documentos con algún término coincidente. El índice no debe modificarse mientras se use el recuperador: IDF y la corrección de longitud se precalculan.
- **Probado localmente** en Windows con Python 3.12: 76 pruebas, incluidas 19 de BM25; Ruff y `git diff --check` sin errores. Los valores de las pruebas están calculados a mano (longitudes, avgdl, K e idf), con casos de empate, df=N, saturación de tf, longitud, índice vacío, índice persistido y Porter.
- Comprobación de integración local con los dos índices FiQA (57 638 documentos, avgdl 66,605): el top-10 de tres consultas por índice coincide en orden y score (tolerancia 1e-9) con un recálculo independiente desde los postings; preparación del recuperador en menos de 0,1 s y consultas entre 24 y 121 ms; una consulta de términos desconocidos devolvió `[]` en ambos. No es una evaluación de relevancia (eso es #11) ni una medición de CI.
- Revisión de Claude Code (12 de noviembre): código y pruebas leídos completos; fórmula coincide con `docs/PLAN.md`; cálculos manuales de las pruebas comprobados; contratos de `ris.retrievers` cumplidos (orden por score y doc_id, sin duplicados, consulta vacía, k no positivo). No se hizo revisión cruzada con otro modelo en esta issue.

## GitHub configurado y verificado por API el 9 de noviembre de 2026

- Ruleset `proteger-main` (id 26888439), activo en la rama por defecto: reglas `deletion`, `non_fast_forward`, `update` y `pull_request`; exige 1 aprobación, CODEOWNER, descarta aprobaciones con commits nuevos y requiere resolver hilos. El rol admin puede omitir requisitos solo mediante bypass `pull_request`; no hay push directo.
- Borrado automático de ramas al hacer merge activado.
- Seis milestones M0–M5 creados con las fechas del plan; 12 etiquetas creadas (9 de área y 3 `resp:*`).
- Las 18 issues #1–#18 están creadas con la numeración del plan. Todas asignadas: Vladimirjon (10), Ismael Freire @BrandonFreire (#3, #6, #10, #14) y Jorge Torres @Almejandro04 (#4, #12, #15, #17).

## Decisiones

Python 3.12, `src/ris/`, FiQA inglés, cuatro recuperadores propios/FAISS, evaluación P@10/R@10 y MAP@1000, Streamlit en HF Spaces CPU básico y artefactos fuera de Git; detalle en [PLAN](docs/PLAN.md).

## Próximos pasos

1. Empezar #9 (embeddings y FAISS), que depende solo de #2. #10 puede avanzar con los contratos existentes; con #6, #7 y #8 integradas, #11 solo espera a #9 y #10.
2. Decidir si se mejora la tokenización de guiones y decimales (ver limitación en #4) antes de #11. El resumen de #4 queda listo para reutilizarse en #17; no se declara terminado el informe completo.

## Aprendido y pendiente de confirmar

- El equipo tenía solo Python 3.14; #1 exige 3.12 (`requires-python = ">=3.12,<3.13"`). Se instaló 3.12 en paralelo con winget y el `.venv` se crea con `py -3.12`.
- Los qrels almacenan los ids como enteros y los Parquet como strings; `ris.data` convierte los IDs de qrels a `str`.
- Las 648 consultas test únicas fueron contadas directamente en el TSV fijado; son distintas de las 1 706 filas de qrels.
