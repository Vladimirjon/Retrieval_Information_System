# Estado del proyecto

Actualizado: jueves 12 de noviembre de 2026 (#2–#9 integradas; siguiente: #10 y #13).

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

## #9 hecha: embeddings y búsqueda exacta con FAISS

- Rama `feat/09-denso-faiss`, creada desde `main` en `002f3a1`, cuyo commit se comprobó en GitHub. Integrada mediante el PR #30. `DenseRetriever` utiliza MiniLM, texto original, vectores float32 normalizados de 384 dimensiones y `IndexFlatIP`. El desempate por ID se aplica a todo el ranking; se conservan los documentos vacíos y los scores negativos.
- La revisión del modelo quedó fijada en `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`; el límite es de 256 tokens del modelo, incluidos los especiales. El índice, embeddings, metadatos y reporte se guardan en `artifacts/`, excluido de Git.
- **Probado localmente**: 112 pruebas y Ruff pasan, con `python -m pytest` y con `pytest` a secas (ver revisión abajo). Con 32 documentos reales, se construyó y recargó el índice y una consulta real coincidió en ranking y scores con un cálculo NumPy independiente. Una muestra independiente de 256 documentos tardó 7,53 s con 4 hilos y 5,40 s con 8 hilos, batch 32; se estimó una generación completa de unos 20–28 minutos, no un tiempo garantizado.
- **FiQA completo, probado localmente**: 57 638 documentos, incluidos los vacíos; batch 32 y 8 hilos en CPU. La generación tardó 1 581,28 s (26 min 21 s), el pico de memoria residente del proceso fue 1 255,46 MiB y los artefactos principales ocupan 169,62 MiB. El índice se guardó en `artifacts/dense/`, se recargó y la consulta `How should I invest my savings?` obtuvo el mismo ranking y scores antes y después de la persistencia, sin recodificar el corpus. Detalles en `artifacts/dense/build_report.json`.
- Se recalculó independientemente el hash de FiQA (`78c669dba3430c37ca631b9c368311799d312f59d82bc850186be08012741bb0`); coincide con el manifiesto. Para dos consultas reales, los 10 primeros IDs y scores de FAISS coincidieron con producto vectorial NumPy (tolerancias `rtol=1e-5`, `atol=1e-6`). Consulta vacía devuelve `[]`. Estos resultados verifican la implementación, no miden relevancia.
- Revisión cruzada (12 de noviembre): `mistral-large-4` de NVIDIA vía OpenCode (gratis, sin Codex), con los cambios como adjunto, sin hallazgos P0/P1; sus P2/P3 se evaluaron uno a uno y casi todos se descartaron tras leer el código (división por cero imposible tras codificar un lote, archivos huérfanos inexistentes porque se reescriben los tres, `faiss-cpu` fijado e instalado). No ejecutó nada, así que no verificó pruebas, versiones ni rendimiento.
- Revisión de Claude Code (12 de noviembre): código, pruebas y script leídos completos; 112 pruebas y Ruff en verde. Con los artefactos reales de FiQA (57 638 documentos): ids y hash del corpus coinciden con el corpus; embeddings float32 de norma 1; 5 documentos recodificados con el MiniLM real (vacío, el más largo de 16 990 caracteres y 3 al azar) difieren de lo guardado como máximo 1,5e-7; el top-10 de 5 consultas coincide en orden y score con NumPy; consultas de 136 a 520 ms en este equipo, incluida la codificación de la consulta (medición local, no del Space). El pico de memoria y los 26 min de construcción se toman de `build_report.json`, no se midieron de nuevo.
- Hallazgo corregido: `pytest` a secas no recolectaba `tests/test_build_artifacts.py` (`No module named 'scripts'`); se añadió `pythonpath = ["."]` en `pyproject.toml`. Pendientes menores sin bloquear: `--output-dir` puede escribir fuera de las carpetas ignoradas por Git, y los hilos de torch al consultar salen de los metadatos (8); revisarlo en #16 si el Space tiene menos CPUs. CI "Lint and Test" del PR #30 en verde. Integrado con el PR #30 (autor @Vladimirjon). La publicación de artefactos y el despliegue corresponden a #16.

## #10 en curso: métricas de recuperación (rama `feat/10-metricas`)

- Rama creada el 13 de noviembre desde `main` en `5748687`, igual a `origin/main` tras `git fetch`. Sin commit, push ni PR todavía.
- `src/ris/evaluation.py` implementa `precision_at_k`, `recall_at_k`, `average_precision` y `mean_average_precision` según el contrato. P@k divide por k aunque el ranking sea corto; R@k y AP dividen por todos los relevantes; AP solo suma hasta el corte k (1000 por defecto). MAP promedia todas las consultas de qrels: ranking ausente o consulta sin relevantes aportan 0; las consultas del run fuera de qrels se ignoran; qrels vacíos devuelven 0. k no positivo o ranking duplicado lanzan ValueError. `relevant` también acepta lista.
- `mean_average_precision` recibe `qrels: dict[str, set[str]]`: #11 debe convertir `load_qrels` (dict de scores) a conjuntos de ids con score>0.
- **Probado localmente** (Python 3.12.10, Windows, 13 de noviembre): `python -m pytest tests/test_evaluation.py` pasa 24 pruebas (ejemplo manual P@2=R@2=0.5, AP@1000=5/6, MAP=5/12, relevante en la posición 1001 fuera del corte, k inválido, duplicados y consulta ausente); la suite completa pasa 136; `python -m ruff check .` sin errores. Comprobación con qrels test reales (offline): run ideal da MAP=1.0 en 648 consultas y run vacío 0.0; no es una evaluación de modelos (eso es #11).
- Pendiente: commit, PR con `Closes #10`, CI y revisión de @Vladimirjon.

## GitHub configurado y verificado por API el 9 de noviembre de 2026

- Ruleset `proteger-main` (id 26888439), activo en la rama por defecto: reglas `deletion`, `non_fast_forward`, `update` y `pull_request`; exige 1 aprobación, CODEOWNER, descarta aprobaciones con commits nuevos y requiere resolver hilos. El rol admin puede omitir requisitos solo mediante bypass `pull_request`; no hay push directo.
- Borrado automático de ramas al hacer merge activado.
- Seis milestones M0–M5 creados con las fechas del plan; 12 etiquetas creadas (9 de área y 3 `resp:*`).
- Las 18 issues #1–#18 están creadas con la numeración del plan. Todas asignadas: Vladimirjon (10), Ismael Freire @BrandonFreire (#3, #6, #10, #14) y Jorge Torres @Almejandro04 (#4, #12, #15, #17).

## Decisiones

Python 3.12, `src/ris/`, FiQA inglés, cuatro recuperadores propios/FAISS, evaluación P@10/R@10 y MAP@1000, Streamlit en HF Spaces CPU básico y artefactos fuera de Git; detalle en [PLAN](docs/PLAN.md).

## Próximos pasos

1. Con #6–#9 integradas, #11 solo espera a #10, y #13 ya puede empezar (depende de #6–#9); #10 puede avanzar con los contratos existentes.
2. Decidir si se mejora la tokenización de guiones y decimales (ver limitación en #4) antes de #11. El resumen de #4 queda listo para reutilizarse en #17; no se declara terminado el informe completo.

## Aprendido y pendiente de confirmar

- El equipo tenía solo Python 3.14; #1 exige 3.12 (`requires-python = ">=3.12,<3.13"`). Se instaló 3.12 en paralelo con winget y el `.venv` se crea con `py -3.12`.
- Los qrels almacenan los ids como enteros y los Parquet como strings; `ris.data` convierte los IDs de qrels a `str`.
- Las 648 consultas test únicas fueron contadas directamente en el TSV fijado; son distintas de las 1 706 filas de qrels.
