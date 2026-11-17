# Sistema de Recuperación de Información

Proyecto universitario de Recuperación de Información, Prof. Iván Carrera. Compara modelos clásicos y semánticos sobre FiQA mediante consultas libres, rankings y métricas.

[Enunciado completo](docs/ESPECIFICACION.md): construir índice invertido, implementar Jaccard, coseno TF-IDF y BM25, generar embeddings y buscar en una base vectorial, ofrecer una interfaz web gratuita, evaluar con qrels y analizar los cuatro modelos. Entregables: URL del sistema, código con instrucciones e informe técnico de máximo 5 páginas.

## Modelos y dataset

| Modelo | Representación y búsqueda |
|---|---|
| Jaccard | Vectores binarios sobre índice invertido propio |
| TF-IDF | tf logarítmico, idf ln(N/df), L2 y coseno; implementación propia |
| BM25 | Implementación propia, k1=1.5, b=0.75 |
| Denso | all-MiniLM-L6-v2, 384 dimensiones, normalización y FAISS IndexFlatIP |

Decisiones del equipo: [PLAN](docs/PLAN.md). FiQA está en inglés y es de dominio financiero; el cargador fija las revisiones de origen y conserva los documentos con texto vacío. [Datos verificados](docs/DATASET.md): 57 638 documentos, 6 648 consultas y 648 consultas test distintas; test tiene 1 706 filas de qrels binarios. Fuentes del profesor: [corpus y consultas](https://huggingface.co/datasets/BeIR/fiqa), [qrels](https://huggingface.co/datasets/BeIR/fiqa-qrels).

Evaluación prevista: P@10 y R@10 por consulta, MAP@1000 por sistema, clásicos con y sin Porter. No hay resultados todavía.

## Stack y estructura prevista

Python 3.12, NLTK, NumPy/SciPy dispersas, sentence-transformers, FAISS y Streamlit; pytest, ruff y GitHub Actions. Las versiones están fijadas en `requirements.txt`. No se permiten motores completos ni rank_bm25.

```text
src/ris/              data, preprocessing, index, retrievers, evaluation
scripts/              build_artifacts.py y evaluate.py
app/                  streamlit_app.py y vistas
tests/                pruebas unitarias e integración
notebooks/            EDA y análisis comparativo
reports/              resultados pequeños CSV/JSON
docs/informe/         fuente e informe final
data/, artifacts/     archivos locales, fuera de Git
```

La implementación está en curso. Se prevé publicar los artefactos en un repo de HF Hub del equipo para que el Space los descargue al arrancar; no se guardan en GitHub.

`scripts/build_artifacts.py` muestra el avance del índice por cada 1.000 documentos,
con porcentaje, vocabulario, velocidad y tiempo transcurrido. Ejecutarlo otra vez es
seguro en ejecuciones secuenciales: reconstruye el índice desde el corpus y reemplaza
`artifacts/index_no_stem.json` o `artifacts/index_stem.json`; no agrega documentos
duplicados. Sí vuelve a consumir el tiempo de carga y construcción.

Para construir y comprobar los índices, desde la raíz del proyecto y con el
entorno virtual activado:

```powershell
# Sin stemming
python scripts/build_artifacts.py

# Con stemming de Porter
python scripts/build_artifacts.py --stem
```

Cada comando carga FiQA, construye el índice, lo guarda en `artifacts/` y lo
recarga para comprobar su contenido y configuración. Al finalizar, muestra
el número de documentos, el vocabulario, el tiempo de construcción y el tamaño
del archivo. La primera ejecución puede necesitar acceso a internet para
descargar FiQA o los recursos de NLTK.

Para el denso, medir primero una muestra antes de procesar el corpus completo:

```powershell
python scripts/build_artifacts.py --dense --limit 256 --batch-size 32 --threads 8
python scripts/build_artifacts.py --dense --batch-size 32 --threads 8
```

La muestra ordenada por ID se guarda en `artifacts/dense_sample_256/` y el corpus
completo en `artifacts/dense/`. El script muestra progreso y genera
`build_report.json` con tiempos, tamaños y pico de memoria residente del proceso
(incluye modelo, corpus, codificación y comprobación de la recarga). Los tiempos
de una muestra sirven para estimar, no garantizan el tiempo del corpus completo.
Un directorio denso con archivos se conserva salvo que se indique `--overwrite`;
`--output-dir` permite elegir otra ubicación. Los artefactos permanecen fuera de Git.

El encoder recibe `title + ' ' + text` y la consulta original, sin preprocesamiento
léxico. Conserva también los documentos vacíos. Usa CPU, vectores float32 de 384
dimensiones, normalización L2 y FAISS `IndexFlatIP`; incluye scores negativos.
La revisión fijada de [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/tree/1110a243fdf4706b3f48f1d95db1a4f5529b4d41)
es `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`, con truncamiento a 256 tokens del
modelo incluidos sus tokens especiales. El límite del encoder se aplica igual
al corpus y a las consultas; los textos largos no se fragmentan.

Cada directorio contiene `embeddings.npy`, `index.faiss` y `metadata.json` con
configuración, versiones, hash del corpus y correspondencia ordenada fila-ID.
Los hashes SHA-256 detectan corrupción accidental; los artefactos se deben obtener
de una fuente de confianza. La recarga rechaza configuraciones incompatibles y
comprueba los vectores contra el índice sin codificar de nuevo el corpus:

```python
from ris.retrievers import DenseRetriever

retriever = DenseRetriever.load("artifacts/dense")
results = retriever.search("How should I invest my savings?", k=10)
```

El modelo se carga en la primera consulta no vacía. Cargar los artefactos no
requiere descargarlo; construir o consultar por primera vez puede necesitar red
si el modelo no está en la caché de Hugging Face. Todos los empates se ordenan por
doc_id, incluso en la frontera del top-k, mediante búsqueda exacta de los scores
del corpus y orden posterior. Las pruebas usan vectores sintéticos y un encoder
doble sin descargar el modelo; la construcción y consulta con MiniLM real se
verifican por separado.

## Equipo

| Persona | GitHub | Responsabilidad |
|---|---|---|
| Vladimirjon | @Vladimirjon | Dueño, CODEOWNER y líder técnico: estructura/CI, datos, índice, TF-IDF/BM25, denso, runner, núcleo de app, despliegue y QA |
| Ismael Freire | @BrandonFreire | Preprocesamiento, Jaccard, métricas y comparación lado a lado |
| Jorge Torres | @Almejandro04 | EDA, vista de métricas, análisis e informe |

## Instalación y ejecución

Requiere Python 3.12. En Windows PowerShell, desde la raíz del repositorio:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pip install --no-deps -e .
python -m pytest
python -m ruff check .
```

Según la salida comprobada en Python 3.12.10, `python -m pytest` pasó 1 prueba y `python -m ruff check .` terminó sin errores.

La aplicación Streamlit existe desde la issue #13 y se inicia con el siguiente comando. Requiere los artefactos; si existe `deploy/artifacts.json`, se descargan automáticamente:

```powershell
python -m streamlit run app/streamlit_app.py
```

## Despliegue

El destino es [Streamlit Community Cloud](https://share.streamlit.io/), con los artefactos fuera de Git en el dataset público de Hugging Face `VLADIMIRJON/jab-fiqa-artifacts`. El código y esta preparación no publican nada automáticamente.

### Publicar artefactos (una persona con sesión de Hugging Face)

Construye antes los índices y el directorio denso con `scripts/build_artifacts.py`. Ejecuta `scripts/publish_artifacts.py` desde una carpeta cuyos `artifacts/` se hayan generado después del PR #35 (tokenización corregida). Inicia sesión localmente con `hf auth login` o configura `HF_TOKEN` en el entorno (nunca lo pegues en comandos ni lo guardes en Git). Revisa primero el plan local y sus hashes:

```powershell
python scripts/publish_artifacts.py --dry-run
```

Cuando decidas publicar, ejecuta `python scripts/publish_artifacts.py`. Acepta `--repo-id` y `--artifacts-dir` para cambiar los valores por defecto. El script crea el repositorio dataset público si hace falta, carga los archivos y una tarjeta con procedencia, y escribe `deploy/artifacts.json` apuntando al SHA del commit resultante. No uses una rama móvil como revisión.

Los archivos se derivan de BeIR/fiqa; la licencia de origen debe consultarse en [su tarjeta](https://huggingface.co/datasets/BeIR/fiqa), no se presupone aquí. La configuración de origen y revisiones se encuentra en `src/ris/data.py`.

### Crear la aplicación

En [share.streamlit.io](https://share.streamlit.io/) despliega el repositorio `Vladimirjon/Retrieval_Information_System`, rama `main`, archivo principal `app/streamlit_app.py`; en la configuración avanzada selecciona Python 3.12. El manifiesto `deploy/artifacts.json` debe estar en esa rama y contener el SHA confirmado del dataset.

Community Cloud instalará `app/requirements.txt`: busca primero el archivo de dependencias en la carpeta del archivo de entrada (`app/`) y después en la raíz; intenta instalar con `uv` y, si falla, con `pip`. Fuente: [dependencias de la app en Community Cloud](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/app-dependencies). `requirements.txt` de la raíz sigue siendo el archivo de desarrollo y CI.

Comprobación local del 17 de noviembre de 2026: `uv pip compile requirements.txt --python-version 3.12 --python-platform x86_64-manylinux_2_28`, con `uv 0.13.5`, no resolvió las dependencias de la raíz. Con `--extra-index-url`, la estrategia `first-index` toma `requests` del índice de PyTorch, que solo ofrece `2.28.1`, incompatible con `jupyterlab`. Por eso `app/requirements.txt` contiene solo dependencias de ejecución, sin `--extra-index-url` ni `--index-url`, y obtiene `torch` CPU mediante la URL directa de la rueda Linux para Python 3.12. Comprobado en el servidor el 17 de noviembre de 2026, en un primer despliegue hecho desde `main` antes de integrar este cambio: `uv` falló con ese mismo error y Community Cloud instaló el `requirements.txt` de la raíz con `pip` en unos 2 min 20 s. La plataforma sustituye además `pyarrow` 26.0.1 por una versión anterior a la 25 (instaló la 24.0.0) e indica como motivo un fallo conocido (apache/arrow#50471); el cargador de FiQA se probó en local con `pyarrow` 24.0.0. Tras integrar este cambio, el registro del servidor del mismo día muestra que Community Cloud instaló `app/requirements.txt` con `uv` (73 paquetes resueltos) y que la app usa 977 MiB de memoria con los cuatro modelos cargados.

El despliegue de Community Cloud y el almacenamiento público de artefactos se planifican en modalidad **gratis**, sin facturación. Streamlit presenta Community Cloud como gratuito y documenta aproximadamente 2,7 GB de RAM “as of February 2024”; ese límite puede cambiar. Las apps entran en reposo tras 12 horas sin tráfico. En la primera visita la app descarga los artefactos (unos 250 MB) y carga los modelos; visitas posteriores reutilizan la caché del proceso mientras la instancia siga activa. Fuentes: [planes de Community Cloud](https://docs.streamlit.io/deploy/streamlit-community-cloud) y [recursos y reposo](https://docs.streamlit.io/deploy/streamlit-community-cloud/manage-your-app).

**URL pública:** <https://jab-retrieval-information-system.streamlit.app/> (comprobada el 17 de noviembre de 2026: búsqueda con los cuatro modelos, top-n, Porter, paginación y Métricas; lo que queda por comprobar se registra en [STATE](STATE.md)).

## Cómo contribuir

Leer STATE, TASKS y AGENTS antes de trabajar. Tomar una issue y usar rama tipo/NN-descripcion, con NN=número de issue y Conventional Commits. Un PR por issue con Closes #NN, pruebas y STATE/TASKS actualizados. Nunca git add . ni atribución de IA en commits/PRs.

El ruleset `proteger-main` (id 26888439), verificado por API el 9 de noviembre de 2026, exige PR y una aprobación del CODEOWNER @Vladimirjon; los commits nuevos invalidan aprobaciones y los hilos deben resolverse. El rol admin tiene bypass en modo `pull_request`, sin habilitar push directo; también están activadas las reglas contra borrado y force push, y la eliminación automática de ramas al integrar. Los PR de Ismael y Jorge requieren aprobación de Vladimirjon como CODEOWNER; él, o Claude Code con su cuenta cuando se lo pida, hace el merge. Los PR de Vladimirjon se integran con bypass de admin después de pasar CI (desde #1); cuando haya tiempo, se recomienda una revisión no bloqueante de un compañero o revisión cruzada de otro modelo de IA. GitHub no permite que el autor apruebe su propio PR: [documentación](https://docs.github.com/en/pull-requests/how-tos/review-pull-requests/approving-a-pull-request-with-required-reviews).

## Estado y fechas

Planificación local al 9 de noviembre de 2026; implementación pendiente. **Todo terminado el martes 17 de noviembre**; entrega oficial el 23. Del 18 al 23 solo contingencia.

- [Plan completo y mapa de requisitos](docs/PLAN.md)
- [Estado actual](STATE.md)
- [Tareas y responsables](TASKS.md)
- [Instrucciones de trabajo](AGENTS.md)
