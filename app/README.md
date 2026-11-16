# App jab (issue #13)

Buscador sobre FiQA con los cuatro modelos. Desde la raíz del repositorio:

```powershell
python -m streamlit run app/streamlit_app.py
```

Requiere los artefactos de `scripts/build_artifacts.py` (`artifacts/index_no_stem.json`, opcionalmente `index_stem.json`, y `artifacts/dense/`) y el corpus FiQA en la caché de Hugging Face. Variables opcionales: `JAB_ARTIFACTS_DIR` y `JAB_REPORTS_DIR`. Si falta un archivo, la interfaz muestra qué falta y el comando que lo genera; los demás modelos siguen funcionando.

## Capas

| Archivo | Responsabilidad |
|---|---|
| `src/ris/engine.py` | `SearchEngine`: único punto de carga y búsqueda. Sin Streamlit. |
| `app/frontend/` | Interfaz de búsqueda: `index.html`, `styles.css` (sistema de diseño, paleta ciruela) y `app.js`. |
| `app/ui.py` | Registra el componente, valida lo que envía el navegador y arma la respuesta. |
| `app/streamlit_app.py` | Página, caché del motor y vistas de los compañeros. |
| `app/context.py` | `AppContext`: lo que reciben las vistas. |
| `app/views/comparison.py`, `app/views/metrics.py` | Reservados para #14 y #15. |
| `app/static/` | Logo `jab` (palabra e icono) en SVG. |

La referencia visual aprobada está en `design/prototipo/` y sus reglas en `design/README.md`.

En la vista Buscar, `app/ui.py` llama a `SearchEngine.search(..., skip_empty=True)` para omitir documentos sin título ni texto y completar el top-n con los siguientes resultados. El valor por defecto de `skip_empty` es `False`: `AppContext.search()` y las evaluaciones conservan el ranking original. La respuesta del componente incluye `terms` y, por hit clásico, `matched` y `spans` (tramos `[inicio, fin]` sobre `text` limpio) para tooltips y resaltado. La paginación de cinco resultados es exclusivamente visual; los ranks globales no cambian.

## Cómo se comunican navegador y Python

El componente envía un único estado `request = {q, model, k, stem, view, seq}`. Cada cambio sube `seq`, así ningún cambio se pierde aunque llegue mientras Python atiende otro. Python valida todo (modelo y vista en lista blanca, `k` en 5/10/20/50, consulta de 300 caracteres como máximo), busca y responde con el mismo `seq` y una `key` de la búsqueda. La interfaz descarta respuestas de un `seq` viejo y no repinta si solo cambió la vista. Si detecta que Python atendió un estado anterior, lo reenvía una vez.

## Contrato `render(context)` para #14 y #15

Cada vista es un módulo de `app/views/` con `render(context: AppContext) -> None`. Se dibuja con Streamlit debajo de la cabecera de jab cuando el usuario elige Comparar o Métricas; la consulta, el modelo, top-n y Porter son los de la barra. El contexto es inmutable:

| Campo | Significado |
|---|---|
| `engine` | `SearchEngine` único, con los modelos ya cargados. No construyas recuperadores ni leas índices por tu cuenta. |
| `query` | Consulta actual sin espacios sobrantes; puede ser `""`. |
| `model` | Modelo elegido: `jaccard`, `tfidf`, `bm25` o `dense`. |
| `k` | Top-n elegido (5, 10, 20 o 50). |
| `stem` | `True` si los clásicos usan el índice con Porter; el denso lo ignora. |
| `reports_dir` | Carpeta de `reports/` (esquema en `reports/README.md`). |

`context.search(model=None)` busca con `query`, `k` y `stem` actuales y devuelve un `SearchResponse` (`hits` con `rank`, `doc_id`, `title`, `text`, `score`, y `elapsed_ms`). Para comparar, llama a `context.search(m)` por cada `m` de `ris.engine.MODELS`. Lanza `EngineError` (`EmptyQueryError`, `ArtifactsMissingError`, `ModelUnavailableError`) con mensajes aptos para `st.error`.

- #14 (comparación): si `context.query` está vacío, muestra un aviso en lugar de buscar. Cada ejecución repite las búsquedas; el denso tarda unos 150 a 500 ms.
- #15 (métricas): lee de `context.reports_dir` y valida `provenance.metrics_per_query_sha256` con `scripts.evaluate.verify_reports` antes de mostrar nada. No usa `context.engine`.
- Usa los colores del tema (`.streamlit/config.toml`, paleta ciruela) y evita estilos propios que choquen con `app/frontend/styles.css`.
- Un cambio incompatible de `AppContext` o de `SearchEngine` se coordina en el PR con ambos consumidores.
