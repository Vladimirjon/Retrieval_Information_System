# App jab (issue #13)

Buscador sobre FiQA con los cuatro modelos. Desde la raíz del repositorio:

```powershell
python -m streamlit run app/streamlit_app.py
```

Requiere los artefactos de `scripts/build_artifacts.py` (`artifacts/index_no_stem.json`, opcionalmente `index_stem.json`, y `artifacts/dense/`) y el corpus FiQA en la caché de Hugging Face. Variables opcionales: `JAB_ARTIFACTS_DIR` y `JAB_REPORTS_DIR`. Si no se define `JAB_ARTIFACTS_DIR` y existe `deploy/artifacts.json`, el arranque descarga desde la revisión fijada los archivos ausentes o con hash incorrecto y comprueba cada SHA-256. Sin manifiesto, se usan los artefactos locales. Si falta un archivo tras la preparación, la interfaz muestra qué falta y el comando que lo genera; los demás modelos siguen funcionando. En Community Cloud la primera visita puede descargar unos 250 MB antes de cargar los modelos.

Community Cloud instala `app/requirements.txt`, con las dependencias de ejecución; el archivo de la raíz se conserva para desarrollo y CI.

## Capas

| Archivo | Responsabilidad |
|---|---|
| `src/ris/engine.py` | `SearchEngine`: único punto de carga y búsqueda. Sin Streamlit. |
| `app/frontend/` | Interfaz de búsqueda: `index.html`, `styles.css` (sistema de diseño, paleta ciruela) y `app.js`. |
| `app/ui.py` | Registra el componente, valida lo que envía el navegador y arma la respuesta. |
| `app/streamlit_app.py` | Página, caché del motor y vistas de los compañeros. |
| `app/context.py` | `AppContext`: lo que reciben las vistas. |
| `app/views/comparison.py` | Comparar: los cuatro rankings lado a lado, etiquetas de cobertura y panel de solapamiento (#14). |
| `app/overlap.py` | Solapamiento entre rankings (funciones puras, sin Streamlit; #14). |
| `app/views/metrics.py` | Métricas guardadas: resumen, comparación global, detalle por qid y procedencia (#15). |
| `app/metrics_data.py` | Lectura y validación completa del par CSV/JSON, sin usar el motor. |
| `app/static/` | Logo `jab` (palabra e icono) en SVG. |

La referencia visual aprobada está en `design/prototipo/` y sus reglas en `design/README.md`.

En la vista Buscar, `app/ui.py` llama a `SearchEngine.search(..., skip_empty=True)` para omitir documentos sin título ni texto y completar el top-n con los siguientes resultados. El valor por defecto de `skip_empty` es `False`: `AppContext.search()` y las evaluaciones conservan el ranking original. La respuesta del componente incluye `terms` y, por hit clásico, `matched` y `spans` (tramos `[inicio, fin]` sobre `text` limpio) para tooltips y resaltado. La paginación de cinco resultados es exclusivamente visual; los ranks globales no cambian.

## Cómo se comunican navegador y Python

El componente envía un único estado `request = {q, model, k, stem, graph, view, seq}`. Cada cambio sube `seq`, así ningún cambio se pierde aunque llegue mientras Python atiende otro. Python valida todo (modelo y vista en lista blanca, `k` en 5/10/20/50, consulta de 300 caracteres como máximo, `graph` solo si llega `true`), busca y responde con el mismo `seq` y una `key` de la búsqueda. La interfaz descarta respuestas de un `seq` viejo y no repinta si solo cambió la vista. Si detecta que Python atendió un estado anterior, lo reenvía una vez.

## Rutas y URL

El mismo estado vive en la URL, para recargar, compartir o usar atrás y adelante:

| Ruta | Vista |
|---|---|
| `/` | Inicio; mientras se escribe, los resultados en vivo aparecen aquí (`/?q=…`) |
| `/search` | Búsqueda confirmada (Enter, una sugerencia, un ejemplo o «Quizás quisiste decir») |
| `/comparison` | Comparar (#14) |
| `/metrics` | Métricas (#15) |

Parámetros: `q`, `model` (`jaccard`, `tfidf`, `bm25` o `embeddings`), `results` (5, 10, 20 o 50), `stemming` (`true`/`false`; no aparece con Embeddings) y `graph` (`true`/`false`). En el inicio solo aparecen los que difieren de lo predeterminado. Ejemplo: `/search?q=index+funds&model=bm25&results=10&stemming=false&graph=true`.

Cada ruta es una página de `st.navigation` (oculta) sin contenido propio; la interfaz se dibuja en `main()`. Cambiar de página con `st.switch_page` vuelve a montar el componente (comprobado en el navegador: se pierden foco y texto a medio escribir), por eso escribir nunca cambia de página: el estado lleva `route` (`home` o `search`) y solo una acción explícita lo pasa a `search`; el logo vuelve a `home`. En `/search`, borrar la consulta deja la vista de resultados vacía en vez de volver al inicio. Tras Enter, el campo recupera el foco al volver a montarse.

`resolve_state` (en `app/ui.py`) decide qué estado manda en cada ejecución: si el navegador envió un `seq` mayor, el del navegador, y Python mueve la URL con `st.switch_page` o `st.query_params`; si la URL cambió sin que lo hiciera la app (recargar, atrás, un enlace), la URL, con un `seq` nuevo y `nav` + 1 para que el componente la adopte. Solo cuenta como cambio una URL distinta de la vista en la ejecución anterior: una ejecución pedida antes de procesar un `switch_page` trae la URL vieja y no debe deshacer lo último que hizo el usuario.

## Escritura asistida

- **Campo como en Google:** es un `textarea` de una línea que, con el foco, crece hasta mostrar toda la consulta (hasta 5 líneas; luego desplaza) y sin foco vuelve a una línea. Enter busca; los saltos de línea al escribir o pegar se convierten en espacios; máximo 300 caracteres. Borrar con ×, aceptar el gris, pegar o elegir una sugerencia pasan por el historial nativo de deshacer (`editInput` con `execCommand` y respaldo), así que Ctrl+Z y Ctrl+Y funcionan.
- **Autocompletado en gris:** mientras se escribe, aparece en gris lo que falta de la primera consulta de FiQA (train) que empieza igual; si no hay, se completa la última palabra con las 12 000 más frecuentes del índice (`SearchEngine.top_terms`). Se dibuja en un espejo superpuesto con las mismas medidas que el campo, así queda alineado también con varias líneas, y el campo crece para mostrar entera la sugerencia. Si lo escrito está en minúsculas, el resto también («he» → «heloc vs. …», no «heLOC»). Tab o flecha derecha lo aceptan.
- **Quizás quisiste decir:** `SearchEngine.correct_query` corrige palabras en minúscula que no están en el índice sin Porter, o que están en menos de 10 documentos cuando hay una forma parecida al menos 20 veces más frecuente (FiQA tiene erratas como «morgage»). Puntúa ln(df) − 2 × distancia de edición (hasta 1 en palabras de hasta 7 letras, 2 en las más largas) y completa palabras empezadas («diabet» → diabetes). No toca palabras con mayúsculas (siglas y nombres propios). Medido sobre las 648 consultas de test: sugiere en 5 (3 erratas reales y 2 discutibles: «excl» → excel, «dishonour» → dishonor).

## Grafo 3D de la consulta (opcional)

Se activa en el menú del chip («Grafo 3D»; apagado por defecto). Con él activo, cada búsqueda trae además `graph`, calculado en ese momento por `SearchEngine.query_graph` (`src/ris/graph.py`):

- Sitúa los 50 primeros resultados del modelo elegido: los del top-n elegido como nodos numerados y el resto, hasta 50, como puntos tenues «alrededor». Con top-n 50 todos son nodos.
- Usa siempre los embeddings MiniLM, sea cual sea el modelo del ranking: sitúa los resultados por significado y no cambia su orden. Codifica la consulta y lee los vectores ya guardados en FAISS; no toca el corpus. Medido en local con los artefactos reales: 12–14 ms para 50 resultados.
- Posición que calcula Python: PCA a 3D de la consulta y los documentos; la consulta queda en el origen y el punto más lejano a distancia 1. `variance` dice qué fracción de la variación conserva esa proyección (22–26 % con 50 resultados en las consultas probadas). El navegador no la dibuja tal cual (ver más abajo): de ella solo toma la dirección, así que la interfaz no cita ese porcentaje.
- Parecido: coseno entre los vectores originales, no entre las posiciones 3D. Con Embeddings coincide con la puntuación del modelo (diferencia máxima medida 2,6e-7).
- Relaciones: la consulta con cada nodo mostrado, y cada nodo mostrado con sus dos mostrados más parecidos (desempate por `doc_id`); los puntos de alrededor no tienen relaciones.

Formato: `graph = {nodes: [[x, y, z, coseno]], edges: [[i, j, coseno]], shown, context: [{rank, doc_id, text}], variance, elapsed_ms}`, con `nodes[i]` en el orden del ranking (los `shown` primeros son los de `hits`), o `{error}` si el modelo denso no está disponible (los resultados se muestran igual).

El navegador pinta en un `<canvas>`, sin bibliotecas ni rejillas. Para que la nube no quede amontonada junto a la consulta, la posición que se dibuja no es la del PCA tal cual: la dirección es la del PCA quitándole el 80 % de la dirección común (para que la nube rodee la consulta) y la distancia al centro sigue el puesto de cada documento según 1 − coseno con la consulta (más lejos, menos parecido; empates al mismo radio, entre 0,22 y 1), con una separación suave para que no se tapen los números. Los nodos llevan degradado y un tono que va del gris tinta al ciruela según su parecido, y la consulta un halo. El encuadre se calcula para que todo quepa al girar; «Recentrar» vuelve a él. Leyenda: «Tu lista (1–N)» y «Puestos N+1–50». Arrastrar gira; la rueda o dos dedos acercan; Mayús + arrastrar, clic derecho o el modo «Mover» desplazan. Mientras se manipula, una pastilla dice «Girando», «Moviendo» o «Zoom 1,6×», y el cursor lo fija una sola función. «Girar solo» es una preferencia: manejar el grafo lo pausa y vuelve a girar unos 2 s después de la última interacción. Los controles tipo mapa (acercar, alejar, mover, girar solo, recentrar) dicen qué hacen al pasar el puntero. Pasar por un punto o por un resultado de la lista resalta sus relaciones con su parecido; pasar o pulsar una relación muestra su número. Pulsar un nodo o una relación centra y acerca el grafo en ello con una animación corta y lo deja resaltado; si es un nodo de la lista, además marca su fila. Pulsar en vacío quita el resaltado. En pantallas de 1100 px o más va al lado de la lista; en las demás, encima. Con `prefers-reduced-motion` no gira solo.

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

`context.search(model=None, *, skip_empty=False)` busca con `query`, `k` y `stem` actuales (con `skip_empty=True` omite los documentos vacíos como la vista Buscar) y devuelve un `SearchResponse` (`hits` con `rank`, `doc_id`, `title`, `text`, `score`, y `elapsed_ms`). Para comparar, llama a `context.search(m)` por cada `m` de `ris.engine.MODELS`. Lanza `EngineError` (`EmptyQueryError`, `ArtifactsMissingError`, `ModelUnavailableError`) con mensajes aptos para `st.error`.

- #14 (comparación): si `context.query` está vacío, muestra un aviso en lugar de buscar. Cada ejecución repite las búsquedas; el denso tarda unos 150 a 500 ms.
- #15 (métricas): lee de `context.reports_dir` y valida `provenance.metrics_per_query_sha256` con `scripts.evaluate.verify_reports` antes de mostrar nada. No usa `context.engine`.
- Usa los colores del tema (`.streamlit/config.toml`, paleta ciruela) y evita estilos propios que choquen con `app/frontend/styles.css`.
- Un cambio incompatible de `AppContext` o de `SearchEngine` se coordina en el PR con ambos consumidores.

## Métricas (#15)

Abra **Métricas** en la navegación de jab; funciona con la barra de búsqueda vacía.
Los filtros **Sin Porter**, **Con Porter** y **Comparar** y los botones de métodos
se aplican al gráfico, las tablas y el detalle. Con todos los métodos se muestran
siete configuraciones al comparar, o cuatro por variante individual. Embeddings
aparece una vez como «No aplica». **Restablecer filtros** recupera las opciones
iniciales. Los filtros se conservan durante la sesión al cambiar de vista.

Elija P@k, R@k o MAP@1000 para el gráfico global, o un **qid** para ver P@k,
R@k y **AP@1000** de una consulta. Las medias globales incluyen todos los qids
y se mantienen al cambiar de consulta o filtrar el listado. Las opciones del
buscador son independientes de la evaluación.

Las tablas se ordenan por encabezado y muestran seis decimales. La negrita
señala los máximos positivos visibles, incluidos empates. En **Comparar** los
desplegables muestran la diferencia absoluta **Con Porter − Sin Porter** para
pares del mismo método y corte. **Explorar todas las consultas** permite filtrar
el listado por identificador. Las ayudas nativas y **Cómo interpretar estos
resultados** explican las métricas y los métodos.

Al salir de Métricas, el panel se oculta en cuanto se activa el destino.
Comparar y Métricas tienen una entrada breve del panel y el logo, que se
omite con movimiento reducido. Las rutas y parámetros compartibles siguen
el mecanismo descrito en **Rutas y URL**.

Los cortes, split, profundidad y conteos proceden del JSON validado. El esquema 1
admite test, AP/MAP@1000 y cualquier k entero positivo hasta `depth` (que debe ser
al menos 1000); las etiquetas de P/R usan ese k. El gráfico conserva la escala
[0,1] y distingue las variantes con etiquetas, sin depender del color. Las tablas
muestran seis decimales; **Procedencia y configuración** permite inspeccionar
conteos, parámetros, revisiones, hashes, versiones y descargar ambos originales
con todos sus decimales.

Antes de mostrar resultados, `app.metrics_data.load_reports` ejecuta
`scripts.evaluate.verify_reports` y complementa su comprobación de hash/líneas:
valida estructura/tipos, modelos/stem, cortes, campos CSV, valores finitos en
[0,1], duplicados, cobertura idéntica, hash de qids, conteos y medias CSV/JSON
(`atol=1e-12`, `rtol=0`). No descarga qids, no usa corpus/índices ni reevalúa test.
Las consultas declaradas sin relevantes deben tener las tres métricas a cero
en todas las configuraciones. Los conteos de relevancia solo se contrastan por coherencia interna: verificar
de nuevo los qrels y los artefactos originales corresponde a la evaluación.

Al entrar o cambiar un filtro global se valida el par completo. Las interacciones
del detalle comprueban los bytes de ambos archivos y reutilizan los datos
validados mientras sean idénticos. Si el par cambia, la página vuelve a validarlo.
Si falla la validación, la vista muestra el motivo y retira los resultados.
**Reintentar carga** vuelve a leer los reportes después de restaurar el par.

Para revisar errores sin alterar los originales, copie ambos archivos a una
carpeta temporal dentro de `artifacts/`, establezca `JAB_REPORTS_DIR` con esa ruta
antes de arrancar Streamlit y modifique solo las copias. Restaure ambos archivos
válidos y pulse **Reintentar carga**. Las pruebas deterministas de lectura,
presentación y recuperación están en `tests/test_metrics.py` y usan `AppTest` de
Streamlit; no necesitan red, corpus ni modelos.

## Comparar (#14)

Abra **Comparar** en la navegación de jab: usa la consulta, el top-n y Porter de la barra (el modelo elegido no cambia nada, siempre salen los cuatro). Sin consulta muestra un aviso y no busca. Cada ejecución repite las cuatro búsquedas con `context.search(m, skip_empty=True)`, así que los documentos son los mismos que en Buscar (sin documentos vacíos y completando hasta k); si un modelo falla (p. ej. Embeddings aún cargando) su columna lo explica y las demás se muestran.

- **Cuatro columnas** (dos en pantallas medianas y una en teléfono): modelo, tiempo de la búsqueda, y por documento posición, `doc_id`, score propio del modelo, fragmento de 200 caracteres (texto limpiado con `app.ui.clean_text`) y «Ver texto completo». Los scores no son comparables entre modelos y no se dibujan como barras.
- **Etiquetas de cobertura:** «En n de 4» (relleno ciruela si salió en los cuatro, suave si en 2 o 3) y «Solo aquí» (borde gris) para lo que recuperó un único modelo; los puntos repiten n y el texto de ayuda nombra a los otros modelos.
- **Solapamiento** (`app/overlap.py`, funciones puras): para cada par, la intersección de `doc_id` de sus rankings. **Denominador: el menor de los dos tamaños**, es decir, el máximo que podrían coincidir; vale k con rankings completos y no castiga a un modelo clásico que devolvió menos de k documentos. Si un ranking está vacío el denominador es 0 y se muestra «—». Seis tarjetas por pares, en dos grupos (léxicos entre sí, léxicos frente a Embeddings) con la media de cada grupo.
- **Estilo:** tarjetas y colores con los tokens de `styles.css` (`:root`) con la paleta ciruela como respaldo; el CSS de la vista va acotado a `.cmp` y solo ensancha `.st-key-jab_views` mientras se muestra Comparar.
- Pruebas: `tests/test_comparison_view.py` (el `tests/test_comparison.py` existente es de #12).
