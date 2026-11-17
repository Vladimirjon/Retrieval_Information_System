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
| `app/views/comparison.py` | Vista reservada para #14. |
| `app/views/metrics.py` | Métricas guardadas: resumen, comparación global, detalle por qid y procedencia (#15). |
| `app/metrics_data.py` | Lectura y validación completa del par CSV/JSON, sin usar el motor. |
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

## Métricas (#15)

Abra **Métricas**; funciona sin consulta libre. El selector **Sin Porter / Con Porter /
Comparar** y los cuatro botones múltiples **Métodos visibles** afectan al gráfico, las tablas globales,
el detalle por qid y el listado de consultas. Con todos los métodos hay cuatro
configuraciones en cada variante individual y siete al comparar. Embeddings se
mantiene una sola vez como referencia «No aplica»: no existe su versión Porter.
Los filtros se conservan al cambiar de vista durante la sesión. **Restablecer filtros**
recupera Comparar, los cuatro métodos, MAP@1000 y el primer qid. Cada método
se activa o desactiva con un clic; los botones pueden pasar a varias líneas en móvil.
Al no haber un desplegable, seleccionar los cuatro no deja una lista vacía abierta.
Si no hay métodos seleccionados, **Mostrar todos los métodos** recupera únicamente
ese filtro. **Limpiar filtro de consultas** borra solo la búsqueda de qids.

La jerarquía es controles → cobertura → gráfico/tabla global → detalle por consulta.
P@k y R@k globales son medias sobre todos los qids; MAP@1000 es la media de AP@1000.
El detalle muestra **AP**, no MAP. Los filtros no recalculan ni modifican las medias.
El buscador y sus opciones no filtran la evaluación.

Las ayudas junto a encabezados y controles son nativas de Streamlit: admiten cursor,
foco del teclado y toque. Las columnas también explican sus métricas. **Cómo interpretar
estos resultados** reúne las explicaciones largas. Las tablas se ordenan pulsando sus
encabezados, usan seis decimales y resaltan en negrita los máximos positivos visibles
(con empates). Los máximos se determinan antes del redondeo; las columnas
completamente a cero no se resaltan. Junto al gráfico se resume el máximo de la
métrica elegida entre las configuraciones visibles; el detalle por consulta agrupa
selector, tabla y diferencias en un panel propio.
El listado opcional **Explorar todas las consultas** permite filtrar qids por coincidencia
literal sin afectar las medias. Los encabezados quedan fijos al recorrer ese listado;
el qid permanece fijado en ese listado, y el método en las tablas global y por
consulta. Una sola columna fija deja más espacio útil al desplazar horizontalmente
en móvil. Las tablas permiten pantalla completa.

Con el modo **Comparar** de Porter, los paneles de diferencias se cargan al abrirlos. Muestran, para la
métrica elegida, **Con Porter − Sin Porter** del mismo método, corte y consultas.
Son diferencias absolutas, no porcentajes ni pruebas de significancia. El signo y
«Aumenta/Disminuye/Sin cambio» acompañan los valores; Embeddings queda excluido.
Una variante ausente nunca se convierte en cero: un reporte incompleto se rechaza;
la función de comparación representa la ausencia sin calcular una diferencia.

Los controles globales y el detalle usan fragmentos anidados de Streamlit: cambiar
un qid solo actualiza su sección. No hay motor adicional ni nueva dependencia.
El detalle identifica la consulta con un título y un resaltado breve al cambiarla
(desactivado si se prefiere movimiento reducido).
Al salir de Métricas, el navegador oculta esa vista de inmediato mientras Python
termina de retirar su contenido. Al recibir datos o navegar también elimina anclas
antiguas de la URL, sin recargar ni añadir entradas al historial.
Las transiciones Métricas ↔ Comparar funcionan aunque ambas pestañas usen el modo
«resultados». La cabecera actualiza la vista dentro de `startViewTransition`,
cancela transiciones anteriores y conserva el último clic. Cada vista nativa monta
su propio contenedor, con una entrada de 260 ms y 8 px; los filtros conservan ese
contenedor. El movimiento reducido desactiva ambos tipos de animación.

Los cortes, split, profundidad y conteos proceden del JSON validado. El esquema 1
admite test, AP/MAP@1000 y cualquier k entero positivo hasta `depth` (que debe ser
al menos 1000); las etiquetas de P/R usan ese k. El gráfico conserva la escala
[0,1] y distingue las variantes con etiquetas y colores constantes, sin depender solo del color. Las tablas
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

Al entrar se valida el par completo; una segunda lectura detecta cambios durante
la validación. Se retiró el botón permanente de recarga porque repetía esa operación.
**Reintentar carga** aparece únicamente si los reportes fallan. En las interacciones de cada
fragmento se releen ambos archivos y se comparan sus bytes con el par ya validado.
Si siguen idénticos, se reutilizan los datos de esa vista sin volver a analizar
4.536 filas. Si cambian o falta alguno, se fuerza una ejecución completa para
validar de nuevo y actualizar también el resumen y la procedencia. No hay caché
global ni se depende de fechas o tamaños de archivo. Si el par es inválido, se
muestra el motivo y cómo restaurarlo; no se presentan tablas/gráficos parciales
ni resultados anteriores. La vista nunca reescribe los reportes.

Para revisar errores sin alterar los originales, copie ambos archivos a una
carpeta temporal dentro de `artifacts/`, establezca `JAB_REPORTS_DIR` con esa ruta
antes de arrancar Streamlit y modifique solo las copias. Restaure ambos archivos
válidos y pulse **Reintentar carga**. Las pruebas deterministas de lectura,
presentación y recuperación están en `tests/test_metrics.py` y usan `AppTest` de
Streamlit; no necesitan red, corpus ni modelos.

## Decisiones visuales y alcance del seguimiento de #15

### Colores de las barras

El gráfico codifica la **variante de preprocesamiento**, con una asignación fija:

| Variante | Color | Motivo |
|---|---|---|
| Sin Porter | Ciruela `#7A2E5E` | Conserva el acento aprobado para la referencia clásica sin stemming. |
| Con Porter | Azul verdoso `#246477` | Distingue la segunda condición experimental al comparar el mismo método. |
| No aplica | Gris `#5E5560` | Identifica la referencia única de Embeddings, que no usa Porter. |

El segundo color sirve para comparar variantes contiguas; no señala al ganador ni
implica que Porter mejore el resultado. La asignación no cambia al filtrar métodos
o elegir otra métrica. Las etiquetas, los tooltips y la tabla aportan el mismo
significado sin depender solo del color. La escala sigue siendo [0,1] y los valores
proceden de los reportes originales. El azul verdoso se limita a la serie de datos;
la identidad de la interfaz conserva la ciruela.

### Tablas y controles

Cabeceras gris claro, bordes definidos, texto oscuro y filas de 40 px facilitan
seguir los datos. La negrita destaca valores positivos máximos sin cambiar las
cifras. El panel blanco del detalle separa la consulta de los resultados globales;
las explicaciones extensas están en ayudas y desplegables para reducir texto repetido.

El resaltado de fila al pasar el puntero es el nativo de Streamlit. Se conserva
evitando fondos por celda que lo oculten y sin añadir selección persistente a la
aplicación; el foco del teclado mantiene su indicador. El contraste calculado de
texto es 14,19:1 en cabeceras y 16,39:1 en el cuerpo. Estas medidas no sustituyen
una prueba visual o con lector de pantalla.

### Cambios compartidos adicionales

La issue original pedía mostrar tablas y gráficos, distinguir configuraciones y
validar los reportes. Los filtros persistentes, las diferencias de Porter y las
acciones de recuperación amplían su usabilidad; no son nuevos requisitos académicos
ni vuelven a evaluar test.

Las transiciones, la eliminación de anclas antiguas y los contenedores por pestaña
tocan la navegación compartida de #13 para integrar Métricas con el resto de la app.
No implementan la comparación de rankings de #14 ni cambian el buscador, el motor
o `render(context)`.

Las opciones `dataframeBorderColor`, `dataframeHeaderBackgroundColor` y
`dataframeHeaderTextColor` afectan a las tablas nativas. El ajuste de
`secondaryBackgroundColor` a `#DCD8DC` también se hereda en controles nativos y
refuerza el contraste del hover. Este efecto es global al tema y debe revisarse
junto con las otras vistas nativas; el fondo de página, el color primario y el
CSS del buscador personalizado se mantienen. No se añaden dependencias ni se
modifican `reports/`, modelos, índices o fórmulas.
