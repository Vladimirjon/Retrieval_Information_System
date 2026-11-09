# Guía de trabajo

Repositorio público universitario: [Vladimirjon/Retrieval_Information_System](https://github.com/Vladimirjon/Retrieval_Information_System).

## Antes de trabajar

Leer primero [STATE.md](STATE.md), después [TASKS.md](TASKS.md) y [docs/PLAN.md](docs/PLAN.md). Fuente de verdad: [docs/ESPECIFICACION.md](docs/ESPECIFICACION.md); leerla en lugar del PDF. [docs/DATASET.md](docs/DATASET.md) distingue datos verificados y pendientes. No modificar estos documentos fuente salvo encargo explícito.

Responder en español claro, con tildes. No inventar resultados: distinguir planeado, probado localmente, CI y comprobación del despliegue. No enviar mensajes, publicar ni entregar por cuenta del usuario sin encargo.

## Estructura prevista

- src/ris/: data, preprocessing, index, retrievers y evaluation.
- scripts/build_artifacts.py: índice clásico, embeddings e índice FAISS.
- scripts/evaluate.py: evaluación reproducible.
- app/streamlit_app.py: núcleo; app/views/comparison.py y app/views/metrics.py: vistas.
- tests/, notebooks/, reports/, docs/informe/.
- data/ y artifacts/: locales, excluidos de Git.

Estos módulos aún no existen: se implementan en las issues del plan.

## Contratos de interfaz

- ris.data: `load_corpus() -> dict[str, Doc]` (doc_id → Doc con title: str y text: str); `load_queries(split="test") -> dict[str, str]`; `load_qrels(split="test") -> dict[str, dict[str, int]]`. Todos los ids son str; filtrar consultas por qrels del split.
- ris.preprocessing: `preprocess(text: str, *, stem: bool = False) -> list[str]`. Stopwords inglesas NLTK y Porter opcional; misma configuración para corpus y consulta clásica.
- ris.index: clase InvertedIndex con postings term → {doc_id: tf}, df, longitudes y save/load con ruta. Persistir N, configuración stem y versión de formato.
- ris.retrievers: protocolo Retriever con `search(query: str, k: int = 10) -> list[Result]`, `Result(doc_id: str, score: float)`; implementaciones JaccardRetriever, TfidfRetriever, BM25Retriever y DenseRetriever.
- Rankings sin duplicados, score descendente y desempate doc_id ascendente. Consulta vacía devuelve []; k no positivo genera ValueError. El top-k puede tener menos de k documentos. El denso codifica texto original, sin el preprocesamiento léxico.
- ris.evaluation: precision_at_k(ranking, relevant, k=10), recall_at_k(ranking, relevant, k=10), average_precision(ranking, relevant, k=1000), mean_average_precision(run, qrels); `run = dict[qid, list[doc_id]]` ordenado, relevant=conjunto de ids con score>0.
- P@k divide por k, R@k y AP por todos los relevantes. MAP@1000 promedia todas las consultas de qrels; rankings ausentes aportan 0. Sin relevantes: 0. Duplicados y k no positivo: ValueError. Tests a mano en #10.
- reports/metrics_per_query.csv: model, stem, qid, k, precision, recall, ap_at_1000. reports/metrics_summary.json: configuración, conteos, versiones y promedios por model/stem. #11 documenta esquema completo antes de #15.
- #13 acuerda render(context) para las dos vistas antes de #14/#15; un solo motor carga los cuatro modelos.

Los contratos se acuerdan antes de programar en paralelo. Cambios incompatibles requieren coordinar a los consumidores en el PR.

## Decisiones obligatorias

Python 3.12, versiones exactas en requirements.txt fijadas en #1, pytest y ruff. No usar Elasticsearch, Solr, Whoosh, rank_bm25 ni otro motor completo; los modelos clásicos son propios. NumPy/SciPy dispersas permitidas.

Jaccard binario; TF-IDF: tf=1+ln(tf) si tf>0, idf=ln(N/df), normalización L2; BM25: k1=1.5 y b=0.75, variante documentada en el plan. Denso: all-MiniLM-L6-v2, 384 dimensiones, normalización y FAISS IndexFlatIP. FiQA inglés, documento=title+text. Evaluar test, clásicos con y sin Porter, P@10/R@10 y MAP@1000. No ajustar hiperparámetros con test.

No subir datos, índices, embeddings ni cachés a GitHub. Generar con scripts/build_artifacts.py; publicar artefactos en Hub del equipo y descargarlos al arrancar el Space. Solo resultados pequeños CSV/JSON en reports/. Nunca subir .env, credenciales ni datos personales. No añadir dependencias, archivos o refactorizaciones ajenas a la tarea.

## Comandos previstos

**Disponibles cuando se cierre la issue #1**; todavía no están ejecutados ni garantizados. PowerShell desde la raíz:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pip install --no-deps -e .
python -m pytest
python -m ruff check .
python -m streamlit run app/streamlit_app.py
```

La app requiere además las issues de modelos e interfaz y sus artefactos. #1 confirma estos comandos en instalación limpia. CI ejecutará lint y tests en cada PR.

## Git y revisión

- Comprobar rama y cambios reales; conservar trabajo ajeno. No hacer commit, push ni merge sin encargo.
- Ramas tipo/NN-descripcion, NN=número de issue, con cero inicial si es menor de 10; ejemplo feat/05-indice-invertido. La rama docs/01-plan-inicial corresponde a la planificación previa, no a la implementación #1.
- Conventional Commits; un PR por issue con Closes #NN y pruebas concretas. Nunca git add .; nombrar cada archivo.
- Nunca push directo a main, force push ni borrar main. Sin reset --hard, limpiezas masivas ni stash del trabajo ajeno.
- Ruleset `proteger-main` (id 26888439) verificado por API el 9 de noviembre de 2026: activo en la rama por defecto, reglas `deletion`, `non_fast_forward`, `update` y `pull_request`, 1 aprobación obligatoria, CODEOWNER, descartar aprobaciones con commits nuevos y resolver todos los hilos. El rol admin puede usar bypass en modo `pull_request`; no habilita push directo.
- Los PR de Ismael y Jorge requieren aprobación de @Vladimirjon como CODEOWNER; Vladimirjon, o Claude Code con su cuenta cuando él lo pida, hace el merge. Los PR de Vladimirjon se integran mediante bypass de admin después de pasar CI (desde #1); cuando haya tiempo, se recomienda una revisión no bloqueante de un compañero o revisión cruzada de otro modelo de IA. GitHub no permite que el autor apruebe su propio PR: [documentación](https://docs.github.com/en/pull-requests/how-tos/review-pull-requests/approving-a-pull-request-with-required-reviews).
- Borrado automático de ramas al hacer merge activado. También verificados: seis milestones M0–M5, 12 etiquetas (9 de área y 3 `resp:*`) y 18 issues #1–#18 creadas con la numeración del plan; las issues de Ismael y Jorge esperan que se unan como colaboradores para asignarlas.
- Prohibida la atribución de IA en commits, PRs, issues y documentos: ni Co-Authored-By ni “Generated with”.

## Cierre y costos

Actualizar STATE.md y TASKS.md al terminar cada tarea con evidencia y pendientes. No marcar hecho solo porque compila: verificar criterios, pruebas pertinentes y revisión. Todo terminado el martes 17 de noviembre de 2026; del 18 al 23 solo contingencia.

Space CPU básico y Colab se planifican gratis; las suscripciones de agentes usan su cupo. Modelos NVIDIA gratis solo para contenido público universitario. No activar facturación, API keys de pago, créditos ni mejoras de plan sin aprobación; informar costo y disparador de cobro antes de cambiar de modalidad.
