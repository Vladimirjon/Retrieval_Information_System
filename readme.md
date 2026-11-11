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

La estructura de implementación aún está pendiente. Los artefactos se generan una vez, se publican en un repo de HF Hub del equipo y el Space los descarga al arrancar; no se guardan en GitHub.

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

La aplicación Streamlit se implementará en la issue #13. Cuando exista `app/streamlit_app.py`, se iniciará con:

```powershell
python -m streamlit run app/streamlit_app.py
```

**URL del despliegue: pendiente (issue de despliegue #16).** Destino acordado: Hugging Face Spaces CPU básico, gratis; cualquier modalidad de pago requiere aprobación. El posible uso de Render con app reducida es una inferencia sin probar.

## Cómo contribuir

Leer STATE, TASKS y AGENTS antes de trabajar. Tomar una issue y usar rama tipo/NN-descripcion, con NN=número de issue y Conventional Commits. Un PR por issue con Closes #NN, pruebas y STATE/TASKS actualizados. Nunca git add . ni atribución de IA en commits/PRs.

El ruleset `proteger-main` (id 26888439), verificado por API el 9 de noviembre de 2026, exige PR y una aprobación del CODEOWNER @Vladimirjon; los commits nuevos invalidan aprobaciones y los hilos deben resolverse. El rol admin tiene bypass en modo `pull_request`, sin habilitar push directo; también están activadas las reglas contra borrado y force push, y la eliminación automática de ramas al integrar. Los PR de Ismael y Jorge requieren aprobación de Vladimirjon como CODEOWNER; él, o Claude Code con su cuenta cuando se lo pida, hace el merge. Los PR de Vladimirjon se integran con bypass de admin después de pasar CI (desde #1); cuando haya tiempo, se recomienda una revisión no bloqueante de un compañero o revisión cruzada de otro modelo de IA. GitHub no permite que el autor apruebe su propio PR: [documentación](https://docs.github.com/en/pull-requests/how-tos/review-pull-requests/approving-a-pull-request-with-required-reviews).

## Estado y fechas

Planificación local al 9 de noviembre de 2026; implementación pendiente. **Todo terminado el martes 17 de noviembre**; entrega oficial el 23. Del 18 al 23 solo contingencia.

- [Plan completo y mapa de requisitos](docs/PLAN.md)
- [Estado actual](STATE.md)
- [Tareas y responsables](TASKS.md)
- [Instrucciones de trabajo](AGENTS.md)
