# Tareas

Plan base: 9 de noviembre de 2026. Las issues #1–#18 están creadas en GitHub con esta numeración. Actualizado el 12 de noviembre de 2026: #1 hecha (PR #21); #2 hecha (PR #23); #3 hecha (PR #24); #4 hecha con el [PR #25](https://github.com/Vladimirjon/Retrieval_Information_System/pull/25) (22 pruebas locales, notebook ejecutado, CI en verde, revisada por @Vladimirjon). #5 hecha (índice invertido y construcción con progreso; 27 pruebas locales). #6 hecha con el [PR #27](https://github.com/Vladimirjon/Retrieval_Information_System/pull/27) (39 pruebas locales, CI en verde, revisada por @Vladimirjon). #7 hecha con el [PR #28](https://github.com/Vladimirjon/Retrieval_Information_System/pull/28) (57 pruebas locales, CI en verde); siguiente: #8. Reparto y plazos: acuerdo del equipo en [PLAN](docs/PLAN.md).

| Número | Título | Responsable | Milestone | Depende de | Estado |
|---|---|---|---|---|---|
| #1 | chore: estructura, entorno y CI | Vladimirjon | M0 Base y datos | Ninguna | hecho |
| #2 | feat: carga reproducible de FiQA y qrels de test | Vladimirjon | M0 Base y datos | #1 | hecho |
| #3 | feat: preprocesamiento en inglés con stemming opcional | Ismael Freire | M0 Base y datos | #1 | hecho |
| #4 | docs: EDA del corpus y consultas FiQA | Jorge Torres | M0 Base y datos | #2, #3 | hecho |
| #5 | feat: índice invertido con persistencia | Vladimirjon | M1 Modelos clásicos | #2, #3 | hecho |
| #6 | feat: recuperación Jaccard con vectores binarios | Ismael Freire | M1 Modelos clásicos | #5 | hecho |
| #7 | feat: recuperación por coseno TF-IDF propia | Vladimirjon | M1 Modelos clásicos | #5 | hecho |
| #8 | feat: recuperación BM25 propia | Vladimirjon | M1 Modelos clásicos | #5 | pendiente |
| #9 | feat: embeddings y búsqueda exacta con FAISS | Vladimirjon | M2 Recuperación semántica | #2 | pendiente |
| #10 | feat: métricas de recuperación con ejemplos manuales | Ismael Freire | M3 Evaluación y comparación | #1 | pendiente |
| #11 | feat: evaluación reproducible de los cuatro modelos | Vladimirjon | M3 Evaluación y comparación | #6, #7, #8, #9, #10 | pendiente |
| #12 | docs: análisis comparativo con consultas concretas | Jorge Torres | M3 Evaluación y comparación | #11 | pendiente |
| #13 | feat: núcleo Streamlit y carga unificada de modelos | Vladimirjon | M4 Interfaz y despliegue | #6, #7, #8, #9 | pendiente |
| #14 | feat: vista de comparación lado a lado | Ismael Freire | M4 Interfaz y despliegue | #13 | pendiente |
| #15 | feat: vista de métricas desde reports | Jorge Torres | M4 Interfaz y despliegue | #11, #13 | pendiente |
| #16 | chore: despliegue gratuito en Hugging Face Spaces | Vladimirjon | M4 Interfaz y despliegue | #9, #13, #14, #15 | pendiente |
| #17 | docs: informe técnico de hasta cinco páginas | Jorge Torres | M5 Informe y entrega | #4, #12, #16 | pendiente |
| #18 | docs: QA final, README y entrega completa | Vladimirjon | M5 Informe y entrega | #16, #17 | pendiente |

Actualizar junto con [STATE](STATE.md) al terminar cada tarea. Estados: pendiente, en curso, hecho, bloqueado. Ismael Freire = @BrandonFreire; Jorge Torres = @Almejandro04.
