# Descripción del corpus utilizado

FiQA, en su versión BEIR en inglés y de dominio financiero, contiene 57 638 documentos y 6 648 consultas globales. El split test comprende 648 consultas distintas y 1 706 pares juzgados, todos con score 1. Los 57 638 títulos están vacíos; se conservan los 38 textos vacíos.

Tras el pipeline inglés acordado (NLTK, minúsculas, tokens alfanuméricos y stopwords), los documentos tienen media 66.61, mediana 46.0 y percentil 99 319.00 tokens. Las consultas test tienen media 6.05 tokens. El vocabulario documental contiene 66 796 términos; Porter lo reduce a 46 821 (29.90% respecto de la versión sin Porter), sin alterar la cantidad de apariciones. Esta reducción no demuestra una mejora de recuperación.

Hay 1 706 documentos relevantes test distintos. Por consulta se observan entre 1 y 15 relevantes (media 2.633, mediana 2.0); 404 consultas tienen como máximo dos. El techo macro teórico de P@10 según estos qrels es 0.2605; no es el resultado de un recuperador. Los juicios disponibles no establecen la irrelevancia de todos los documentos sin juzgar. Hay 1 documento relevante sin texto, un caso que debe inspeccionarse al evaluar la evidencia textual disponible. La efectividad se medirá posteriormente mediante P@10, R@10 y MAP@1000, sin ajustar hiperparámetros con test.

Fuentes: [corpus y consultas](https://huggingface.co/datasets/BeIR/fiqa), [qrels](https://huggingface.co/datasets/BeIR/fiqa-qrels), [revisiones y hashes](../DATASET.md), [notebook ejecutado](../../notebooks/04_eda_fiqa.ipynb) y [agregados reproducibles](../../reports/eda_summary.json). Fundamento conceptual: Iván Carrera, *Fundamentos de Recuperación de Información* (pp. 20–33), *Modelo vectorial* (pp. 5–15, 24) y *Modelo BM25* (pp. 4–8, 14), material docente EPN.
