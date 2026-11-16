# Ejemplos y análisis comparativo de métricas

Sobre las mismas 648 consultas test de FiQA se compararon siete configuraciones:
Jaccard, TF-IDF y BM25 con y sin Porter, y MiniLM/FAISS una sola vez con texto
original. Relevante significa `score > 0` en qrels. P@10 divide por diez; R@10 y
AP@1000 por todos los relevantes; MAP@1000 promedia AP sobre todas las consultas.
Se conservaron los parámetros del proyecto, sin afinarlos con test.

| Modelo | Porter | P@10 | R@10 | MAP@1000 |
| --- | --- | ---: | ---: | ---: |
| Jaccard | no | 0,014506 | 0,057940 | 0,037585 |
| Jaccard | sí | 0,015123 | 0,059875 | 0,035224 |
| TF-IDF | no | 0,046759 | 0,203157 | 0,125020 |
| TF-IDF | sí | 0,047531 | 0,206329 | 0,132544 |
| BM25 | no | 0,062037 | 0,276610 | 0,178056 |
| BM25 | sí | 0,067747 | 0,303305 | 0,198736 |
| Denso | n/a | 0,104475 | 0,441306 | 0,311586 |

Frente al comparador fijo BM25 con Porter, el denso aumenta MAP en 0,112850
(56,78% relativo): AP mejora en 456 consultas, empeora en 148 y empata en 44
(tolerancia absoluta 1e-12). En P@10 y R@10 los conteos son 227/59/362. Porter
cambia MAP en −0,002362 para Jaccard (−6,28%), +0,007524 para TF-IDF (+6,02%) y
+0,020680 para BM25 (+11,61%). Una media mayor no implica mejora universal.

Los tipos de consulta se definieron por longitud tras el pipeline sin Porter
(0–3/4–7/≥8 tokens: n=81/409/158), presencia de dígitos (sí/no: 73/575) y formas
ASCII de dos o más mayúsculas (sí/no: 118/530; indicador imperfecto de acrónimo).
Cada familia particiona test, pero las familias se solapan. El denso obtiene la
mayor MAP media en los siete grupos. Para ≥8 tokens alcanza 0,367902, frente a
0,222311 en 0–3; BM25/Porter alcanza 0,225478 y 0,120893, respectivamente. Estas
comparaciones no aíslan longitud ni dificultad. TF-IDF con Porter baja en el
grupo corto (0,062396 → 0,056484), aunque mejore globalmente. No hubo etiquetado
temático manual ni pruebas de significancia; los hallazgos son descriptivos.

Los ejemplos se eligieron como máximo y mínimo de AP@1000 denso−BM25/Porter,
con desempate lexicográfico por `qid`, entre las 648 consultas. En **1074**,
*How common is “pass-through” health insurance?*, el relevante `443960` explica
el plan *self-funded*: el denso lo coloca primero (P@10=0,1, R@10=1, AP=1) y
BM25 no lo recupera en top-1000 (tres métricas cero). El preprocesamiento elimina
`pass-through`; esa pérdida se observa, pero no se ha demostrado su efecto
causal mediante una ablación. En **1310**, *Is is possible to take a mortgage
using Bitcoin as collateral?*, el relevante `535651` describe bitcoins como
garantía y sus riesgos: BM25 lo coloca primero (0,1/1/1), el denso en 113
(0/0/0,00884956). Los extremos ilustran contrastes y no representan la consulta
típica. El primer denso, `101103`, define la hipoteca como deuda garantizada por
una propiedad y no tiene juicio para esta consulta; el relevante describe de
forma explícita bitcoins como garantía. Los scores solo se interpretan dentro
de cada modelo.

Los qrels pueden ser incompletos: no juzgado se distingue de juzgado no
relevante. La verificación clásica original usa una muestra de 500 documentos,
no una garantía total; los dos índices reconstruidos aquí tienen los mismos
hashes de #11. Los hashes NLTK describen recursos de evaluación, no certifican
por sí solos la construcción original. El pipeline descarta ciertos guiones,
decimales y contracciones; MiniLM limita su entrada a 256 tokens. No se atribuyen
causas a estas limitaciones sin experimentos específicos. La reproducción bit a
bit del denso depende del entorno; se registran versiones y hashes locales.
Los embeddings y FAISS reconstruidos tienen hashes distintos de los originales
de #11; los cuatro rankings de los casos reproducen sus métricas con tolerancia
1e-12. No se afirma identidad de vectores ni se reevalúan los agregados: estos
proceden del CSV original validado.

Fuentes y evidencia: [notebook ejecutado](../../notebooks/12_analisis_modelos.ipynb),
[resumen estructurado](../../reports/comparison_summary.json),
[métricas originales](../../reports/metrics_summary.json),
[reproducción](../../notebooks/README.md) y [datos fijados](../DATASET.md).
Este texto es un aporte para #17; no constituye el informe final ni verifica el
despliegue.
