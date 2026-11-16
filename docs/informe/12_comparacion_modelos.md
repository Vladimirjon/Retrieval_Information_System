# Ejemplos y análisis comparativo de métricas

Sobre las mismas 648 consultas test de FiQA se compararon siete configuraciones:
Jaccard, TF-IDF y BM25 con y sin Porter, y MiniLM/FAISS una sola vez con texto
original. Relevante significa `score > 0` en qrels. P@10 divide por diez; R@10 y
AP@1000 por todos los relevantes; MAP@1000 promedia AP sobre todas las consultas.
Se conservaron los parámetros del proyecto, sin afinarlos con test. Cifras
regeneradas el 16 de noviembre tras corregir el preprocesamiento de #3 (los
tokens con guion, punto o barra se parten en lugar de descartarse).

| Modelo | Porter | P@10 | R@10 | MAP@1000 |
| --- | --- | ---: | ---: | ---: |
| Jaccard | no | 0,015432 | 0,063079 | 0,038520 |
| Jaccard | sí | 0,015123 | 0,060502 | 0,038461 |
| TF-IDF | no | 0,049074 | 0,217460 | 0,133028 |
| TF-IDF | sí | 0,050463 | 0,219553 | 0,139718 |
| BM25 | no | 0,063735 | 0,284832 | 0,180993 |
| BM25 | sí | 0,069444 | 0,313021 | 0,206219 |
| Denso | n/a | 0,104475 | 0,441306 | 0,311586 |

Frente al comparador fijo BM25 con Porter, el denso aumenta MAP en 0,105367
(51,09% relativo): AP mejora en 443 consultas, empeora en 156 y empata en 49
(tolerancia absoluta 1e-12). En P@10 y R@10 los conteos son 219/60/369. Porter
cambia MAP en −0,000058 para Jaccard (−0,15%), +0,006689 para TF-IDF (+5,03%) y
+0,025227 para BM25 (+13,94%). Una media mayor no implica mejora universal.

Los tipos de consulta se definieron por longitud tras el pipeline sin Porter
(0–3/4–7/≥8 tokens: n=62/402/184), presencia de dígitos (sí/no: 73/575) y formas
ASCII de dos o más mayúsculas (sí/no: 118/530; indicador imperfecto de acrónimo).
Cada familia particiona test, pero las familias se solapan. El denso obtiene la
mayor MAP media en los siete grupos. Para ≥8 tokens alcanza 0,358192, frente a
0,192947 en 0–3; BM25/Porter alcanza 0,223031 y 0,105802, respectivamente. Estas
comparaciones no aíslan longitud ni dificultad. TF-IDF con Porter baja en el
grupo corto (0,072929 → 0,056409), aunque mejore globalmente. No hubo etiquetado
temático manual ni pruebas de significancia; los hallazgos son descriptivos.

Los ejemplos se eligieron como máximo y mínimo de AP@1000 denso−BM25/Porter,
con desempate lexicográfico por `qid`, entre las 648 consultas. En **2076**,
*Can vet / veterinary bills be considered deductions (tax-deductible) for Income
Tax purposes [Canada]?*, hay dos relevantes: `278824` (los animales de servicio
pueden deducirse, las mascotas no) y `184646` (gastos médicos ante la Canada
Revenue Agency). El denso los coloca primero y segundo (P@10=0,2, R@10=1, AP=1);
BM25 con Porter pone primero `541809`, sobre gastos deducibles de una empresa,
deja `184646` en 796 y no recupera `278824` en top-1000 (0/0/0,000628). Ambos
relevantes hablan de mascotas y perros sin repetir "vet" ni "veterinary", el
vocabulario de la consulta. Antes de la corrección, el máximo era **1074**
(*How common is “pass-through” health insurance?*): BM25 no recuperaba su
relevante porque el preprocesamiento descartaba `pass-through`; ahora lo
recupera, aunque en la posición 756 (AP=0,001323). Es compatible con esa
hipótesis, porque el preprocesamiento fue el único cambio, pero no aísla
`pass-through` de los demás trozos nuevos. En **1310**, *Is is possible to take a mortgage
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
hashes que la evaluación regenerada de #11. Los hashes NLTK describen recursos de evaluación, no certifican
por sí solos la construcción original. El pipeline parte guiones y decimales en
trozos (`1.000` → `1`, `000`) y descarta `n't`; MiniLM limita su entrada a 256 tokens. No se atribuyen
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
