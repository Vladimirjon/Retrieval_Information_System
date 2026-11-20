# Informe técnico (#17)

Versión revisada el 19 de noviembre de 2026 para la issue #17, basada en
`main` en `66aea9d2849cf4d4e0ed58c3bfff1ce5b23bfaa0`. Queda pendiente la
revisión de los integrantes en el pull request.

El [enunciado](../ESPECIFICACION.md) y la [issue #17](https://github.com/Vladimirjon/fiqa-information-retrieval/issues/17)
piden cuatro secciones y hasta cinco páginas totales. En los materiales del
proyecto no se encontró una plantilla obligatoria. La primera página incluye
la Escuela Politécnica Nacional, el título, los integrantes, la asignatura y
el docente, y comienza el informe en esa misma página.

## Archivos

| Archivo | Función |
| --- | --- |
| [informe_tecnico.md](informe_tecnico.md) | Fuente editable completa, en español. |
| [informe_tecnico.pdf](informe_tecnico.pdf) | PDF de cinco páginas A4, incluidas referencias. |
| `generar_pdf.py` | Convierte el subconjunto de Markdown usado aquí, con tablas y saltos explícitos. |
| `requirements-pdf.txt` | Herramientas del documento, separadas de la app y del entorno de evaluación. |
| `verificar_fuentes.py` | Auditoría independiente de reportes, tablas del informe y texto del PDF. |
| [verificacion_fuentes.json](verificacion_fuentes.json) | Hashes completos, base, cifras, comprobaciones y discrepancias. |

Los aportes `04_descripcion_corpus.md` y `12_comparacion_modelos.md` sirven de
referencia. Las páginas presentan el corpus, el diseño y despliegue, los casos,
las métricas globales y el análisis por grupos con las fuentes. El diseño usa
tipografía Times, cuerpo de 11,5 puntos, tablas con líneas horizontales y
márgenes laterales de unos 22 mm.

La bibliografía incluye siete fuentes externas y los enlaces al corpus y los
juicios de relevancia facilitados por el profesor. La auditoría adjunta deja
constancia de los reportes utilizados.

## Reproducir desde la raíz

Requiere Git con el commit base disponible y Python 3.12. El entorno de edición
se puede mantener fuera del entorno de la app y dentro de una carpeta ignorada:

```powershell
python -m venv artifacts/informe-env
.\artifacts\informe-env\Scripts\python.exe -m pip install -r docs/informe/requirements-pdf.txt
.\artifacts\informe-env\Scripts\python.exe docs/informe/generar_pdf.py
.\artifacts\informe-env\Scripts\python.exe docs/informe/verificar_fuentes.py --pdf
```

En Linux, usar `artifacts/informe-env/bin/python` en lugar de `Scripts\python.exe`.
El generador resuelve su fuente desde su propia carpeta, no descarga datos ni
ejecuta búsquedas. Fija los metadatos temporales al 19-11-2026, 05:00 UTC para no
introducir variaciones por la hora de generación. Rechaza glifos ausentes y una
salida distinta de cinco páginas. Si no cabe el texto, revisar la redacción y
los saltos antes de reducir tipografía.

Para comprobar una segunda generación sin sobrescribir el PDF final:

```powershell
.\artifacts\informe-env\Scripts\python.exe docs/informe/generar_pdf.py --output artifacts/informe-segunda.pdf
Get-FileHash docs/informe/informe_tecnico.pdf, artifacts/informe-segunda.pdf -Algorithm SHA256
```

Ambas salidas fueron idénticas con Python 3.12.16 y las versiones fijadas de
`requirements-pdf.txt`. El SHA-256 registrado es
`991a1b45fc7ff001405fac09c4fa54f2fecd205a7e04005d0ec4d206230a4910`.

La extracción de texto no sustituye la revisión visual. Con Poppler disponible,
renderizar **todas** las páginas en una carpeta ignorada y abrirlas:

```powershell
New-Item -ItemType Directory -Force artifacts/informe-qa | Out-Null
pdfinfo docs/informe/informe_tecnico.pdf
pdftoppm -r 130 -png docs/informe/informe_tecnico.pdf artifacts/informe-qa/pagina
```

Se renderizaron y revisaron las cinco páginas A4 con Poppler 26.07.0. Se
comprobaron la legibilidad de tablas y referencias, las tildes y los márgenes.
`verificar_fuentes.py --pdf` comprueba que **todo el texto de cada
página** coincide con su bloque de Markdown. También revisa los folios, los
integrantes en los metadatos y que los enlaces HTTPS coincidan con los del
Markdown. Repetir generación, auditoría y revisión visual al editar.

## Evidencia y alcance de la validación

- Reportes posteriores al PR #35, fijados en la base indicada: integridad exacta
  CSV/JSON, 4 536 filas, siete configuraciones con los mismos 648 qids, medias
  globales, 49 medias de grupo, 27 conjuntos de métricas pareadas y extremos de
  AP. Tablas globales, de longitud y de casos contrastadas con la fuente final.
- Los cuatro rankings de casos se auditan desde las posiciones relevantes con
  fracciones exactas y contra el CSV: mejora `2076`, pérdida `1310`.
- En la revisión del 17 de noviembre, con la caché de FiQA en modo offline,
  `scripts.evaluate.verify_reports`, `app.metrics_data.load_reports` y
  `ris.comparison.validate_reports` pasan. Corpus, consultas y qrels coinciden
  con los conteos; las siete pertenencias de grupos se recomputan desde las
  consultas; los 43 fragmentos/juicios guardados coinciden con los originales.
  También se comprobó el documento relevante vacío `117276` de `qid 5206`.
- `python -m ruff check .` sin errores; `pip check` del entorno de edición sin
  conflictos. No se reevalúa test ni se reconstruyen índices o embeddings.
- El estado de CI de esta versión se consulta en el pull request de la #17.

## Discrepancias resueltas y pendientes

El aporte de #12 conserva una advertencia antigua sobre embeddings/FAISS con
hashes distintos. En el resumen estructurado **actual**, los hashes locales de
los índices y del denso coinciden con #11 y con `deploy/artifacts.json`. Se
contrastaron esos campos entre los tres archivos, sin certificar de nuevo los
artefactos binarios completos. Se sigue la evidencia estructurada y se deja
explicada la divergencia; no se sobrescribe el aporte parcial.

La copia Windows de los JSON usa CRLF. La huella de `metrics_summary.json`
registrada por #12 corresponde al blob LF de Git: coincide con el snapshot y
su contenido es idéntico al JSON local. El CSV sí conserva LF y satisface su
SHA-256 exacto. La auditoría registra ambos hashes; no normaliza los originales.

El informe toma la evidencia del despliegue de `STATE.md`, el manifiesto y
los PR #38/#39. Los 977 MiB corresponden a la RSS tras la carga inicial.
Siguen pendientes una medición del pico de RAM bajo uso, el arranque tras
reposo y la vista Comparar de la issue #14. Esta revisión verifica el
documento y sus fuentes locales.
