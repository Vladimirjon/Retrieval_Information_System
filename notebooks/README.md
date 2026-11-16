# Análisis exploratorio de FiQA (#4)

[04_eda_fiqa.ipynb](04_eda_fiqa.ipynb) caracteriza el corpus completo y las
consultas test usando los cargadores de #2 y el preprocesamiento aprobado de #3.
Incluye tamaños, integridad, campos vacíos, longitudes, vocabulario, CF frente a
DF, efecto de stopwords y Porter, cobertura léxica de consultas y distribución de
relevantes. No evalúa recuperadores ni modifica sus contratos.

Con Python 3.12 y el entorno de `requirements.txt`, desde la raíz del repositorio:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install --no-deps -e .
python -m ipykernel install --prefix .venv --name ris-eda --display-name "Python 3.12 (RIS EDA)"
python -m jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=1800 --ExecutePreprocessor.kernel_name=ris-eda notebooks/04_eda_fiqa.ipynb
python -m pytest
python -m ruff check .
```

Para ejecución interactiva: `python -m jupyter lab`, abrir el notebook, seleccionar
el kernel `Python 3.12 (RIS EDA)` y ejecutar todas las celdas desde un kernel nuevo.
La primera ejecución descarga FiQA en sus revisiones fijadas y los recursos NLTK
en `data/`, fuera de Git. No requiere credenciales ni un servicio de pago. Las
siguientes ejecuciones reutilizan esos recursos; no se modifican el corpus ni el
preprocesador. En Windows pueden aparecer avisos sobre caché sin symlinks o el
bucle de eventos de Jupyter, sin impedir la ejecución.

El notebook genera programáticamente:

- [reports/eda_summary.json](../reports/eda_summary.json): esquema v1, tamaños,
  campos vacíos, validaciones, estadísticas de población, distribuciones completas
  de longitudes y CF, vocabularios y términos frecuentes con CF/DF, reducciones,
  consultas OOV y relevancia test. Registra revisiones, SHA-256 de los cinco archivos
  del dataset, módulos de carga/preprocesamiento, recursos NLTK y versiones.
- [docs/informe/04_descripcion_corpus.md](../docs/informe/04_descripcion_corpus.md):
  texto breve reutilizable en #17, calculado desde los mismos agregados.

`original` cuenta tokens de NLTK con caso y puntuación. `normalized` es una base
diagnóstica en minúsculas con tokens alfanuméricos que permite aislar el filtrado de
stopwords. `lexical` y `porter` llaman a `preprocess` con su configuración acordada.
Las longitudes cuentan apariciones; los vocabularios cuentan formas distintas.
CF suma apariciones en la colección; DF cuenta documentos que contienen la forma.
En los agregados de consultas, el campo `df` cuenta consultas distintas, se usa
solo para describir esa colección y no es la DF documental utilizada para IDF.
Las estadísticas incluyen vacíos, usan desviación poblacional y percentiles lineales.
Los techos ideales de métricas son límites derivados de los qrels, sin resultados
de modelos. La salida completa es determinista salvo el tiempo orientativo de ejecución.

La lógica auxiliar está en [src/ris/eda.py](../src/ris/eda.py) y sus pruebas manuales
pequeñas en [tests/test_eda.py](../tests/test_eda.py), sin descargar FiQA ni NLTK.
El notebook añade comprobaciones con los datos reales, incluyendo cobertura global,
consistencia de distribuciones, preservación de longitudes por Porter y una
recomprobación independiente de CF/DF sobre 100 documentos. Los datos completos y
los PDF del profesor no forman parte de los entregables versionados.

# Comparación de modelos (#12)

[12_analisis_modelos.ipynb](12_analisis_modelos.ipynb) consume los resultados de
#11: siete configuraciones sobre las mismas 648 consultas test. Incluye tablas
de P@10/R@10/MAP@1000, diferencias pareadas, efecto de Porter, grupos observables
y dos casos extremos de AP@1000 denso−BM25 con Porter (desempate por `qid` como
cadena). No se ajustan modelos con test. Las explicaciones están en español;
los fragmentos conservan el inglés original del corpus.

Desde la raíz, con Python 3.12 y las versiones exactas de `requirements.txt`:

```powershell
python -m pip install -r requirements.txt
python -m pip install --no-deps -e .
$env:HF_HOME = Join-Path $PWD 'data/hf_home'
$env:NLTK_DATA = Join-Path $PWD 'data/nltk_data'
$env:MPLCONFIGDIR = Join-Path $PWD 'data/matplotlib'
$env:JUPYTER_CONFIG_DIR = Join-Path $PWD 'data/jupyter/config'
$env:JUPYTER_RUNTIME_DIR = Join-Path $PWD 'data/jupyter/runtime'
$env:IPYTHONDIR = Join-Path $PWD 'data/ipython'
$env:PYTHONUTF8 = '1'
python -m nltk.downloader -d data/nltk_data punkt_tab stopwords
```

Reutilizar artefactos compatibles si ya existen. Si faltan, construir **solo los
ausentes**, sin `--overwrite` para el denso:

```powershell
python scripts/build_artifacts.py
python scripts/build_artifacts.py --stem
python scripts/build_artifacts.py --dense --threads 8 --batch-size 32
```

Se usan las revisiones fijadas en `ris.data` y `ris.retrievers.dense`; descargas,
índices y embeddings quedan en `data/` y `artifacts/`, excluidos de Git. No hay
ubicación oficial de descarga de artefactos documentada al preparar #12 (#16
sigue pendiente). La construcción completa del denso puede tardar varios minutos.
El notebook carga el índice existente y codifica únicamente las dos consultas.

Ejecutar desde cero:

```powershell
python -m ipykernel install --prefix .venv --name ris-analysis --display-name "Python 3.12 (RIS análisis)"
python -m jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=1800 --ExecutePreprocessor.kernel_name=ris-analysis notebooks/12_analisis_modelos.ipynb
```

El kernel resuelve la raíz desde la raíz del repositorio o desde `notebooks/`.
Después de descargar todo, puede usarse `$env:HF_HUB_OFFLINE = '1'`.
El notebook escribe [reports/comparison_summary.json](../reports/comparison_summary.json)
y cuatro figuras PNG de inspección local en `artifacts/analysis/figures/`; las
figuras también quedan embebidas en sus salidas. El resumen incluye todos los
agregados, pertenencias por `qid`, deltas, casos con posiciones/scores/fragmentos,
hashes, versiones y configuración. Los cuatro rankings de casos completos se
guardan solo localmente en `artifacts/analysis/`.

La misma lógica puede ejecutarse sin interfaz Jupyter, sin sustituir las métricas
de #11. Esta segunda ejecución permite comparar de forma determinista:

```powershell
python -m scripts.analyze_models --output artifacts/analysis/comparison_second.json
Get-FileHash reports/comparison_summary.json, artifacts/analysis/comparison_second.json -Algorithm SHA256
python -m pytest
python -m ruff check .
git diff --check
```

La igualdad byte a byte se comprueba dentro del mismo entorno y con los mismos
archivos; no se promete entre máquinas. El resumen registra el commit del código
de recuperación/evaluación y hashes de las fuentes de análisis (evita una
referencia circular al commit que contiene el propio resumen).
[docs/informe/12_comparacion_modelos.md](../docs/informe/12_comparacion_modelos.md)
ofrece texto breve reutilizable en #17; no es el informe final.

Si falla el hash del CSV, investigar antes de analizar. En Windows, una copia
anterior puede tener CRLF por `core.autocrlf=true`; `.gitattributes` fija LF para
el CSV de métricas. Con un árbol limpio y sin ediciones propias en ese archivo,
`git restore --source=HEAD --worktree -- reports/metrics_per_query.csv` recupera
los bytes de Git bajo esa regla. No regenerar ni normalizar reportes distintos
para encubrir inconsistencias. Las medias CSV/JSON y las métricas de los casos
se contrastan con `atol=1e-12`, sin tolerancia relativa.
