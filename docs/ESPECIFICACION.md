*Transcripción fiel de docs/proy01.pdf. Úsese este archivo en lugar del PDF.*

## Proyecto de 1er Bimestre: Sistema de Recuperación de Información
**Prof. Iván Carrera -- Recuperación de Información**  
**Fecha de entrega: 23 de noviembre de 2026**

## Objetivo
Diseñar e implementar un sistema de recuperación de información que permita comparar modelos clásicos y modernos de recuperación. El sistema deberá indexar un corpus textual utilizando un índice invertido para modelos basados en términos, y una base de datos vectorial para recuperación semántica basada en embeddings. El sistema deberá permitir ejecutar consultas de texto libre, mostrar rankings de documentos y evaluar la calidad de los resultados mediante métricas de Recuperación de Información.

## Alcance y funcionalidades mínimas
El sistema debe cumplir con los siguientes requerimientos:

*   **a. Construcción del índice**
    *   Leer un corpus de documentos en texto plano.
    *   Procesamiento básico: tokenización, normalización y remoción de stopwords.
    *   Construcción de un índice invertido que almacene, para cada término, los documentos en los que aparece y su frecuencia.

*   **b. Modelo de recuperación**
    *   Implementar recuperación basada en similitud Jaccard utilizando vectores binarios.
    *   Implementar recuperación basada en similitud de coseno utilizando TF-IDF.
    *   Implementar recuperación con BM25.
    *   Permitir la ejecución de consultas de texto libre.
    *   Mostrar un ranking de documentos ordenados por relevancia.

*   **c. Recuperación semántica con embeddings**
    *   Generar embeddings para los documentos del corpus utilizando un modelo preentrenado.
    *   Generar embeddings para las consultas de texto libre.
    *   Almacenar los embeddings en una base de datos vectorial, como ChromaDB o FAISS.
    *   Recuperar los documentos más similares usando búsqueda vectorial.
    *   Mostrar un ranking de resultados basado en similitud vectorial.

*   **d. Interfaz básica**
    *   Interfaz web de recuperación de consultas, desplegada en un servicio gratuito (Render, Vercel, etc.).
    *   Funcionalidades mínimas:
        *   Realizar consultas.
        *   Elegir el modelo de recuperación (Binario, Vectorial con TF-IDF, Probabilístico, Vectorial con Embeddings).
        *   Visualizar el top-n de resultados en los diferentes modelos.
        *   Comparar resultados entre los diferentes modelos.
        *   Visualizar las métricas de evaluación de la recuperación.

*   **e. Evaluación de resultados**
    *   Usar un conjunto de consultas de prueba y documentos relevantes (qrels).
    *   Calcular para cada consulta:
        *   Precisión
        *   Recall
    *   Calcular para todo el sistema:
        *   MAP

*   **f. Comparación de modelos**
    *   Comparar los resultados obtenidos con los diferentes modelos de recuperación: Binario, Vectorial con TF-IDF, Probabilístico, Vectorial con Embeddings.
    *   Analizar en qué tipos de consultas funciona mejor cada modelo.
    *   Identificar casos donde la recuperación semántica mejora o empeora respecto a BM25 o TF-IDF.

## Requisitos técnicos
*   Lenguaje sugerido: Python.
*   Se pueden usar bibliotecas como NumPy, pandas, NLTK, spaCy, scikit-learn, sentence-transformers.
*   Para recuperación vectorial se debe usar ChromaDB o FAISS.
*   No se permite el uso de motores completos de búsqueda como Elasticsearch, Solr o Whoosh.
*   La implementación de los modelos clásicos debe ser propia o claramente explicada.

## Entregables
1. URL del despliegue del sistema.
2. Código fuente con instrucciones claras de ejecución (README).
3. Informe técnico (máximo 5 páginas), que incluya:
    *   Descripción del corpus utilizado.
    *   Explicación de las decisiones de diseño.
    *   Ejemplos de consultas y resultados.
    *   Análisis de métricas de evaluación.

## Materiales entregados por el profesor
*   https://huggingface.co/datasets/BeIR/fiqa
*   https://huggingface.co/datasets/BeIR/fiqa-qrels
