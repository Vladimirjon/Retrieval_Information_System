"""Recuperación por coseno sobre vectores TF-IDF dispersos propios."""

from __future__ import annotations

from collections import Counter
from math import log, sqrt

from ris.index import InvertedIndex
from ris.preprocessing import preprocess
from ris.retrievers.base import Result


class TfidfRetriever:
    """Aplica los pesos acordados en docs/PLAN.md para la issue #7.

    Documentos y consultas usan tf=1+ln(f), idf=ln(N/df) y norma L2,
    sin suavizado. El espacio vectorial es el vocabulario del corpus:
    términos desconocidos se ignoran y términos con df=N tienen peso cero.
    Se recuperan únicamente documentos con coseno positivo; una consulta
    de norma cero devuelve []. El índice debe permanecer sin cambios
    durante la vida del recuperador, pues IDF y normas se precalculan.
    """

    def __init__(self, index: InvertedIndex, stem: bool = False) -> None:
        if index.stem != stem:
            raise ValueError("El índice tiene una configuración de stemming distinta")

        self.index = index
        self.stem = stem
        self.idf = {term: log(index.N / df) for term, df in index.df.items()}

        squared_norms: Counter[str] = Counter()
        for term, postings in index.postings.items():
            idf = self.idf[term]
            if idf == 0:
                continue
            for doc_id, tf in postings.items():
                weight = (1 + log(tf)) * idf
                squared_norms[doc_id] += weight * weight
        self.doc_norms = {
            doc_id: sqrt(squared_norm) for doc_id, squared_norm in squared_norms.items()
        }

    def search(self, query: str, k: int = 10) -> list[Result]:
        if k <= 0:
            raise ValueError("k debe ser un entero positivo")

        frequencies = Counter(preprocess(query, stem=self.stem))
        query_weights = {
            term: (1 + log(tf)) * self.idf[term]
            for term, tf in frequencies.items()
            if self.idf.get(term, 0) > 0
        }
        query_norm = sqrt(sum(weight * weight for weight in query_weights.values()))
        if query_norm == 0:
            return []

        dot_products: Counter[str] = Counter()
        for term, query_weight in query_weights.items():
            idf = self.idf[term]
            for doc_id, tf in self.index.postings[term].items():
                dot_products[doc_id] += query_weight * (1 + log(tf)) * idf

        results = [
            Result(doc_id=doc_id, score=dot / (query_norm * self.doc_norms[doc_id]))
            for doc_id, dot in dot_products.items()
        ]
        results.sort(key=lambda result: (-result.score, result.doc_id))
        return results[:k]
