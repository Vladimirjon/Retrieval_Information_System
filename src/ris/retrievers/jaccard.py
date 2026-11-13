"""Recuperación con coeficiente de Jaccard sobre vectores binarios."""

from __future__ import annotations

from collections import Counter

from ris.index import InvertedIndex
from ris.preprocessing import preprocess
from ris.retrievers.base import Result


class JaccardRetriever:
    """Ordena documentos por Jaccard entre los conjuntos de términos."""

    def __init__(self, index: InvertedIndex, stem: bool = False) -> None:
        if index.stem != stem:
            raise ValueError("El índice tiene una configuración de stemming distinta")

        self.index = index
        self.stem = stem
        # doc_lengths cuenta repeticiones; Jaccard necesita términos únicos,
        # que equivalen al número de posting lists donde aparece cada documento
        self.unique_terms: Counter[str] = Counter()
        for postings in index.postings.values():
            self.unique_terms.update(postings.keys())

    def search(self, query: str, k: int = 10) -> list[Result]:
        if k <= 0:
            raise ValueError("k debe ser un entero positivo")

        query_terms = set(preprocess(query, stem=self.stem))
        if not query_terms:
            return []

        intersections: Counter[str] = Counter()
        for term in query_terms:
            intersections.update(self.index.postings.get(term, {}).keys())

        results = []
        for doc_id, intersection in intersections.items():
            # |Q ∪ D| = |Q| + |D| - |Q ∩ D|; los términos desconocidos cuentan en |Q|
            union = len(query_terms) + self.unique_terms[doc_id] - intersection
            results.append(Result(doc_id=doc_id, score=intersection / union))

        # Score descendente y, ante empate, doc_id ascendente
        results.sort(key=lambda result: (-result.score, result.doc_id))
        return results[:k]
