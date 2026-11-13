"""Recuperación BM25 propia sobre el índice invertido compartido."""

from __future__ import annotations

from collections import Counter
from math import log1p

from ris.index import InvertedIndex
from ris.preprocessing import preprocess
from ris.retrievers.base import Result


class BM25Retriever:
    """Aplica BM25 con k1=1.5 y b=0.75, según docs/PLAN.md.

    IDF = ln(1 + (N-df+0.5)/(df+0.5)). Cada término único de consulta
    aporta IDF * tf*(k1+1)/(tf+k1*(1-b+b*dl/avgdl)). N y avgdl incluyen
    los documentos vacíos. Se devuelven solo documentos coincidentes;
    una consulta vacía o avgdl=0 devuelve []. El índice debe permanecer
    sin cambios durante la vida del recuperador: IDF y la corrección
    de longitud se precalculan.
    """

    def __init__(self, index: InvertedIndex, stem: bool = False) -> None:
        if index.stem != stem:
            raise ValueError("El índice tiene una configuración de stemming distinta")

        self.index = index
        self.stem = stem
        self.k1 = 1.5
        self.b = 0.75
        self.avgdl = sum(index.doc_lengths.values()) / index.N if index.N else 0.0
        self.idf = {
            term: log1p((index.N - df + 0.5) / (df + 0.5))
            for term, df in index.df.items()
        }
        self.length_normalization = (
            {
                doc_id: self.k1 * (1 - self.b + self.b * dl / self.avgdl)
                for doc_id, dl in index.doc_lengths.items()
            }
            if self.avgdl > 0
            else {}
        )

    def search(self, query: str, k: int = 10) -> list[Result]:
        if k <= 0:
            raise ValueError("k debe ser un entero positivo")

        if self.avgdl == 0:
            return []

        scores: Counter[str] = Counter()
        # Orden estable: la permutación de la consulta no cambia las sumas.
        for term in sorted(set(preprocess(query, stem=self.stem))):
            if term not in self.index.postings:
                continue

            idf = self.idf[term]
            for doc_id, tf in self.index.postings[term].items():
                scores[doc_id] += (
                    idf * tf * (self.k1 + 1)
                    / (tf + self.length_normalization[doc_id])
                )

        results = [Result(doc_id=doc_id, score=score) for doc_id, score in scores.items()]
        results.sort(key=lambda result: (-result.score, result.doc_id))
        return results[:k]