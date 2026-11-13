"""Modelos de recuperación del sistema."""

from ris.retrievers.base import Result, Retriever
from ris.retrievers.bm25 import BM25Retriever
from ris.retrievers.dense import DenseRetriever
from ris.retrievers.jaccard import JaccardRetriever
from ris.retrievers.tfidf import TfidfRetriever

__all__ = [
    "BM25Retriever", "DenseRetriever", "JaccardRetriever", "Result", "Retriever",
    "TfidfRetriever",
]
