"""Modelos de recuperación del sistema."""

from ris.retrievers.base import Result, Retriever
from ris.retrievers.jaccard import JaccardRetriever
from ris.retrievers.tfidf import TfidfRetriever

__all__ = ["JaccardRetriever", "Result", "Retriever", "TfidfRetriever"]
