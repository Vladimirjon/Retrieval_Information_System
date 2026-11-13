"""Modelos de recuperación del sistema."""

from ris.retrievers.base import Result, Retriever
from ris.retrievers.jaccard import JaccardRetriever

__all__ = ["JaccardRetriever", "Result", "Retriever"]
