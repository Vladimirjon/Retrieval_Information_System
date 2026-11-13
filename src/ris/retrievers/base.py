"""Tipos compartidos por todos los recuperadores."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class Result:
    """Documento recuperado con su puntuación."""

    doc_id: str
    score: float


class Retriever(Protocol):
    """Interfaz común de los modelos de recuperación."""

    def search(self, query: str, k: int = 10) -> list[Result]: ...
