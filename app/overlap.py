"""Solapamiento entre rankings (issue #14). Funciones puras, sin Streamlit.

El solapamiento de un par de rankings es la intersección de sus `doc_id`.
Denominador: el menor de los dos tamaños, o sea, el máximo de documentos que
podrían coincidir. Con rankings completos vale k; si un modelo devuelve menos
de k, no se castiga por documentos que no pudo dar. Si alguno está vacío el
denominador es 0 y no hay porcentaje.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import combinations


@dataclass(frozen=True)
class PairOverlap:
    """Documentos compartidos por dos modelos y su denominador."""

    a: str
    b: str
    shared: int
    total: int

    @property
    def percent(self) -> float | None:
        """Porcentaje 0–100, o None si no hay nada que comparar (total = 0)."""
        return self.shared / self.total * 100 if self.total else None


def pair_overlap(a: str, ids_a: Sequence[str], b: str, ids_b: Sequence[str]) -> PairOverlap:
    set_a, set_b = set(ids_a), set(ids_b)
    return PairOverlap(a, b, len(set_a & set_b), min(len(set_a), len(set_b)))


def overlaps(rankings: Mapping[str, Sequence[str]]) -> list[PairOverlap]:
    """Todos los pares de modelos, en el orden en que llegan los rankings."""
    return [
        pair_overlap(a, rankings[a], b, rankings[b])
        for a, b in combinations(rankings, 2)
    ]


def mean_percent(pairs: Sequence[PairOverlap]) -> float | None:
    """Promedio de los porcentajes de los pares que tienen algo que comparar."""
    values = [p.percent for p in pairs if p.percent is not None]
    return sum(values) / len(values) if values else None


def appearances(rankings: Mapping[str, Sequence[str]]) -> dict[str, tuple[str, ...]]:
    """Para cada doc_id, los modelos que lo recuperaron (en el orden de `rankings`)."""
    found: dict[str, list[str]] = {}
    for model, ids in rankings.items():
        for doc_id in dict.fromkeys(ids):
            found.setdefault(doc_id, []).append(model)
    return {doc_id: tuple(models) for doc_id, models in found.items()}


def exclusive_counts(rankings: Mapping[str, Sequence[str]]) -> dict[str, int]:
    """Cuántos documentos de cada ranking no aparecen en ningún otro."""
    seen = appearances(rankings)
    return {
        model: sum(1 for doc_id in set(ids) if len(seen[doc_id]) == 1)
        for model, ids in rankings.items()
    }
