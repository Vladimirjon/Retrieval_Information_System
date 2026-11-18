"""Grafo de una consulta: la consulta y sus resultados situados en 3D.

Se calcula en el momento de buscar y solo con los vectores de la consulta y de los
documentos devueltos (no con el corpus), así que cuesta milisegundos. Sirve para
mirar los resultados, no para ordenarlos: el ranking es el del modelo elegido.

- Posición: PCA a tres componentes de esos vectores. La consulta queda en el origen
  y el punto más lejano a distancia 1. Es una proyección: `explained_variance` dice
  qué fracción de la variación conserva; las distancias dibujadas son aproximadas.
- Parecido: coseno entre los vectores originales (no entre las posiciones 3D).
- Aristas: cada documento se une a sus `neighbors` documentos más parecidos. Con
  `linked`, solo los `linked` primeros (los resultados visibles) se unen, entre sí;
  los demás se sitúan en el mapa como contexto, sin aristas.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

_COMPONENTS = 3
_EPS = 1e-9


@dataclass(frozen=True)
class GraphNode:
    """Un documento del grafo. `similarity` es su coseno con la consulta."""

    doc_id: str
    position: tuple[float, float, float]
    similarity: float


@dataclass(frozen=True)
class QueryGraph:
    """Nodos en el orden recibido y aristas (i, j, coseno) con i < j, sin repetir."""

    nodes: tuple[GraphNode, ...]
    edges: tuple[tuple[int, int, float], ...]
    explained_variance: float


def _unit_rows(vectors: object, name: str) -> np.ndarray:
    values = np.asarray(vectors, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] == 0 or not np.isfinite(values).all():
        raise ValueError(f"{name} debe ser una matriz finita de vectores")
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if (norms <= 0).any():
        raise ValueError(f"{name} contiene un vector nulo")
    return values / norms


def _project(points: np.ndarray) -> tuple[np.ndarray, float]:
    """PCA a 3D con signo determinista; la primera fila (la consulta) queda en el origen."""
    centered = points - points.mean(axis=0)
    left, singular, _ = np.linalg.svd(centered, full_matrices=False)
    variance = singular**2
    total = float(variance.sum())
    coords = np.zeros((len(points), _COMPONENTS))
    if total <= _EPS:  # todos los vectores coinciden: no hay nada que separar
        return coords, 0.0
    kept = min(_COMPONENTS, len(singular))
    coords[:, :kept] = left[:, :kept] * singular[:kept]
    coords[:, :kept][:, singular[:kept] <= _EPS * singular[0]] = 0.0
    for axis in range(kept):  # el signo de la SVD es arbitrario: se fija el del mayor valor
        column = coords[:, axis]
        if column[np.argmax(np.abs(column))] < 0:
            column *= -1
    coords -= coords[0]
    radius = float(np.linalg.norm(coords, axis=1).max())
    if radius > _EPS:
        coords /= radius
    return coords, float(variance[:kept].sum() / total)


def build_query_graph(
    query_vector: object,
    doc_ids: Sequence[str],
    doc_vectors: object,
    *,
    neighbors: int = 2,
    linked: int | None = None,
) -> QueryGraph:
    """Construye el grafo de una consulta y sus documentos (en el orden del ranking)."""
    if isinstance(neighbors, bool) or not isinstance(neighbors, int) or neighbors < 0:
        raise ValueError("neighbors debe ser un entero no negativo")
    if linked is not None and (isinstance(linked, bool) or not isinstance(linked, int)
                               or linked < 0):
        raise ValueError("linked debe ser un entero no negativo o None")
    doc_ids = list(doc_ids)
    if not doc_ids:
        raise ValueError("El grafo necesita al menos un documento")
    if len(set(doc_ids)) != len(doc_ids):
        raise ValueError("doc_ids no debe tener duplicados")
    docs = _unit_rows(doc_vectors, "doc_vectors")
    query = _unit_rows(np.asarray(query_vector, dtype=np.float64).reshape(1, -1), "query_vector")
    if docs.shape != (len(doc_ids), query.shape[1]):
        raise ValueError("doc_vectors no corresponde a doc_ids o a la dimensión de la consulta")

    coords, explained = _project(np.vstack([query, docs]))
    to_query = docs @ query[0]
    between = docs @ docs.T

    pairs: set[tuple[int, int]] = set()
    count = len(doc_ids) if linked is None else min(linked, len(doc_ids))
    for i in range(count):
        others = sorted(
            (j for j in range(count) if j != i),
            key=lambda j: (-between[i, j], doc_ids[j]),
        )
        pairs.update((min(i, j), max(i, j)) for j in others[:neighbors])

    return QueryGraph(
        nodes=tuple(
            GraphNode(doc_id, tuple(float(value) for value in coords[row + 1]),
                      float(to_query[row]))
            for row, doc_id in enumerate(doc_ids)
        ),
        edges=tuple((i, j, float(between[i, j])) for i, j in sorted(pairs)),
        explained_variance=explained,
    )
