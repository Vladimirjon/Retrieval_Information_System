"""Grafo de una consulta: valores esperados calculados a mano."""

import numpy as np
import pytest

from ris.graph import build_query_graph


def unit(index, dimension=6):
    vector = np.zeros(dimension)
    vector[index] = 1.0
    return vector


def positions(graph):
    return np.array([node.position for node in graph.nodes])


def test_query_sits_at_the_origin_and_the_farthest_document_at_distance_one():
    # Puntos (1,0), (1,0) y (0,1): solo varían en la dirección (1,-1)/√2. d1 coincide con
    # la consulta (distancia 0) y d2 queda a √2, que tras escalar es 1 sobre el primer eje.
    graph = build_query_graph(unit(0), ["d1", "d2"], [unit(0), unit(1)])
    assert positions(graph) == pytest.approx(np.array([[0, 0, 0], [1, 0, 0]]), abs=1e-12)
    assert [node.doc_id for node in graph.nodes] == ["d1", "d2"]
    assert [node.similarity for node in graph.nodes] == pytest.approx([1.0, 0.0])
    assert graph.explained_variance == pytest.approx(1.0)
    assert graph.edges == ((0, 1, pytest.approx(0.0)),)


def test_four_points_fit_in_3d_so_distances_are_exact_up_to_scale():
    # Consulta y tres documentos ortonormales: todas las distancias valen √2 y cuatro
    # puntos caben en 3D sin pérdida; al escalar por √2 todas pasan a valer 1.
    graph = build_query_graph(unit(0), ["a", "b", "c"], [unit(1), unit(2), unit(3)])
    points = positions(graph)
    assert np.linalg.norm(points, axis=1) == pytest.approx([1, 1, 1])
    for i, j in ((0, 1), (0, 2), (1, 2)):
        assert np.linalg.norm(points[i] - points[j]) == pytest.approx(1.0)
    assert graph.explained_variance == pytest.approx(1.0)


def test_explained_variance_reports_what_three_components_keep():
    # Cinco puntos ortonormales centrados: cuatro direcciones con la misma varianza.
    # Tres componentes conservan 3/4.
    graph = build_query_graph(unit(0), list("abcd"), [unit(i) for i in range(1, 5)])
    assert graph.explained_variance == pytest.approx(0.75)
    assert np.linalg.norm(positions(graph), axis=1).max() == pytest.approx(1.0)


def test_each_document_links_to_its_most_similar_documents():
    # Cosenos: a·b=0,8  a·c=0  a·d=-1  b·c=0,6  b·d=-0,8  c·d=0.
    vectors = [[1, 0], [0.8, 0.6], [0, 1], [-1, 0]]
    graph = build_query_graph([1, 1], list("abcd"), vectors, neighbors=1)
    assert graph.edges == (
        (0, 1, pytest.approx(0.8)),  # a→b y b→a, una sola arista
        (1, 2, pytest.approx(0.6)),  # c→b
        (2, 3, pytest.approx(0.0)),  # d→c: el menos lejano de d
    )
    assert build_query_graph([1, 1], list("abcd"), vectors, neighbors=0).edges == ()


def test_similarity_ties_prefer_the_smaller_doc_id():
    # z es igual de parecido a "b" y a "a" (coseno 0): gana "a", que está en la posición 2.
    graph = build_query_graph(unit(3), ["z", "b", "a"], [unit(0), unit(1), unit(1)], neighbors=1)
    assert [edge[:2] for edge in graph.edges] == [(0, 2), (1, 2)]


def test_cosines_do_not_depend_on_the_length_of_the_vectors():
    graph = build_query_graph([2, 0], ["a", "b"], [[0, 5], [3, 0]])
    assert [node.similarity for node in graph.nodes] == pytest.approx([0.0, 1.0])


def test_identical_vectors_collapse_to_the_origin_without_dividing_by_zero():
    graph = build_query_graph(unit(0), ["a", "b"], [unit(0), unit(0)])
    assert positions(graph) == pytest.approx(np.zeros((2, 3)))
    assert graph.explained_variance == 0.0
    assert np.isfinite(positions(graph)).all()


def test_same_input_gives_the_same_graph():
    rng = np.random.default_rng(7)
    query, docs = rng.normal(size=16), rng.normal(size=(12, 16))
    ids = [f"d{i:02d}" for i in range(12)]
    first = build_query_graph(query, ids, docs)
    assert build_query_graph(query.copy(), list(ids), docs.copy()) == first
    assert len(first.nodes) == 12
    assert all(i < j for i, j, _ in first.edges) and len(set(first.edges)) == len(first.edges)
    assert 0.0 < first.explained_variance < 1.0


@pytest.mark.parametrize(
    ("doc_ids", "doc_vectors", "query"),
    [
        ([], np.empty((0, 2)), [1, 0]),                 # sin documentos
        (["a", "a"], [[1, 0], [0, 1]], [1, 0]),         # ids repetidos
        (["a", "b"], [[1, 0]], [1, 0]),                 # un vector de menos
        (["a"], [[1, 0, 0]], [1, 0]),                   # otra dimensión
        (["a"], [[0, 0]], [1, 0]),                      # documento nulo
        (["a"], [[1, 0]], [0, 0]),                      # consulta nula
        (["a"], [[float("nan"), 0]], [1, 0]),           # valor no finito
    ],
)
def test_invalid_input_is_rejected(doc_ids, doc_vectors, query):
    with pytest.raises(ValueError):
        build_query_graph(query, doc_ids, doc_vectors)


@pytest.mark.parametrize("neighbors", [-1, 1.5, True, None])
def test_invalid_neighbors_is_rejected(neighbors):
    with pytest.raises(ValueError):
        build_query_graph([1, 0], ["a"], [[1, 0]], neighbors=neighbors)


def test_linked_only_joins_the_first_documents_and_keeps_every_position():
    # a·b = 0,8 y b·c = 0,6; d (alrededor) es el más parecido a c, pero con linked=3
    # no recibe aristas: solo se unen los tres primeros, entre sí.
    vectors = [[1, 0, 0], [0.8, 0.6, 0], [0, 1, 0], [0, 0.9, 0.1]]
    full = build_query_graph([1, 1, 0], list("abcd"), vectors, neighbors=1)
    linked = build_query_graph([1, 1, 0], list("abcd"), vectors, neighbors=1, linked=3)
    assert [edge[:2] for edge in linked.edges] == [(0, 1), (1, 2)]
    assert all(j < 3 for _, j, _ in linked.edges)
    assert (2, 3) in [edge[:2] for edge in full.edges]
    assert positions(linked) == pytest.approx(positions(full))
    assert build_query_graph([1, 1, 0], list("abcd"), vectors, linked=0).edges == ()
    assert build_query_graph([1, 1, 0], list("abcd"), vectors, linked=99) == \
        build_query_graph([1, 1, 0], list("abcd"), vectors)


@pytest.mark.parametrize("linked", [-1, 1.5, True])
def test_invalid_linked_is_rejected(linked):
    with pytest.raises(ValueError):
        build_query_graph([1, 0], ["a"], [[1, 0]], linked=linked)
