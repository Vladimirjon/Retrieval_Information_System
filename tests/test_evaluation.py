import pytest

from ris.evaluation import (
    average_precision,
    mean_average_precision,
    precision_at_k,
    recall_at_k,
)

RANKING = ["a", "b", "c"]
RELEVANT = {"a", "c"}


def test_manual_example_precision_recall_ap():
    assert precision_at_k(RANKING, RELEVANT, k=2) == pytest.approx(0.5)
    assert recall_at_k(RANKING, RELEVANT, k=2) == pytest.approx(0.5)
    assert average_precision(RANKING, RELEVANT, k=1000) == pytest.approx(
        (1 / 1 + 2 / 3) / 2
    )
    assert average_precision(RANKING, RELEVANT) == pytest.approx(5 / 6)


def test_map_with_query_without_hits():
    run = {"q1": RANKING, "q2": ["x", "y", "z"]}
    qrels = {"q1": RELEVANT, "q2": {"w"}}
    assert mean_average_precision(run, qrels) == pytest.approx(5 / 12)


def test_precision_divides_by_k_with_short_ranking():
    assert precision_at_k(["a"], {"a"}, k=10) == pytest.approx(0.1)


def test_recall_divides_by_all_relevant():
    assert recall_at_k(["a", "b"], {"a", "c", "d", "e"}, k=10) == pytest.approx(0.25)


def test_ap_cutoff_ignores_relevant_beyond_k():
    # El relevante "c" (posición 3) queda fuera del corte, pero sigue en el
    # denominador: AP@2 = (1/1) / 2.
    assert average_precision(RANKING, RELEVANT, k=2) == pytest.approx(0.5)


def test_ap_cutoff_1000_with_relevant_at_1001():
    ranking = ["rel1"] + [f"d{i}" for i in range(999)] + ["rel2"]
    relevant = {"rel1", "rel2"}
    assert len(ranking) == 1001
    assert average_precision(ranking, relevant) == pytest.approx(0.5)
    assert average_precision(ranking, relevant, k=1001) == pytest.approx(
        (1 + 2 / 1001) / 2
    )


def test_ap_divides_by_all_relevant_not_only_retrieved():
    assert average_precision(["a"], {"a", "z"}) == pytest.approx(0.5)


def test_empty_relevant_returns_zero():
    assert precision_at_k(RANKING, set(), k=2) == 0.0
    assert recall_at_k(RANKING, set(), k=2) == 0.0
    assert average_precision(RANKING, set()) == 0.0


def test_relevant_accepts_list():
    assert average_precision(RANKING, ["a", "c"]) == pytest.approx(5 / 6)


def test_empty_ranking_returns_zero():
    assert precision_at_k([], RELEVANT) == 0.0
    assert recall_at_k([], RELEVANT) == 0.0
    assert average_precision([], RELEVANT) == 0.0


def test_map_missing_query_in_run_counts_as_zero():
    run = {"q1": RANKING}
    qrels = {"q1": RELEVANT, "q2": {"w"}}
    assert mean_average_precision(run, qrels) == pytest.approx(5 / 12)


def test_map_query_without_relevant_counts_as_zero():
    run = {"q1": RANKING, "q2": ["x"]}
    qrels = {"q1": RELEVANT, "q2": set()}
    assert mean_average_precision(run, qrels) == pytest.approx(5 / 12)


def test_map_ignores_queries_not_in_qrels():
    run = {"q1": RANKING, "extra": ["a"]}
    qrels = {"q1": RELEVANT}
    assert mean_average_precision(run, qrels) == pytest.approx(5 / 6)


def test_map_empty_qrels_returns_zero():
    assert mean_average_precision({"q1": RANKING}, {}) == 0.0


@pytest.mark.parametrize("k", [0, -1])
@pytest.mark.parametrize("metric", [precision_at_k, recall_at_k, average_precision])
def test_non_positive_k_raises(metric, k):
    with pytest.raises(ValueError):
        metric(RANKING, RELEVANT, k=k)


@pytest.mark.parametrize("metric", [precision_at_k, recall_at_k, average_precision])
def test_duplicated_ranking_raises(metric):
    with pytest.raises(ValueError):
        metric(["a", "b", "a"], RELEVANT, k=10)


def test_map_duplicated_ranking_raises():
    with pytest.raises(ValueError):
        mean_average_precision({"q1": ["a", "a"]}, {"q1": RELEVANT})
