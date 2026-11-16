"""Casos manuales de validación, agrupación, pareado y selección de #12."""

import hashlib
import json
from copy import deepcopy
from pathlib import Path

import pytest

from ris import comparison as c
from ris.data import Doc
from ris.retrievers import Result


def small_rows(qids=("02", "10")):
    return [{"model": model, "stem": stem, "qid": qid, "k": 10,
             "precision": value, "recall": value, "ap_at_1000": value}
            for model, stem in c.CONFIGURATIONS
            for qid, value in zip(qids, (.0, .5), strict=True)]


def write_reports(path, rows):
    header = ["model", "stem", "qid", "k", *c.METRICS]
    raw = (",".join(header) + "\n" + "\n".join(
        ",".join(str(r[key]) for key in header) for r in rows) + "\n").encode()
    summary = {"schema_version": 1, "config": {
        "split": "test", "k": 10, "ap_cutoff": 1000, "depth": 1000,
        "models": ["jaccard", "tfidf", "bm25", "dense"],
        "stem_values": {model: [s for m, s in c.CONFIGURATIONS if m == model]
                        for model in ("jaccard", "tfidf", "bm25", "dense")},
    }, "provenance": {"metrics_per_query_sha256": hashlib.sha256(raw).hexdigest()},
        "counts": {"n_rows": len(rows), "n_queries": 2}, "results": [
            {"model": model, "stem": stem, "n_queries": 2,
             "mean_precision_at_k": .25, "mean_recall_at_k": .25, "map_at_1000": .25}
            for model, stem in c.CONFIGURATIONS]}
    (path / "metrics_per_query.csv").write_bytes(raw)
    (path / "metrics_summary.json").write_text(json.dumps(summary), "utf-8")
    return summary


def test_validate_preserves_string_ids_and_checks_means(tmp_path):
    write_reports(tmp_path, small_rows())
    rows, summary = c.validate_reports(tmp_path, {"02", "10"})
    assert rows[0]["qid"] == "02"
    expected = c.aggregate(rows, ["02", "10"])
    for actual, expected_entry in zip(summary["results"], expected, strict=True):
        assert actual["n_queries"] == expected_entry["n"]
        assert actual["mean_precision_at_k"] == expected_entry["precision"]
        assert actual["mean_recall_at_k"] == expected_entry["recall"]
        assert actual["map_at_1000"] == expected_entry["ap_at_1000"]


@pytest.mark.parametrize("mutation", ["hash", "count", "mean", "split", "depth"])
def test_rejects_inconsistent_metadata(tmp_path, mutation):
    summary = write_reports(tmp_path, small_rows())
    if mutation == "hash":
        summary["provenance"]["metrics_per_query_sha256"] = "0" * 64
    elif mutation == "count":
        summary["counts"]["n_rows"] -= 1
    elif mutation == "mean":
        summary["results"][0]["map_at_1000"] += 1e-9
    else:
        summary["config"][mutation] = "train" if mutation == "split" else 999
    (tmp_path / "metrics_summary.json").write_text(json.dumps(summary), "utf-8")
    with pytest.raises(ValueError):
        c.validate_reports(tmp_path, {"02", "10"})


@pytest.mark.parametrize("mutation", ["duplicate", "missing", "stem", "nan", "k"])
def test_rejects_invalid_rows_even_with_matching_hash(tmp_path, mutation):
    rows = small_rows()
    if mutation == "duplicate":
        rows[-1] = deepcopy(rows[-2])
    elif mutation == "missing":
        rows.pop()
    elif mutation == "stem":
        rows[-1]["stem"] = "true"  # el denso nunca tiene Porter
    elif mutation == "nan":
        rows[0]["precision"] = float("nan")
    else:
        rows[0]["k"] = 5
    write_reports(tmp_path, rows)
    with pytest.raises(ValueError):
        c.validate_reports(tmp_path, {"02", "10"})


def test_group_boundaries_and_overlapping_families(monkeypatch):
    monkeypatch.setattr(c, "preprocess", lambda text, *, stem: text.split())
    queries = {"a": "IRA 401k now", "b": "one two three four",
               "c": "one two three four five six seven", "d": "1 2 3 4 5 6 7 8"}
    groups = c.query_groups(queries)
    assert groups["length_0_3"] == ["a"]
    assert groups["length_4_7"] == ["b", "c"]
    assert groups["length_8_plus"] == ["d"]
    assert groups["digits_yes"] == ["a", "d"]
    assert groups["uppercase_yes"] == ["a"]
    assert sum(map(len, groups.values())) == 3 * len(queries)
    assert set(groups["length_0_3"]) & set(groups["digits_yes"]) & set(
        groups["uppercase_yes"]
    )


def test_paired_deltas_counts_percentiles_and_zero_relative():
    rows = small_rows()
    for row in rows:
        if row["model"] == "dense":
            row["ap_at_1000"] = .25
    diff = c.paired_differences(rows, ("dense", "na"), c.BASELINE, ["02", "10"])
    assert diff["ap_at_1000"]["improved"] == 1
    assert diff["ap_at_1000"]["worsened"] == 1
    assert diff["ap_at_1000"]["percentiles"]["median"] == 0
    assert diff["precision"]["tied"] == 2
    for row in rows:
        if (row["model"], row["stem"]) == c.BASELINE:
            row["precision"] = 0
    diff = c.paired_differences(rows, ("dense", "na"), c.BASELINE, ["02", "10"])
    assert diff["precision"]["relative_mean_delta"] is None


def test_case_selection_extrema_lexicographic_ties_and_no_case():
    rows = small_rows(("9", "10"))
    assert c.select_cases(rows) == {"dense_improves": None, "dense_worsens": None}
    for row in rows:
        if (row["model"], row["stem"]) == c.BASELINE:
            row["ap_at_1000"] = 0
        if row["model"] == "dense":
            row["ap_at_1000"] = 1
    assert c.select_cases(rows)["dense_improves"]["qid"] == "10"
    for row in rows:
        if row["qid"] == "10" and row["model"] == "dense":
            row["ap_at_1000"] = -.1
    assert c.select_cases(rows)["dense_worsens"]["qid"] == "10"


def test_independent_metrics_all_relevants_in_denominator_and_cutoff():
    result = c.metrics_from_positions([1, 3, 1001, None], 4)
    assert result == {"precision": .2, "recall": .5, "ap_at_1000": pytest.approx(5/12)}
    assert c.metrics_from_positions([], 0) == dict.fromkeys(c.METRICS, 0)
    with pytest.raises(ValueError):
        c.metrics_from_positions([1, 1], 2)


def test_case_distinguishes_unjudged_nonrelevant_and_missing():
    corpus = {doc: Doc("", doc * 700) for doc in ("a", "b", "c", "d")}
    ranking = [Result("a", .9), Result("b", .8), Result("c", .7)]
    expected = {"x": {"precision": .1, "recall": .5, "ap_at_1000": .5}}
    evidence = c.case_evidence("02", "query", {"a": 1, "b": 0, "d": 1}, corpus,
                               {"x": ranking}, expected)
    docs = evidence["rankings"]["x"]["documents"]
    assert [doc["judgment"] for doc in docs] == [
        "relevante", "juzgado no relevante", "no juzgado", "relevante"]
    assert docs[-1]["rank"] is None
    assert docs[-1]["rank_status"] == "no recuperado en top-1000"
    assert docs[0]["snippet"] == corpus["a"].text[:600]
    expected["x"]["precision"] = .9
    with pytest.raises(ValueError, match="difiere del CSV"):
        c.case_evidence("02", "query", {"a": 1, "b": 0, "d": 1}, corpus,
                        {"x": ranking}, expected)


def test_committed_analysis_is_consistent_with_inputs_and_case_positions():
    """Guarda de los entregables reales sin red, corpus ni artefactos pesados."""
    root = Path(__file__).resolve().parents[1]
    summary = json.loads((root / "reports/comparison_summary.json").read_text("utf-8"))
    assert summary["provenance"]["metrics_per_query_sha256"] == c.sha256(
        root / "reports/metrics_per_query.csv"
    )
    groups = summary["groups"]
    qids = set(groups["digits_yes"]["qids"]) | set(groups["digits_no"]["qids"])
    rows, _ = c.validate_reports(root / "reports", qids)
    assert len(qids) == 648
    assert summary["overall"] == c.aggregate(rows, sorted(qids))
    for family in (("length_0_3", "length_4_7", "length_8_plus"),
                   ("digits_yes", "digits_no"), ("uppercase_yes", "uppercase_no")):
        ids = [qid for name in family for qid in groups[name]["qids"]]
        assert len(ids) == len(set(ids)) == 648
        assert set(ids) == qids
    for group in groups.values():
        assert group["n"] == len(group["qids"])
        assert all(type(qid) is str for qid in group["qids"])
        assert group["means"] == c.aggregate(rows, group["qids"])
    selected = c.select_cases(rows)
    for kind, case in summary["cases"].items():
        assert case["qid"] == selected[kind]["qid"]
        for ranking in case["rankings"].values():
            documents = {d["doc_id"]: d for d in ranking["documents"]}
            positions = [documents[doc]["rank"] for doc in case["relevant_doc_ids"]]
            assert c.metrics_from_positions(positions, case["n_relevant"]) == ranking["metrics"]
            assert all(type(doc) is str for doc in documents)
            assert sum(d["rank"] is not None and d["rank"] <= 10
                       for d in documents.values()) == 10
