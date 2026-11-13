import argparse
import json

import pytest

from scripts import build_artifacts


def test_dense_script_saves_verifies_and_reports_without_network(monkeypatch, tmp_path):
    from test_dense import Encoder

    from ris.data import Doc
    from ris.retrievers.dense import DenseRetriever

    corpus = {
        "d1": Doc("Title", "Original, $3.50!"),
        "d2": Doc("", "second"), "d3": Doc("", ""),
    }
    monkeypatch.setattr(build_artifacts, "load_corpus", lambda: corpus)
    original_build = DenseRetriever.build

    def build(corpus, **kwargs):
        from test_dense import vector

        encoder = Encoder()
        encoder.vectors["How should I invest my savings?"] = vector(1, 2)
        return original_build(corpus, encoder=encoder, **kwargs)

    monkeypatch.setattr(DenseRetriever, "build", build)
    args = argparse.Namespace(limit=2, output_dir=tmp_path, batch_size=2,
                              threads=1, overwrite=False)
    build_artifacts._build_dense(args)
    metadata = json.loads((tmp_path / "metadata.json").read_text(encoding="utf-8"))
    report = json.loads((tmp_path / "build_report.json").read_text(encoding="utf-8"))
    assert metadata["doc_ids"] == ["d1", "d2"]
    assert report["documents"] == 2
    assert report["full_corpus_documents"] == 3
    assert report["threads"] == 1
    assert report["verification_query"] == "How should I invest my savings?"
    assert len(report["verification_ranking"]) == 2
    assert report["peak_process_memory_mib"] > 0
    assert report["total_seconds"] >= report["build_seconds"] > 0
    assert report["artifact_bytes"]["index.faiss"] > 0
    assert DenseRetriever.load(tmp_path).doc_ids == ("d1", "d2")


def test_dense_script_preserves_existing_artifact_before_loading_corpus(monkeypatch, tmp_path):
    marker = tmp_path / "metadata.json"
    marker.write_text("existing", encoding="utf-8")
    monkeypatch.setattr(build_artifacts, "load_corpus", lambda: pytest.fail("No cargar"))
    args = argparse.Namespace(limit=None, output_dir=tmp_path, overwrite=False)
    with pytest.raises(ValueError, match="overwrite"):
        build_artifacts._build_dense(args)
    assert marker.read_text(encoding="utf-8") == "existing"


@pytest.mark.parametrize("arguments", [["--dense", "--stem"], ["--limit", "2"],
                                     ["--dense", "--batch-size", "0"]])
def test_invalid_cli_options_fail_before_corpus_load(monkeypatch, arguments):
    monkeypatch.setattr("sys.argv", ["build_artifacts.py", *arguments])
    monkeypatch.setattr(build_artifacts, "load_corpus", lambda: pytest.fail("No cargar"))
    with pytest.raises(SystemExit) as error:
        build_artifacts.main()
    assert error.value.code == 2


@pytest.mark.parametrize("stem", [False, True])
def test_existing_classical_cli_still_builds_and_reloads(monkeypatch, tmp_path, stem):
    from ris.data import Doc
    from ris.index import InvertedIndex

    corpus = {"doc": Doc("Bank", "loan")}
    monkeypatch.setattr(build_artifacts, "load_corpus", lambda: corpus)
    monkeypatch.setattr("ris.index.preprocess", lambda text, **kwargs: text.lower().split())
    arguments = ["build_artifacts.py", "--output-dir", str(tmp_path)]
    if stem:
        arguments.append("--stem")
    monkeypatch.setattr("sys.argv", arguments)
    build_artifacts.main()
    path = tmp_path / ("index_stem.json" if stem else "index_no_stem.json")
    loaded = InvertedIndex.load(path, expected_stem=stem)
    assert loaded.postings == {"bank": {"doc": 1}, "loan": {"doc": 1}}
    assert loaded.N == 1
