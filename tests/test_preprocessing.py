import nltk
import pytest

from ris import preprocessing
from ris.preprocessing import preprocess, setup_nltk


def test_preprocess_lowercase():
    text = "FINANCIAL Markets And STOCKS"
    assert preprocess(text) == ["financial", "markets", "stocks"]


def test_preprocess_removes_punctuation_and_keeps_numbers():
    text = "Hello, world! Investing $100 (or 25%) in 2026 for a 401k..."
    assert preprocess(text) == ["hello", "world", "investing", "100", "25", "2026", "401k"]


def test_preprocess_removes_english_stopwords():
    text = "The investment is in the market and it is for the future"
    assert preprocess(text) == ["investment", "market", "future"]
    assert preprocess("the and or but in on at to for of") == []


def test_preprocess_empty_and_whitespace_text():
    assert preprocess("") == []
    assert preprocess("   \t\n  ") == []
    assert preprocess("... !!! ???") == []


def test_preprocess_stem_false_and_stem_true():
    text = "The investors are running and investing in financial markets"

    assert preprocess(text, stem=False) == [
        "investors",
        "running",
        "investing",
        "financial",
        "markets",
    ]
    assert preprocess(text, stem=True) == [
        "investor",
        "run",
        "invest",
        "financi",
        "market",
    ]


def test_preprocess_requires_keyword_argument_for_stem():
    with pytest.raises(TypeError):
        preprocess("financial markets", True)  # type: ignore[misc]


def test_setup_nltk_downloads_only_missing_resources(monkeypatch):
    downloaded = []

    def fake_find(resource_path):
        if resource_path == "tokenizers/punkt_tab":
            raise LookupError(resource_path)
        return True

    def fake_download(package_name, quiet=True):
        downloaded.append((package_name, quiet))
        return True

    monkeypatch.setattr(nltk.data, "find", fake_find)
    monkeypatch.setattr(nltk, "download", fake_download)

    setup_nltk()

    assert downloaded == [("punkt_tab", True)]
    assert hasattr(preprocessing, "preprocess")
