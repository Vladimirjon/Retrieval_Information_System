"""Construcción y persistencia de un índice invertido."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Callable
from pathlib import Path

from ris.data import Doc
from ris.preprocessing import preprocess


class InvertedIndex:
    """Índice invertido para los modelos clásicos de recuperación."""

    def __init__(self, *, stem: bool = False) -> None:
        self.stem = stem
        self.postings: dict[str, dict[str, int]] = {}
        self.df: dict[str, int] = {}
        self.doc_lengths: dict[str, int] = {}
        self.N = 0

    @classmethod
    def build(
        cls,
        corpus: dict[str, Doc],
        *,
        stem: bool = False,
        progress_callback: Callable[[int, int, int], None] | None = None,
    ) -> InvertedIndex:
        """Construye un índice a partir del corpus."""
        index = cls(stem=stem)
        index.N = len(corpus)

        for processed, (doc_id, doc) in enumerate(corpus.items(), start=1):
            text = f"{doc.title} {doc.text}"
            tokens = preprocess(text, stem=stem)
            frequencies = Counter(tokens)

            index.doc_lengths[doc_id] = len(tokens)

            for term, tf in frequencies.items():
                index.postings.setdefault(term, {})[doc_id] = tf
                index.df[term] = index.df.get(term, 0) + 1

            if progress_callback is not None:
                progress_callback(processed, index.N, len(index.postings))

        return index

    def save(self, path: str | Path) -> None:
        """Guarda el índice y sus metadatos en un archivo JSON."""
        data = {
            "format_version": 1,
            "stem": self.stem,
            "N": self.N,
            "postings": self.postings,
            "df": self.df,
            "doc_lengths": self.doc_lengths,
        }

        with Path(path).open("w", encoding="utf-8", newline="\n") as file:
            json.dump(data, file, ensure_ascii=False)

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        expected_stem: bool | None = None,
    ) -> InvertedIndex:
        """Carga un índice y comprueba su versión y configuración."""
        with Path(path).open("r", encoding="utf-8") as file:
            data = json.load(file)

        if data["format_version"] != 1:
            raise ValueError("Versión de formato del índice no compatible")

        if not isinstance(data["stem"], bool):
            raise TypeError("La configuración de stemming debe ser booleana")

        if expected_stem is not None and data["stem"] != expected_stem:
            raise ValueError("El índice tiene una configuración de stemming distinta")

        index = cls(stem=data["stem"])
        index.N = data["N"]
        index.postings = data["postings"]
        index.df = data["df"]
        index.doc_lengths = data["doc_lengths"]

        return index