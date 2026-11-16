"""Preprocesamiento léxico en inglés para documentos y consultas del corpus FiQA."""

from __future__ import annotations

import re
from functools import lru_cache

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

_NLTK_RESOURCES = (
    ("tokenizers/punkt_tab", "punkt_tab"),
    ("corpora/stopwords", "stopwords"),
)
_SEPARATORS = re.compile(r"[\W_]+")


def setup_nltk() -> None:
    """Asegura que los recursos de NLTK estén disponibles localmente."""
    for resource_path, package_name in _NLTK_RESOURCES:
        try:
            nltk.data.find(resource_path)
        except LookupError:
            # Se descarga bajo demanda solo si el recurso no está instalado en el entorno
            nltk.download(package_name, quiet=True)


@lru_cache(maxsize=1)
def _get_stopwords() -> frozenset[str]:
    setup_nltk()
    return frozenset(stopwords.words("english"))


def preprocess(text: str, *, stem: bool = False) -> list[str]:
    """Tokeniza, normaliza y filtra un texto en inglés para los modelos clásicos.

    Política de números y puntuación:
    - Un token alfanumérico (`token.isalnum()`) se conserva tal cual.
    - Cualquier otro token se parte en sus trozos alfanuméricos en lugar de
      descartarse: 'goog-stock' → 'goog', 'stock'; '1.000' → '1', '000';
      'pros/cons' → 'pros', 'cons'. Los símbolos sueltos no dejan trozos.
    - La contracción "n't" (de "isn't") se descarta; las demás ('s, 'll...)
      quedan como stopwords.
    - Números: se conservan los tokens numéricos y alfanuméricos (ej. '2026',
      '500', '401k') por su relevancia en el dominio financiero de FiQA.
    """
    if not text or not text.strip():
        return []

    stop_words = _get_stopwords()
    tokens = word_tokenize(text.lower())
    filtered_tokens = [
        piece
        for token in tokens
        for piece in alnum_pieces(token)
        if piece not in stop_words
    ]

    if stem:
        stemmer = PorterStemmer()
        return [stemmer.stem(token) for token in filtered_tokens]

    return filtered_tokens


def alnum_pieces(token: str) -> list[str]:
    """Trozos alfanuméricos de un token de `word_tokenize` (sin quitar stopwords)."""
    if token.isalnum():
        return [token]
    if token == "n't":
        return []
    return [piece for piece in _SEPARATORS.split(token) if piece]
