"""Contrato entre la app (#13) y las vistas de #14 y #15.

Cada vista es un módulo de `app/views/` que expone `render(context)`. La app
construye un `AppContext` en cada ejecución y llama solo a la vista activa.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ris.engine import SearchEngine, SearchResponse


@dataclass(frozen=True)
class AppContext:
    """Estado compartido que reciben las vistas. Inmutable: las vistas no lo modifican.

    - engine: motor único con los cuatro modelos ya cargados y en caché. Las
      vistas NO deben construir recuperadores ni leer índices por su cuenta.
    - query: consulta actual, sin espacios sobrantes; puede ser "" (no se ha escrito).
    - model: modelo elegido en la pantalla de búsqueda (jaccard, tfidf, bm25, dense).
    - k: cantidad de resultados pedida (top-n).
    - stem: si los modelos clásicos usan el índice con Porter. El denso lo ignora.
    - reports_dir: carpeta con las métricas de #11 (metrics_per_query.csv y
      metrics_summary.json). #15 debe validar metrics_per_query_sha256.
    """

    engine: SearchEngine
    query: str
    model: str
    k: int
    stem: bool
    reports_dir: Path

    def search(self, model: str | None = None) -> SearchResponse:
        """Busca con la consulta, k y stem actuales; `model` por defecto es el elegido.

        Lanza EngineError (EmptyQueryError, ArtifactsMissingError,
        ModelUnavailableError) con mensajes aptos para mostrar al usuario.
        """
        return self.engine.search(
            model or self.model, self.query, self.k, stem=self.stem
        )
