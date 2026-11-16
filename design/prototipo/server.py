"""Servidor del prototipo de jab: sirve los archivos estáticos y busca con el motor real.

Solo para revisar el diseño en local; no es la app de entrega. Desde la raíz del repo:
    python design/prototipo/server.py        (http://localhost:8700)
"""

from __future__ import annotations

import json
import sys
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT / "src"))

from ris.data import load_queries
from ris.engine import MODELS, EngineError, SearchEngine

ENGINE = SearchEngine(ROOT / "artifacts")
for _model in MODELS:
    try:
        ENGINE.warm_up((_model,))
    except EngineError as error:
        print(f"Aviso: {error}", flush=True)
SUGGESTIONS = sorted({" ".join(q.split()) for q in load_queries("train").values()})


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(HERE), **kwargs)

    def log_message(self, *args):  # silencio en consola
        pass

    def _json(self, payload: dict, status: int = 200) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        url = urlparse(self.path)
        if url.path == "/api/suggestions":
            return self._json({"items": SUGGESTIONS})
        if url.path == "/api/search":
            q = parse_qs(url.query)
            try:
                response = ENGINE.search(
                    q.get("model", ["bm25"])[0], q.get("q", [""])[0],
                    int(q.get("k", ["10"])[0]), stem=q.get("stem", ["0"])[0] == "1",
                )
            except (EngineError, ValueError) as error:
                return self._json({"error": str(error)}, 400)
            return self._json({
                "model": response.model, "stem": response.stem,
                "elapsed_ms": round(response.elapsed_ms, 1),
                "hits": [
                    {"rank": h.rank, "doc_id": h.doc_id, "text": h.text, "score": h.score}
                    for h in response.hits
                ],
            })
        return super().do_GET()


if __name__ == "__main__":
    print("Prototipo en http://localhost:8700", flush=True)
    ThreadingHTTPServer(("127.0.0.1", 8700), Handler).serve_forever()
