"""Vista de comparación lado a lado. La implementa #14 (Ismael Freire).

Contrato: `render(context)` recibe un `app.context.AppContext`. Usa
`context.search(model)` para cada modelo de `ris.engine.MODELS`; no construyas
recuperadores propios. Si `context.query` está vacío, muestra un aviso en lugar
de buscar. Captura `ris.engine.EngineError` y muéstrala con `st.error`.
"""

from __future__ import annotations

import streamlit as st

from app.context import AppContext


def render(context: AppContext) -> None:
    st.info("La comparación lado a lado se implementa en la issue #14.")
