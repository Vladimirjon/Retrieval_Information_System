"""Comprobaciones sin red del archivo de dependencias de Community Cloud."""

import re
from pathlib import Path
from urllib.parse import unquote, urlparse

ROOT = Path(__file__).resolve().parents[1]


def _requirements(path: Path) -> list[str]:
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def test_app_pins_match_root_requirements() -> None:
    root = _requirements(ROOT / "requirements.txt")
    app = _requirements(ROOT / "app/requirements.txt")
    pins = [line for line in app if "==" in line]
    assert pins
    for line in pins:
        assert re.fullmatch(r"[\w.-]+==[\w.+-]+", line)
        assert line in root
    assert all("==" in line or line.startswith("torch @ ") for line in app)


def test_app_has_no_index_options() -> None:
    app = _requirements(ROOT / "app/requirements.txt")
    assert not any(line.startswith(("--extra-index-url", "--index-url")) for line in app)


def test_torch_direct_url_matches_root_cpu_pin() -> None:
    root = _requirements(ROOT / "requirements.txt")
    app = _requirements(ROOT / "app/requirements.txt")
    torch_pins = [line for line in root if line.startswith("torch==")]
    torch_urls = [line for line in app if line.startswith("torch @ ")]
    assert len(torch_pins) == len(torch_urls) == 1
    version = torch_pins[0].removeprefix("torch==")
    assert version.endswith("+cpu")
    url = urlparse(torch_urls[0].removeprefix("torch @ "))
    assert url.scheme == "https"
    assert url.netloc == "download.pytorch.org"
    assert unquote(url.path) == (
        f"/whl/cpu/torch-{version}-cp312-cp312-manylinux_2_28_x86_64.whl"
    )


def test_app_excludes_development_dependencies() -> None:
    app = _requirements(ROOT / "app/requirements.txt")
    names = {re.split(r"[= @]", line, maxsplit=1)[0].lower() for line in app}
    assert names.isdisjoint({"jupyterlab", "ipykernel", "matplotlib", "seaborn", "pytest", "ruff"})
