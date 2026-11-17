"""Publica artefactos locales en Hugging Face Hub y genera el manifiesto."""

from __future__ import annotations

import argparse
import hashlib
import json
from io import BytesIO
from pathlib import Path

DEFAULT_REPO_ID = "VLADIMIRJON/jab-fiqa-artifacts"
DEFAULT_ARTIFACTS_DIR = Path("artifacts")
MANIFEST_PATH = Path("deploy/artifacts.json")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def collect_files(root: Path) -> dict[str, dict[str, int | str]]:
    files = sorted(path for path in root.rglob("*") if path.is_file())
    if not files:
        raise ValueError(f"No hay archivos de artefactos en {root}")
    return {
        path.relative_to(root).as_posix(): {
            "sha256": sha256(path),
            "size": path.stat().st_size,
        }
        for path in files
    }


def dataset_card() -> str:
    return """# jab FiQA artifacts

Artefactos derivados del dataset **BeIR/fiqa**. Para la procedencia y la licencia
de origen, consulte la tarjeta de [BeIR/fiqa](https://huggingface.co/datasets/BeIR/fiqa);
este repositorio no atribuye una licencia distinta.

Se generaron con `python scripts/build_artifacts.py`, incluyendo los índices
clásicos sin y con Porter y los embeddings/índice FAISS densos. El corpus FiQA y
el modelo usados están fijados en `src/ris/data.py` y en la configuración del
artefacto denso.
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--artifacts-dir", type=Path, default=DEFAULT_ARTIFACTS_DIR)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    root = args.artifacts_dir.resolve()
    files = collect_files(root)
    print(f"Repositorio dataset: {args.repo_id}")
    print(f"Artefactos: {root}")
    for name, metadata in files.items():
        print(f"{name}\t{metadata['size']} bytes\tsha256:{metadata['sha256']}")
    print("README.md\tTarjeta de procedencia del dataset")
    if args.dry_run:
        print("Dry run: no se contactó Hugging Face Hub ni se modificó el manifiesto.")
        return

    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(repo_id=args.repo_id, repo_type="dataset", private=False, exist_ok=True)
    commit = api.upload_folder(
        repo_id=args.repo_id,
        repo_type="dataset",
        folder_path=str(root),
        path_in_repo="",
        commit_message="Publish FiQA retrieval artifacts",
    )
    api.upload_file(
        path_or_fileobj=BytesIO(dataset_card().encode("utf-8")),
        path_in_repo="README.md",
        repo_id=args.repo_id,
        repo_type="dataset",
        commit_message="Add dataset provenance card",
    )
    # El manifiesto debe apuntar al commit que contiene tanto los binarios como README.
    revision = api.repo_info(repo_id=args.repo_id, repo_type="dataset").sha
    if not revision or len(revision) != 40:
        raise RuntimeError("Hugging Face no devolvió un SHA de commit válido")
    manifest = {
        "repo_id": args.repo_id,
        "repo_type": "dataset",
        "revision": revision,
        "files": files,
    }
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    print(f"Manifiesto escrito en {MANIFEST_PATH} (commit {revision}; {commit.commit_url})")


if __name__ == "__main__":
    main()
