"""Descarga y verifica artefactos publicados en Hugging Face Hub."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path, PurePosixPath

from huggingface_hub import hf_hub_download


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_relative_path(value: str) -> Path:
    relative = PurePosixPath(value)
    if relative.is_absolute() or not relative.parts or ".." in relative.parts:
        raise ValueError(f"Ruta de artefacto no válida: {value!r}")
    return Path(*relative.parts)


def ensure_artifacts(target_dir: str | Path, manifest_path: str | Path) -> Path:
    """Asegura los archivos manifestados y rechaza descargas con hash distinto."""
    target = Path(target_dir)
    manifest_file = Path(manifest_path)
    if not manifest_file.is_file():
        return target

    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    repo_id = manifest["repo_id"]
    revision = manifest["revision"]
    repo_type = manifest.get("repo_type", "dataset")
    if not isinstance(revision, str) or len(revision) != 40:
        raise ValueError("El manifiesto debe fijar revision a un SHA de commit de 40 caracteres")
    files = manifest["files"]
    if not isinstance(files, dict):
        raise TypeError("El manifiesto debe contener files como objeto")

    for relative_name, metadata in files.items():
        relative = _safe_relative_path(relative_name)
        expected_hash = metadata["sha256"]
        destination = target / relative
        if destination.is_file() and _sha256(destination) == expected_hash:
            continue

        target.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=target, prefix=".descarga-") as local_dir:
            downloaded = Path(
                hf_hub_download(
                    repo_id=repo_id,
                    filename=relative_name,
                    repo_type=repo_type,
                    revision=revision,
                    local_dir=local_dir,
                )
            )
            actual_hash = _sha256(downloaded)
            if actual_hash != expected_hash:
                destination.unlink(missing_ok=True)
                raise ValueError(
                    f"SHA-256 incorrecto para {relative_name}: "
                    f"esperado {expected_hash}, recibido {actual_hash}"
                )
            destination.parent.mkdir(parents=True, exist_ok=True)
            os.replace(downloaded, destination)
    return target
