from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from ris import artifacts
from scripts import publish_artifacts


def _manifest(path: Path, source: Path, name: str = "nested/file.bin") -> Path:
    manifest = path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "repo_id": "team/artifacts",
                "repo_type": "dataset",
                "revision": "a" * 40,
                "files": {
                    name: {
                        "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                        "size": source.stat().st_size,
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    return manifest


def test_matching_artifact_does_not_download(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = tmp_path / "source.bin"
    source.write_bytes(b"complete")
    target = tmp_path / "target"
    destination = target / "nested/file.bin"
    destination.parent.mkdir(parents=True)
    destination.write_bytes(source.read_bytes())
    manifest = _manifest(tmp_path, source)

    def fail_download(**_: object) -> Path:
        raise AssertionError("no debe descargar")

    monkeypatch.setattr(artifacts, "hf_hub_download", fail_download)
    assert artifacts.ensure_artifacts(target, manifest) == target


def _download_to_local_dir(kwargs: dict[str, object], target: Path, payload: bytes) -> str:
    local_dir = Path(kwargs["local_dir"])
    assert local_dir.parent == target
    assert local_dir.name.startswith(".descarga-")
    assert local_dir.is_dir()
    assert kwargs == {
        "repo_id": "team/artifacts",
        "filename": "nested/file.bin",
        "repo_type": "dataset",
        "revision": "a" * 40,
        "local_dir": str(local_dir),
    }
    downloaded = local_dir / "nested/file.bin"
    downloaded.parent.mkdir(parents=True)
    downloaded.write_bytes(payload)
    return str(downloaded)


@pytest.mark.parametrize("corrupt_existing", [False, True])
def test_missing_artifact_downloads_and_verifies(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, corrupt_existing: bool
) -> None:
    source = tmp_path / "source.bin"
    source.write_bytes(b"valid payload")
    target = tmp_path / "target"
    destination = target / "nested/file.bin"
    if corrupt_existing:
        destination.parent.mkdir(parents=True)
        destination.write_bytes(b"old corrupt")
    manifest = _manifest(tmp_path, source)
    calls: list[dict[str, object]] = []

    def download(**kwargs: object) -> str:
        calls.append(kwargs)
        return _download_to_local_dir(kwargs, target, source.read_bytes())

    monkeypatch.setattr(artifacts, "hf_hub_download", download)
    assert artifacts.ensure_artifacts(target, manifest) == target
    assert destination.read_bytes() == source.read_bytes()
    assert len(calls) == 1
    assert not list(target.glob(".descarga-*"))


@pytest.mark.parametrize("corrupt_existing", [False, True])
def test_bad_download_hash_raises_and_leaves_no_corrupt_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, corrupt_existing: bool
) -> None:
    source = tmp_path / "source.bin"
    source.write_bytes(b"expected")
    target = tmp_path / "target"
    destination = target / "nested/file.bin"
    if corrupt_existing:
        destination.parent.mkdir(parents=True)
        destination.write_bytes(b"old corrupt")
    manifest = _manifest(tmp_path, source)

    def download(**kwargs: object) -> str:
        return _download_to_local_dir(kwargs, target, b"wrong")

    monkeypatch.setattr(artifacts, "hf_hub_download", download)
    with pytest.raises(ValueError, match="SHA-256 incorrecto"):
        artifacts.ensure_artifacts(target, manifest)
    assert not destination.exists()
    assert not list(target.glob(".descarga-*"))


def test_download_error_cleans_temporary_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source.bin"
    source.write_bytes(b"expected")
    target = tmp_path / "target"
    manifest = _manifest(tmp_path, source)

    def download(**kwargs: object) -> str:
        _download_to_local_dir(kwargs, target, b"partial")
        raise OSError("download interrupted")

    monkeypatch.setattr(artifacts, "hf_hub_download", download)
    with pytest.raises(OSError, match="download interrupted"):
        artifacts.ensure_artifacts(target, manifest)
    assert not (target / "nested/file.bin").exists()
    assert not list(target.glob(".descarga-*"))


def test_missing_manifest_does_nothing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    target = tmp_path / "not-created"
    monkeypatch.setattr(
        artifacts, "hf_hub_download", lambda **_: pytest.fail("no debe descargar")
    )
    assert artifacts.ensure_artifacts(target, tmp_path / "absent.json") == target
    assert not target.exists()


def test_publish_dry_run_hashes_and_sizes_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    artifact_root = tmp_path / "artifacts"
    artifact_root.mkdir()
    (artifact_root / "one.bin").write_bytes(b"abc")
    (artifact_root / "nested").mkdir()
    (artifact_root / "nested/two.bin").write_bytes(b"12345")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "sys.argv",
        ["publish_artifacts.py", "--artifacts-dir", str(artifact_root), "--dry-run"],
    )
    publish_artifacts.main()
    assert publish_artifacts.collect_files(artifact_root) == {
        "nested/two.bin": {"sha256": hashlib.sha256(b"12345").hexdigest(), "size": 5},
        "one.bin": {"sha256": hashlib.sha256(b"abc").hexdigest(), "size": 3},
    }
    assert not (tmp_path / "deploy/artifacts.json").exists()
