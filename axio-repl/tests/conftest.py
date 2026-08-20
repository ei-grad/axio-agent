"""Shared test fixtures for axio-repl."""

from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def repl_history_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Keep prompt-toolkit history writes inside each test's temporary directory."""
    from axio_repl import _panel

    history_path = tmp_path / "history"
    monkeypatch.setattr(_panel, "HISTORY_PATH", history_path)
    return history_path


@pytest.fixture(autouse=True)
def isolated_axio_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Prevent main() tests from inheriting the developer's Axio profile."""

    config_dir = tmp_path / "axio-config"
    config_dir.mkdir()
    monkeypatch.setenv("AXIO_CONFIG_DIR", str(config_dir))
    return config_dir
