"""Shared test fixtures for axio-repl."""

from pathlib import Path

import pytest


@pytest.fixture
def repl_history_path(tmp_path: Path) -> Path:
    """Return an explicit persistent history path for prompt behavior tests."""

    return tmp_path / "history"


@pytest.fixture(autouse=True)
def isolated_repl_state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Prevent main() and child processes from writing to the developer's state."""

    state_dir = tmp_path / "state"
    monkeypatch.setenv("XDG_STATE_HOME", str(state_dir))
    return state_dir


@pytest.fixture(autouse=True)
def isolated_axio_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Prevent main() tests from inheriting the developer's Axio profile."""

    config_dir = tmp_path / "axio-config"
    config_dir.mkdir()
    monkeypatch.setenv("AXIO_CONFIG_DIR", str(config_dir))
    return config_dir
