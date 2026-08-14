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
