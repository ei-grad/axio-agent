from __future__ import annotations

import sys
from pathlib import Path

import pytest

from axio_repl import _version, main


def test_version_report_uses_injected_provenance_sources(tmp_path: Path) -> None:
    launcher = tmp_path / "bin" / "axio-repl"
    module = tmp_path / "src" / "axio_repl" / "__init__.py"
    interpreter = tmp_path / "python"

    report = _version.version_report(
        argv0=str(launcher),
        module_source=str(module),
        interpreter=str(interpreter),
        distribution_version_provider=lambda: "9.8.7",
        git_revision_provider=lambda source: "abc123def456" if source == str(module.resolve()) else "wrong",
    )

    assert report.splitlines() == [
        "axio-repl 9.8.7",
        f"launcher: {launcher.resolve()}",
        f"module: {module.resolve()}",
        f"interpreter: {interpreter.resolve()}",
        "git revision: abc123def456",
    ]


def test_git_revision_reads_a_checkout_without_running_git(tmp_path: Path) -> None:
    module = tmp_path / "project" / "axio_repl" / "__init__.py"
    module.parent.mkdir(parents=True)
    module.write_text("", encoding="utf-8")
    git_dir = tmp_path / "project" / ".git"
    (git_dir / "refs" / "heads").mkdir(parents=True)
    (git_dir / "HEAD").write_text("ref: refs/heads/main\n", encoding="ascii")
    (git_dir / "refs" / "heads" / "main").write_text(
        "ABCDEF1234567890ABCDEF1234567890ABCDEF12\n",
        encoding="ascii",
    )

    assert _version.git_revision(str(module)) == "abcdef123456"
    assert _version.git_revision(str(tmp_path / "wheel" / "axio_repl" / "__init__.py")) == "unavailable"


async def test_version_exits_before_configuration_sandbox_or_transport_initialization(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import axio_repl

    def unexpected(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("runtime initialization must not run for --version")

    monkeypatch.setattr(axio_repl._agent_config, "default_config_dir", unexpected)
    monkeypatch.setattr(axio_repl._agent_config, "load_agent_profile", unexpected)
    monkeypatch.setattr(axio_repl._sandbox, "build_tools", unexpected)
    monkeypatch.setattr(axio_repl, "_select_transport", unexpected)
    monkeypatch.setattr(axio_repl._version, "version_report", lambda **_kwargs: "deterministic provenance")
    monkeypatch.setattr(sys, "argv", ["axio-repl", "--version"])

    with pytest.raises(SystemExit) as exited:
        await main()

    assert exited.value.code == 0
    assert capsys.readouterr().out == "deterministic provenance\n"
