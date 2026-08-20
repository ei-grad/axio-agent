from __future__ import annotations

import sys
from pathlib import Path

import pytest

from axio_repl import _version, main

_FULL_REVISION = "ABCDEF1234567890ABCDEF1234567890ABCDEF12"


def _write_manifest(project_dir: Path, *, name: str = "axio-repl") -> None:
    project_dir.mkdir(parents=True, exist_ok=True)
    (project_dir / "pyproject.toml").write_text(f'[project]\nname = "{name}"\n', encoding="utf-8")


def _write_git_directory(checkout_root: Path, *, write_ref: bool = True) -> Path:
    git_dir = checkout_root / ".git"
    (git_dir / "refs" / "heads").mkdir(parents=True)
    (git_dir / "HEAD").write_text("ref: refs/heads/main\n", encoding="ascii")
    if write_ref:
        (git_dir / "refs" / "heads" / "main").write_text(f"{_FULL_REVISION}\n", encoding="ascii")
    return git_dir


def _write_standalone_checkout(tmp_path: Path, *, write_ref: bool = True) -> tuple[Path, Path, Path]:
    root = tmp_path / "project"
    module = root / "src" / "axio_repl" / "__init__.py"
    module.parent.mkdir(parents=True)
    module.write_text("", encoding="utf-8")
    _write_manifest(root)
    return root, module, _write_git_directory(root, write_ref=write_ref)


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
    _, module, _ = _write_standalone_checkout(tmp_path)

    assert _version.git_revision(str(module)) == "abcdef123456"
    assert _version.git_revision(str(tmp_path / "wheel" / "axio_repl" / "__init__.py")) == "unavailable"


def test_git_revision_supports_a_monorepo_worktree_marker(tmp_path: Path) -> None:
    checkout = tmp_path / "checkout"
    module = checkout / "axio-repl" / "src" / "axio_repl" / "__init__.py"
    module.parent.mkdir(parents=True)
    module.write_text("", encoding="utf-8")
    _write_manifest(checkout / "axio-repl")
    common_dir = tmp_path / "git-data"
    git_dir = common_dir / "worktrees" / "checkout"
    git_dir.mkdir(parents=True)
    (checkout / ".git").write_text(f"gitdir: {git_dir}\n", encoding="utf-8")
    (git_dir / "commondir").write_text("../..\n", encoding="utf-8")
    (git_dir / "HEAD").write_text("ref: refs/heads/main\n", encoding="ascii")
    (common_dir / "refs" / "heads").mkdir(parents=True)
    (common_dir / "refs" / "heads" / "main").write_text(f"{_FULL_REVISION}\n", encoding="ascii")

    assert _version.git_revision(str(module)) == "abcdef123456"


def test_git_revision_ignores_an_unrelated_enclosing_repository(tmp_path: Path) -> None:
    outer = tmp_path / "dotfiles"
    module = outer / ".venv" / "lib" / "python" / "site-packages" / "axio_repl" / "__init__.py"
    module.parent.mkdir(parents=True)
    module.write_text("", encoding="utf-8")
    _write_manifest(outer, name="dotfiles")
    _write_git_directory(outer)

    assert _version.git_revision(str(module)) == "unavailable"


def test_axio_named_outer_checkout_does_not_claim_a_nested_wheel(tmp_path: Path) -> None:
    outer = tmp_path / "outer"
    module = outer / ".venv" / "site-packages" / "axio_repl" / "__init__.py"
    module.parent.mkdir(parents=True)
    module.write_text("", encoding="utf-8")
    _write_manifest(outer)
    _write_git_directory(outer)

    assert _version.git_revision(str(module)) == "unavailable"


def test_broken_nearest_git_marker_stops_search_before_an_outer_checkout(tmp_path: Path) -> None:
    outer = tmp_path / "outer"
    module = outer / "nested" / "src" / "axio_repl" / "__init__.py"
    module.parent.mkdir(parents=True)
    module.write_text("", encoding="utf-8")
    _write_manifest(outer)
    _write_git_directory(outer)
    (outer / "nested" / ".git").symlink_to(outer / "missing-git-dir", target_is_directory=True)

    assert _version.git_revision(str(module)) == "unavailable"


@pytest.mark.parametrize("layer", ["project", "head", "ref", "packed-refs"])
def test_undecodable_checkout_metadata_degrades_to_unavailable(tmp_path: Path, layer: str) -> None:
    root, module, git_dir = _write_standalone_checkout(tmp_path, write_ref=layer != "packed-refs")
    if layer == "project":
        (root / "pyproject.toml").write_bytes(b"\xff")
    elif layer == "head":
        (git_dir / "HEAD").write_bytes(b"\xff")
    elif layer == "ref":
        (git_dir / "refs" / "heads" / "main").write_bytes(b"\xff")
    else:
        (git_dir / "packed-refs").write_bytes(b"\xff")

    assert _version.git_revision(str(module)) == "unavailable"


def test_undecodable_gitdir_and_commondir_degrade_to_unavailable(tmp_path: Path) -> None:
    checkout = tmp_path / "checkout"
    module = checkout / "src" / "axio_repl" / "__init__.py"
    module.parent.mkdir(parents=True)
    module.write_text("", encoding="utf-8")
    _write_manifest(checkout)
    (checkout / ".git").write_bytes(b"\xff")
    assert _version.git_revision(str(module)) == "unavailable"

    second_checkout = tmp_path / "second-checkout"
    second_module = second_checkout / "src" / "axio_repl" / "__init__.py"
    second_module.parent.mkdir(parents=True)
    second_module.write_text("", encoding="utf-8")
    _write_manifest(second_checkout)
    git_dir = tmp_path / "git-data" / "worktrees" / "second"
    git_dir.mkdir(parents=True)
    (second_checkout / ".git").write_text(f"gitdir: {git_dir}\n", encoding="utf-8")
    (git_dir / "commondir").write_bytes(b"\xff")

    assert _version.git_revision(str(second_module)) == "unavailable"


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


async def test_version_action_exits_zero_with_corrupt_checkout_metadata(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    import axio_repl

    _, module, git_dir = _write_standalone_checkout(tmp_path)
    (git_dir / "HEAD").write_bytes(b"\xff")
    original_report = _version.version_report
    monkeypatch.setattr(
        axio_repl._version,
        "version_report",
        lambda **_kwargs: original_report(
            argv0="axio-repl",
            module_source=str(module),
            interpreter="python",
            distribution_version_provider=lambda: "9.8.7",
        ),
    )
    monkeypatch.setattr(sys, "argv", ["axio-repl", "--version"])

    with pytest.raises(SystemExit) as exited:
        await main()

    assert exited.value.code == 0
    assert "git revision: unavailable" in capsys.readouterr().out
