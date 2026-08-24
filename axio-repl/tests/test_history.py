from __future__ import annotations

import multiprocessing
import os
import stat
from pathlib import Path
from typing import Any

import pytest

from axio_repl._history import PrivateFileHistory, default_state_home, project_history_path


def _append_history(filename: str, prefix: str, count: int) -> None:
    history = PrivateFileHistory(Path(filename))
    for index in range(count):
        history.store_string(f"{prefix}-{index}")


def test_default_state_home_uses_absolute_xdg_override(tmp_path: Path) -> None:
    state_home = tmp_path / "state"

    assert default_state_home(environ={"XDG_STATE_HOME": str(state_home)}, home=tmp_path / "home") == state_home


def test_default_state_home_ignores_relative_xdg_override(tmp_path: Path) -> None:
    home = tmp_path / "home"

    assert default_state_home(environ={"XDG_STATE_HOME": "relative"}, home=home) == home / ".local" / "state"
    assert default_state_home(environ={"XDG_STATE_HOME": "~/state"}, home=home) == home / ".local" / "state"
    assert default_state_home(environ={"XDG_STATE_HOME": "~other/state"}, home=home) == home / ".local" / "state"


def test_project_history_path_is_stable_bounded_and_inspectable(tmp_path: Path) -> None:
    project = tmp_path / "private-parent-marker" / "A project with spaces"
    project.mkdir(parents=True)
    state_home = tmp_path / "state"

    first = project_history_path(project, state_home=state_home)
    second = project_history_path(project, state_home=state_home)

    assert first == second
    assert first.parent == state_home / "axio" / "history"
    assert first.name.startswith("A-project-with-spaces-")
    assert first.suffix == ".history"
    assert len(first.name) <= 48 + 1 + 24 + len(".history")
    assert str(project) not in first.name
    assert "private-parent-marker" not in first.name


def test_symlink_and_canonical_project_root_share_history(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    alias = tmp_path / "project-alias"
    alias.symlink_to(project, target_is_directory=True)
    state_home = tmp_path / "state"

    canonical_history = project_history_path(project, state_home=state_home)
    alias_history = project_history_path(alias, state_home=state_home)
    PrivateFileHistory(canonical_history).store_string("shared")

    assert canonical_history == alias_history
    assert list(PrivateFileHistory(alias_history).load_history_strings()) == ["shared"]


def test_project_histories_are_content_isolated(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    state_home = tmp_path / "state"
    PrivateFileHistory(project_history_path(first, state_home=state_home)).store_string("only first")

    second_history = PrivateFileHistory(project_history_path(second, state_home=state_home))

    assert list(second_history.load_history_strings()) == []


def test_distinct_projects_with_same_basename_do_not_collide(tmp_path: Path) -> None:
    first = tmp_path / "one" / "project"
    second = tmp_path / "two" / "project"
    first.mkdir(parents=True)
    second.mkdir(parents=True)
    state_home = tmp_path / "state"

    first_history = project_history_path(first, state_home=state_home)
    second_history = project_history_path(second, state_home=state_home)

    assert first_history.name.startswith("project-")
    assert second_history.name.startswith("project-")
    assert first_history != second_history


def test_root_and_non_utf8_names_have_safe_labels(tmp_path: Path) -> None:
    state_home = tmp_path / "state"
    odd_name = os.fsdecode(b"\xff\xfe")
    odd_project = tmp_path / odd_name
    odd_project.mkdir()

    assert project_history_path(Path("/"), state_home=state_home).name.startswith("root-")
    assert project_history_path(odd_project, state_home=state_home).name.startswith("project-")


def test_history_is_private_and_persists_across_instances(tmp_path: Path) -> None:
    filename = project_history_path(tmp_path / "project", state_home=tmp_path / "state")
    (tmp_path / "project").mkdir()
    history = PrivateFileHistory(filename)
    history.store_string("first")
    history.store_string("second\nline")

    reloaded = PrivateFileHistory(filename)

    assert list(reloaded.load_history_strings()) == ["second\nline", "first"]
    assert stat.S_IMODE(filename.parent.stat().st_mode) == 0o700
    assert stat.S_IMODE(filename.stat().st_mode) == 0o600


def test_history_rejects_symlinks_in_managed_storage_components(tmp_path: Path) -> None:
    state_home = tmp_path / "state"
    outside = tmp_path / "outside"
    state_home.mkdir()
    outside.mkdir()
    (state_home / "axio").symlink_to(outside, target_is_directory=True)
    project = tmp_path / "project"
    project.mkdir()
    history = PrivateFileHistory(project_history_path(project, state_home=state_home))

    with pytest.raises(OSError):
        history.store_string("must not escape")

    assert list(outside.iterdir()) == []


def test_history_open_stays_in_verified_parent_during_path_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    state_home = tmp_path / "state"
    filename = project_history_path(project, state_home=state_home)
    PrivateFileHistory(filename).store_string("initial")
    history_dir = filename.parent
    verified_dir = state_home / "verified-history"
    outside = tmp_path / "outside"
    outside.mkdir()
    original_open = os.open
    replaced = False

    def replace_before_file_open(
        file: Any,
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        nonlocal replaced
        if not replaced and file == filename.name and dir_fd is not None:
            history_dir.rename(verified_dir)
            history_dir.symlink_to(outside, target_is_directory=True)
            replaced = True
        return original_open(file, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(os, "open", replace_before_file_open)
    PrivateFileHistory(filename).store_string("after replacement")

    assert replaced
    assert "after replacement" in (verified_dir / filename.name).read_text()
    assert list(outside.iterdir()) == []


def test_concurrent_sessions_append_complete_entries(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    filename = project_history_path(project, state_home=tmp_path / "state")
    process_context = multiprocessing.get_context("fork")
    processes = [
        process_context.Process(target=_append_history, args=(str(filename), prefix, 20))
        for prefix in ("first", "second", "third")
    ]

    try:
        for process in processes:
            process.start()
        for process in processes:
            process.join(timeout=10)
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
            process.join(timeout=10)

    assert all(process.exitcode == 0 for process in processes)
    entries = list(PrivateFileHistory(filename).load_history_strings())
    assert len(entries) == 60
    assert set(entries) == {f"{prefix}-{index}" for prefix in ("first", "second", "third") for index in range(20)}
