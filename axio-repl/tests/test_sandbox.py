from contextlib import AsyncExitStack
from pathlib import Path
from typing import Any

import pytest
from axio.tool import Tool

from axio_repl import _sandbox, _search


def test_ast_grep_argv_never_invokes_sg() -> None:
    # `sg` is shadow-utils' setgid helper on Linux; invoking it instead of
    # ast-grep would run an unrelated program.
    argv = _sandbox._ast_grep_argv("$A == $A", ".", "python")
    assert argv[0] == "ast-grep"
    assert "--lang" in argv and "python" in argv


def test_ast_grep_argv_omits_lang_when_unset() -> None:
    assert "--lang" not in _sandbox._ast_grep_argv("$A", "src", None)


def test_truncate_reports_the_cut() -> None:
    assert _sandbox._truncate("", 10) == "No matches"
    assert _sandbox._truncate("a\nb", 10) == "a\nb"
    assert _sandbox._truncate("a\nb\nc", 2).endswith("[truncated at 2 lines]")


async def _noop() -> str:
    return ""


@pytest.mark.asyncio
async def test_host_mode_drops_sandbox_only_tools(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_sandbox, "ast_grep_available", lambda: False)
    tools: list[Tool[Any]] = [Tool(name="shell", handler=_noop), Tool(name="run_python", handler=_noop)]
    async with AsyncExitStack() as stack:
        result, desc, tool_root, note = await _sandbox.build_tools(stack, tools, "none", "img", Path("/tmp"))
    assert [t.name for t in result] == ["shell"]
    assert desc.startswith("host")
    # On the host the tools see the same path the user does.
    assert tool_root == Path("/tmp")
    # Nothing to say about an environment the user already knows.
    assert note == ""


@pytest.mark.asyncio
async def test_host_mode_adds_ast_grep_when_present(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_sandbox, "ast_grep_available", lambda: True)
    async with AsyncExitStack() as stack:
        result, _, _root, _note = await _sandbox.build_tools(stack, [], "none", "img", Path("/tmp"))
    assert [t.name for t in result] == ["ast_grep"]


class _StubSandbox:
    """Answers the probe with a fixed set of installed commands."""

    def __init__(self, present: list[str] | None = None, fail: bool = False) -> None:
        self.present = present or []
        self.fail = fail

    async def exec(self, command: str, timeout: float = 30) -> str:
        if self.fail:
            raise RuntimeError("container is gone")
        return "\n".join(self.present)


@pytest.mark.asyncio
async def test_environment_note_splits_present_from_missing() -> None:
    note = await _sandbox.describe_environment(_StubSandbox(["python3", "grep"]), "img", networking=False)
    assert "Available: python3, grep" in note
    assert "git" in note.split("Not installed:")[1]
    assert "python3" not in note.split("Not installed:")[1]


@pytest.mark.asyncio
async def test_environment_note_warns_that_nothing_can_be_installed() -> None:
    offline = await _sandbox.describe_environment(_StubSandbox(["python3"]), "img", networking=False)
    assert "pip install" in offline
    online = await _sandbox.describe_environment(_StubSandbox(["python3"]), "img", networking=True)
    assert "pip install" not in online


@pytest.mark.asyncio
async def test_environment_note_calls_out_the_unusable_ast_grep_tool() -> None:
    # The tool is registered unconditionally in a sandbox, so a model that is not
    # told will keep calling something that cannot work.
    without = await _sandbox.describe_environment(_StubSandbox(["python3"]), "img", networking=False)
    assert "ast_grep" in without
    with_it = await _sandbox.describe_environment(_StubSandbox(["python3", "ast-grep"]), "img", networking=False)
    assert "ast_grep" not in with_it


@pytest.mark.asyncio
async def test_environment_note_is_empty_when_the_probe_says_nothing() -> None:
    # A sandbox that cannot answer must not put guesses in the system prompt.
    assert await _sandbox.describe_environment(_StubSandbox(fail=True), "img", networking=False) == ""
    assert await _sandbox.describe_environment(_StubSandbox([]), "img", networking=False) == ""


def test_search_module_is_self_contained() -> None:
    # The sandbox ships this file's source into a container that has nothing but
    # the standard library, so it must not import from the rest of axio-repl.
    source = Path(_search.__file__).read_text(encoding="utf-8")
    imports = [ln for ln in source.splitlines() if ln.startswith(("import ", "from "))]
    assert imports and all("axio" not in ln for ln in imports)
