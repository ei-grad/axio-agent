"""Tests for PatchFile tool handler.

All line numbers are 1-indexed, both inclusive:
  from_line=2, to_line=4 replaces lines 2, 3, 4.
  Insert without deleting: to_line = from_line - 1.
"""

from __future__ import annotations

import json
import os
from collections.abc import Generator
from pathlib import Path
from typing import Any

import pytest
from axio.agent import Agent
from axio.context import MemoryContextStore
from axio.events import IterationEnd, ToolInputDelta, ToolResult, ToolUseStart
from axio.exceptions import HandlerError
from axio.testing import StubTransport, make_text_response
from axio.tool import Tool
from axio.types import StopReason, Usage

from axio_tools_local.patch_file import patch_file


@pytest.fixture()
def tmp_cwd(tmp_path: Path) -> Generator[Path, None, None]:
    old = os.getcwd()
    os.chdir(tmp_path)
    yield tmp_path
    os.chdir(old)


async def patch(path: Path, from_line: int, to_line: int, content: str) -> str:
    return await patch_file(path=path.name, from_line=from_line, to_line=to_line, content=content)


class TestPatchBasic:
    async def test_replace_middle_lines(self, tmp_cwd: Path) -> None:
        f = tmp_cwd / "f.txt"
        f.write_text("line1\nline2\nline3\nline4\n")
        await patch(f, 2, 3, "replaced\n")
        assert f.read_text() == "line1\nreplaced\nline4\n"

    async def test_replace_first_line(self, tmp_cwd: Path) -> None:
        f = tmp_cwd / "f.txt"
        f.write_text("line1\nline2\nline3\n")
        await patch(f, 1, 1, "new1\n")
        assert f.read_text() == "new1\nline2\nline3\n"

    async def test_replace_last_line(self, tmp_cwd: Path) -> None:
        f = tmp_cwd / "f.txt"
        f.write_text("line1\nline2\nline3\n")
        await patch(f, 3, 3, "new3\n")
        assert f.read_text() == "line1\nline2\nnew3\n"

    async def test_replace_all_lines(self, tmp_cwd: Path) -> None:
        f = tmp_cwd / "f.txt"
        f.write_text("a\nb\nc\n")
        await patch(f, 1, 3, "x\ny\n")
        assert f.read_text() == "x\ny\n"

    async def test_replace_with_more_lines(self, tmp_cwd: Path) -> None:
        """Replacing 1 line with 3 lines."""
        f = tmp_cwd / "f.txt"
        f.write_text("a\nb\nc\n")
        await patch(f, 2, 2, "x\ny\nz\n")
        assert f.read_text() == "a\nx\ny\nz\nc\n"

    async def test_replace_with_fewer_lines(self, tmp_cwd: Path) -> None:
        """Replacing 3 lines with 1 line."""
        f = tmp_cwd / "f.txt"
        f.write_text("a\nb\nc\nd\n")
        await patch(f, 1, 3, "only\n")
        assert f.read_text() == "only\nd\n"


class TestPatchInsertDelete:
    async def test_insert_without_deleting(self, tmp_cwd: Path) -> None:
        """to_line = from_line - 1 inserts before from_line without deleting."""
        f = tmp_cwd / "f.txt"
        f.write_text("a\nb\nc\n")
        await patch(f, 2, 1, "inserted\n")
        assert f.read_text() == "a\ninserted\nb\nc\n"

    async def test_insert_at_start(self, tmp_cwd: Path) -> None:
        f = tmp_cwd / "f.txt"
        f.write_text("a\nb\n")
        await patch(f, 1, 0, "before_a\n")
        assert f.read_text() == "before_a\na\nb\n"

    async def test_append_to_end(self, tmp_cwd: Path) -> None:
        """Insert after last line: from_line = N+1, to_line = N."""
        f = tmp_cwd / "f.txt"
        f.write_text("a\nb\n")
        await patch(f, 3, 2, "appended\n")
        assert f.read_text() == "a\nb\nappended\n"

    async def test_delete_lines_empty_content(self, tmp_cwd: Path) -> None:
        """Empty content string deletes the specified lines."""
        f = tmp_cwd / "f.txt"
        f.write_text("a\nb\nc\n")
        await patch(f, 2, 2, "")
        assert f.read_text() == "a\nc\n"

    async def test_delete_multiple_lines(self, tmp_cwd: Path) -> None:
        f = tmp_cwd / "f.txt"
        f.write_text("a\nb\nc\nd\n")
        await patch(f, 2, 3, "")
        assert f.read_text() == "a\nd\n"


class TestFramedContent:
    async def test_replaces_indented_line_from_provider_safe_framed_content(self, tmp_cwd: Path) -> None:
        f = tmp_cwd / "index.html"
        f.write_text("before\n          old();\nafter\n")

        result = await patch_file(path=f.name, from_line=2, to_line=2, content="│          replacement();")

        assert f.read_text() == "before\n          replacement();\nafter\n"
        assert "-          old();\n" in result
        assert "+          replacement();\n" in result

    async def test_preserves_multiline_whitespace_empty_line_and_trailing_newline(self, tmp_cwd: Path) -> None:
        f = tmp_cwd / "f.txt"
        f.write_text("before\nold one\nold two\nafter\n")

        await patch_file(
            path=f.name,
            from_line=2,
            to_line=3,
            content="│      first\n│\tsecond\n│\n",
        )

        assert f.read_bytes() == b"before\n      first\n\tsecond\n\nafter\n"

    async def test_preserves_crlf_from_framed_content_and_untouched_lines(self, tmp_cwd: Path) -> None:
        f = tmp_cwd / "f.txt"
        f.write_bytes(b"before\r\nold\r\nafter\r\n")

        await patch_file(path=f.name, from_line=2, to_line=2, content="│    replacement\r\n")

        assert f.read_bytes() == b"before\r\n    replacement\r\nafter\r\n"

    async def test_framed_empty_line_is_not_a_deletion(self, tmp_cwd: Path) -> None:
        f = tmp_cwd / "f.txt"
        f.write_text("before\nold\nafter\n")

        await patch_file(path=f.name, from_line=2, to_line=2, content="│")

        assert f.read_text() == "before\n\nafter\n"

    @pytest.mark.parametrize(
        ("before", "from_line", "to_line"),
        [("old\n", 1, 1), ("", 1, 0)],
    )
    async def test_framed_empty_line_survives_at_eof(
        self,
        tmp_cwd: Path,
        before: str,
        from_line: int,
        to_line: int,
    ) -> None:
        f = tmp_cwd / "f.txt"
        f.write_text(before)

        await patch_file(path=f.name, from_line=from_line, to_line=to_line, content="│")

        assert f.read_bytes() == b"\n"

    async def test_framed_insert_and_eof_without_newline(self, tmp_cwd: Path) -> None:
        f = tmp_cwd / "f.txt"
        f.write_text("after\n")

        await patch_file(path=f.name, from_line=1, to_line=0, content="│   inserted")
        await patch_file(path=f.name, from_line=2, to_line=2, content="│last")

        assert f.read_bytes() == b"   inserted\nlast"

    async def test_double_sentinel_produces_literal_source_sentinel(self, tmp_cwd: Path) -> None:
        f = tmp_cwd / "f.txt"
        f.write_text("old\n")

        await patch_file(path=f.name, from_line=1, to_line=1, content="││literal\n")

        assert f.read_text() == "│literal\n"

    async def test_legacy_unframed_content_remains_byte_literal(self, tmp_cwd: Path) -> None:
        f = tmp_cwd / "f.txt"
        f.write_text("old\n")
        content = " \tfirst\n\tsecond\n"

        await patch_file(path=f.name, from_line=1, to_line=1, content=content)

        assert f.read_bytes() == content.encode()

    @pytest.mark.parametrize("content", ["│framed\nlegacy", "legacy\n│framed"])
    async def test_rejects_mixed_framing_without_writing(self, tmp_cwd: Path, content: str) -> None:
        f = tmp_cwd / "f.txt"
        original = "old\n"
        f.write_text(original)

        with pytest.raises(HandlerError, match="every content line"):
            await patch_file(path=f.name, from_line=1, to_line=1, content=content)

        assert f.read_text() == original

    async def test_rejects_read_file_metadata_without_writing(self, tmp_cwd: Path) -> None:
        f = tmp_cwd / "f.txt"
        original = "old\n"
        f.write_text(original)

        with pytest.raises(HandlerError, match=r"remove the L<number> prefix"):
            await patch_file(path=f.name, from_line=1, to_line=1, content="L489│          replacement")

        assert f.read_text() == original

    async def test_schema_requires_visible_framing_and_has_no_indent_workaround(self) -> None:
        tool: Tool[Any] = Tool(name="patch_file", handler=patch_file)

        assert "first_line_indent" not in tool.schema["properties"]
        content = tool.schema["properties"]["content"]
        assert "every logical line" in content["description"]
        assert "│" in content["description"]
        assert "L<number>" in content["description"]

    async def test_fragmented_tool_input_repairs_counter_and_keeps_raw_input_in_result(self, tmp_cwd: Path) -> None:
        f = tmp_cwd / "index.html"
        f.write_text("$counter.textContent =\n")
        arguments = {
            "path": f.name,
            "from_line": 1,
            "to_line": 1,
            "content": "│          $counter.textContent =",
        }
        raw = json.dumps(arguments, ensure_ascii=False)
        sentinel_at = raw.index("│")
        transport = StubTransport(
            [
                [
                    ToolUseStart(0, "patch-1", "patch_file"),
                    ToolInputDelta(0, "patch-1", raw[:sentinel_at]),
                    ToolInputDelta(0, "patch-1", raw[sentinel_at : sentinel_at + 4]),
                    ToolInputDelta(0, "patch-1", raw[sentinel_at + 4 :]),
                    IterationEnd(1, StopReason.tool_use, Usage(10, 5)),
                ],
                make_text_response("Done"),
            ]
        )
        agent = Agent(system="test", tools=[Tool(name="patch_file", handler=patch_file)], transport=transport)

        events = [event async for event in agent.run_stream("repair", MemoryContextStore())]

        result = next(event for event in events if isinstance(event, ToolResult))
        assert result.input == arguments
        assert not result.is_error
        assert result.content != "No changes"
        assert f.read_text() == "          $counter.textContent ="


class TestNewlineHandling:
    async def test_content_without_trailing_newline(self, tmp_cwd: Path) -> None:
        """Content without \\n must not corrupt next line."""
        f = tmp_cwd / "f.txt"
        f.write_text("a\nb\nc\n")
        await patch(f, 2, 2, "replaced")  # no trailing newline
        assert f.read_text() == "a\nreplaced\nc\n"

    async def test_multiline_content_no_trailing_newline(self, tmp_cwd: Path) -> None:
        f = tmp_cwd / "f.txt"
        f.write_text("a\nb\nc\n")
        await patch(f, 2, 2, "x\ny")  # no trailing newline on last line
        assert f.read_text() == "a\nx\ny\nc\n"

    async def test_adjacent_lines_not_corrupted(self, tmp_cwd: Path) -> None:
        """Lines before and after patch range are exactly preserved."""
        f = tmp_cwd / "f.txt"
        f.write_text("first\nsecond\nthird\nfourth\nfifth\n")
        await patch(f, 3, 3, "REPLACED\n")
        lines = f.read_text().splitlines()
        assert lines == ["first", "second", "REPLACED", "fourth", "fifth"]

    async def test_content_extra_trailing_newlines(self, tmp_cwd: Path) -> None:
        """Content with extra trailing newlines creates blank lines."""
        f = tmp_cwd / "f.txt"
        f.write_text("a\nb\nc\n")
        await patch(f, 2, 2, "x\n\n")
        assert f.read_text() == "a\nx\n\nc\n"

    async def test_single_line_file_replace(self, tmp_cwd: Path) -> None:
        f = tmp_cwd / "f.txt"
        f.write_text("only line\n")
        await patch(f, 1, 1, "new line\n")
        assert f.read_text() == "new line\n"

    async def test_single_line_file_no_trailing_newline(self, tmp_cwd: Path) -> None:
        f = tmp_cwd / "f.txt"
        f.write_text("only line")
        await patch(f, 1, 1, "replaced\n")
        assert f.read_text() == "replaced\n"

    async def test_file_no_trailing_newline_patch_middle(self, tmp_cwd: Path) -> None:
        f = tmp_cwd / "f.txt"
        f.write_text("a\nb\nc")  # c has no trailing newline
        await patch(f, 2, 2, "B\n")
        assert f.read_text() == "a\nB\nc"

    @pytest.mark.parametrize(
        ("before", "from_line", "to_line", "content", "after", "expected_result"),
        [
            (
                "same",
                1,
                1,
                "same\n",
                "same\n",
                "+1 -1\n@@ -1 +1 @@\n-same\n\\ No newline at end of file\n+same\n",
            ),
            (
                "same\n",
                1,
                1,
                "same",
                "same",
                "+1 -1\n@@ -1 +1 @@\n-same\n+same\n\\ No newline at end of file\n",
            ),
            (
                "one\ntwo",
                2,
                2,
                "two\n",
                "one\ntwo\n",
                "+1 -1\n@@ -1,2 +1,2 @@\n one\n-two\n\\ No newline at end of file\n+two\n",
            ),
            (
                "one\ntwo\n",
                2,
                2,
                "two",
                "one\ntwo",
                "+1 -1\n@@ -1,2 +1,2 @@\n one\n-two\n+two\n\\ No newline at end of file\n",
            ),
            ("", 1, 0, "\n", "\n", "+1 -0\n@@ -0,0 +1 @@\n+\n"),
            ("\n", 1, 1, "", "", "+0 -1\n@@ -1 +0,0 @@\n-\n"),
        ],
    )
    async def test_final_newline_changes_are_exact(
        self,
        tmp_cwd: Path,
        before: str,
        from_line: int,
        to_line: int,
        content: str,
        after: str,
        expected_result: str,
    ) -> None:
        f = tmp_cwd / "f.txt"
        f.write_text(before)

        result = await patch(f, from_line, to_line, content)

        assert f.read_text() == after
        assert result == expected_result


class TestEdgeCases:
    async def test_file_not_found(self, tmp_cwd: Path) -> None:
        with pytest.raises(HandlerError, match="missing.txt"):
            await patch(tmp_cwd / "missing.txt", 1, 1, "x")

    async def test_wrong_type_raises_with_distinct_message(self, tmp_cwd: Path) -> None:
        """A path that exists but is a directory gets a different message than a missing path."""
        d = tmp_cwd / "adir"
        d.mkdir()
        with pytest.raises(HandlerError, match="Not a file"):
            await patch(d, 1, 1, "x")

    async def test_non_utf8_file_raises_handler_error(self, tmp_cwd: Path) -> None:
        f = tmp_cwd / "b.dat"
        f.write_bytes(b"\x80\x81\xff")
        with pytest.raises(HandlerError, match="not valid UTF-8"):
            await patch(f, 1, 1, "x")

    async def test_permission_denied_raises_handler_error(self, tmp_cwd: Path) -> None:
        if os.geteuid() == 0:
            pytest.skip("root ignores file permission bits")
        f = tmp_cwd / "locked.txt"
        f.write_text("a\nb\n")
        f.chmod(0o000)
        try:
            with pytest.raises(HandlerError, match="locked.txt"):
                await patch(f, 1, 1, "x")
        finally:
            f.chmod(0o644)

    async def test_reports_compact_path_free_diff_of_edit(self, tmp_cwd: Path) -> None:
        """The result keeps exact changes without repeating the tool input path."""
        f = tmp_cwd / "f.txt"
        f.write_text("line1\nline2\nline3\n")
        result = await patch(f, 2, 2, "CHANGED")
        assert result.startswith("+1 -1\n@@ -1,3 +1,3 @@\n")
        assert "f.txt" not in result
        assert "Wrote" not in result
        assert "---" not in result
        assert "+++" not in result
        assert "-line2\n" in result
        assert "+CHANGED\n" in result
        assert " line1\n" in result
        assert " line3\n" in result

    async def test_python_patch_reports_function_context(self, tmp_cwd: Path) -> None:
        f = tmp_cwd / "service.py"
        f.write_text("class Service:\n    def run(self):\n        return 1\n")

        result = await patch(f, 3, 3, "        return 2\n")

        assert "@@ -1,3 +1,3 @@ Service.run\n" in result

    async def test_empty_file_append(self, tmp_cwd: Path) -> None:
        f = tmp_cwd / "f.txt"
        f.write_text("")
        await patch(f, 1, 0, "new\n")
        assert f.read_text() == "new\n"

    async def test_indented_code_preserved(self, tmp_cwd: Path) -> None:
        f = tmp_cwd / "f.py"
        f.write_text("def foo():\n    return 1\n\ndef bar():\n    return 2\n")
        await patch(f, 2, 2, "    return 42\n")
        assert f.read_text() == "def foo():\n    return 42\n\ndef bar():\n    return 2\n"

    async def test_unicode_content(self, tmp_cwd: Path) -> None:
        f = tmp_cwd / "f.txt"
        f.write_text("привет\nмир\n")
        await patch(f, 1, 1, "hello\n")
        assert f.read_text() == "hello\nмир\n"
