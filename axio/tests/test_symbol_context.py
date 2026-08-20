"""Function-context coverage for compact patch diffs."""

from __future__ import annotations

import pytest

from axio.diff import describe_patch


def _replace(path: str, before: str, old: str, new: str) -> str:
    return describe_patch(path, before, before.replace(old, new))


def test_python_prefers_the_nearest_nested_function_or_method() -> None:
    nested = "def outer():\n    def inner():\n        return 1\n    return inner()\n"
    method = "class Greeter:\n    def greet(self):\n        return 'hello'\n"

    assert "@@ -1,4 +1,4 @@ outer.inner\n" in _replace("nested.py", nested, "return 1", "return 2")
    assert "@@ -1,3 +1,3 @@ Greeter.greet\n" in _replace("greeter.py", method, "return 'hello'", "return 'hi'")


def test_python_decorator_change_uses_the_decorated_function() -> None:
    before = "@cached\ndef compute():\n    return 1\n"

    assert "@@ -1,3 +1,3 @@ compute\n" in _replace("cache.py", before, "@cached", "@traced")


def test_python_async_multiline_signature_retains_class_method_context() -> None:
    before = (
        "class Service:\n"
        "    @traced\n"
        "    async def run(\n"
        "        self,\n"
        "        value: int,\n"
        "    ) -> int:\n"
        "        return value + 1\n"
    )

    result = _replace("service.py", before, "value + 1", "value + 2")

    assert "@@ -4,4 +4,4 @@ Service.run\n" in result


@pytest.mark.parametrize(
    ("path", "before", "old", "new", "symbol"),
    [
        (
            "app.js",
            "const render = (value) => {\n  return value + 1;\n};\n",
            "value + 1",
            "value + 2",
            "render",
        ),
        (
            "app.ts",
            "class View {\n  async render(value: string): Promise<string> {\n    return value;\n  }\n}\n",
            "return value",
            "return value.trim()",
            "View.render",
        ),
        (
            "service.go",
            "type Service struct {}\n\nfunc (s *Service) Run() error {\n\treturn nil\n}\n",
            "return nil",
            'return errors.New("no")',
            "Service.Run",
        ),
        (
            "lib.rs",
            "impl Worker {\n    pub fn run(&self) -> i32 {\n        1\n    }\n}\n",
            "        1",
            "        2",
            "Worker.run",
        ),
        (
            "trait_impl.rs",
            "impl Runnable for Worker {\n    fn run(&self) -> i32 {\n        1\n    }\n}\n",
            "        1",
            "        2",
            "Worker.run",
        ),
        (
            "worker.c",
            "static int run(int value) {\n    return value + 1;\n}\n",
            "value + 1",
            "value + 2",
            "run",
        ),
        (
            "Worker.java",
            "class Worker {\n    public int run(int value) {\n        return value + 1;\n    }\n}\n",
            "value + 1",
            "value + 2",
            "Worker.run",
        ),
    ],
)
def test_brace_languages_use_conservative_signatures_and_nesting(
    path: str,
    before: str,
    old: str,
    new: str,
    symbol: str,
) -> None:
    result = _replace(path, before, old, new)

    assert "@@ -" in result
    assert f"@@ {symbol}\n" in result


def test_strings_and_comments_do_not_create_or_break_brace_context() -> None:
    before = (
        "class Real {\n"
        "  run() {\n"
        '    const sample = "function fake() { }";\n'
        "    /* class Pretend { */\n"
        "    return 1;\n"
        "  }\n"
        "}\n"
    )

    result = _replace("real.js", before, "return 1", "return 2")

    assert "@@ -2,6 +2,6 @@ Real.run\n" in result
    comment_only = "// function fake() {\nconst value = 1;\n"
    assert "@@ -1,2 +1,2 @@\n" in _replace("plain.js", comment_only, "value = 1", "value = 2")


def test_non_code_and_unicode_or_control_tainted_symbols_have_no_context() -> None:
    text = "heading\nvalue one\nvalue two\n"
    assert "@@ -1,3 +1,3 @@\n" in _replace("notes.txt", text, "value one", "changed")

    unicode_name = "def привет():\n    return 1\n"
    assert "@@ -1,2 +1,2 @@\n" in _replace("unicode.py", unicode_name, "return 1", "return 2")

    tainted = "def bad\x1bname():\n    return 1\n"
    result = _replace("tainted.py", tainted, "return 1", "return 2")
    assert "\x1bname" in result
    assert "@@ -1,2 +1,2 @@\n" in result


def test_one_hunk_touching_multiple_symbols_omits_ambiguous_context() -> None:
    before = "def one():\n    return 1\n\ndef two():\n    return 2\n"
    after = before.replace("return 1", "return 10").replace("return 2", "return 20")

    result = describe_patch("pair.py", before, after)

    assert result.count("@@ -") == 1
    assert "@@ -1,5 +1,5 @@\n" in result


def test_multiple_hunks_choose_their_own_context() -> None:
    padding = "".join(f"gap_{index} = {index}\n" for index in range(12))
    before = f"def first():\n    return 1\n\n{padding}\ndef second():\n    return 2\n"
    after = before.replace("return 1", "return 10").replace("return 2", "return 20")

    result = describe_patch("pair.py", before, after)

    assert result.count("@@ -") == 2
    assert "@@ first\n" in result
    assert "@@ second\n" in result


def test_added_deleted_and_renamed_signatures_use_the_correct_snapshot() -> None:
    added = describe_patch("add.py", "value = 1\n", "value = 1\n\ndef added():\n    return 2\n")
    deleted = describe_patch("delete.py", "def removed():\n    return 1\n\nvalue = 2\n", "value = 2\n")
    renamed = describe_patch("rename.py", "def old_name():\n    return 1\n", "def new_name():\n    return 1\n")

    assert "@@ added\n" in added
    assert "@@ removed\n" in deleted
    assert "@@ new_name\n" in renamed


def test_symbol_suffix_is_bounded() -> None:
    long_name = "function_" + "x" * 300
    before = f"def {long_name}():\n    return 1\n"

    result = _replace("long.py", before, "return 1", "return 2")
    header = next(line for line in result.splitlines() if line.startswith("@@"))
    suffix = header.rsplit("@@ ", maxsplit=1)[1]

    assert len(suffix.encode("utf-8")) <= 120
