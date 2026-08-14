from contextlib import AsyncExitStack
from pathlib import Path
from typing import Any, Self

import pytest
from axio.tool import Tool
from axio_tools_docker import sandbox as docker_sandbox

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


def test_sandbox_options_reject_builtin_routed_networks() -> None:
    with pytest.raises(ValueError, match="user-defined internal"):
        _sandbox.SandboxOptions(network="bridge")


def test_sandbox_options_reject_empty_network_name() -> None:
    with pytest.raises(ValueError, match="non-empty Docker network"):
        _sandbox.SandboxOptions(network="")


@pytest.mark.parametrize(
    ("field", "value"),
    [("memory", "lots"), ("memory", "0"), ("cpus", "fast"), ("cpus", "0"), ("cpus", "nan")],
)
def test_sandbox_options_reject_invalid_resource_limits(field: str, value: str) -> None:
    kwargs: dict[str, Any] = {field: value}
    with pytest.raises(ValueError, match=f"--sandbox-{field}"):
        _sandbox.SandboxOptions(**kwargs)


@pytest.mark.parametrize(
    ("memory", "expected"),
    [("256m", 256 * 1024**2), ("1g", 1024**3), ("512k", 512 * 1024), ("1048576", 1048576)],
)
def test_sandbox_memory_matches_docker_parser(memory: str, expected: int) -> None:
    options = _sandbox.SandboxOptions(memory=memory)

    assert docker_sandbox.parse_memory(options.memory) == expected


def test_sandbox_options_require_network_for_endpoints() -> None:
    with pytest.raises(ValueError, match="require --sandbox-network"):
        _sandbox.SandboxOptions(pypi_index="http://nexus/pypi/simple")


@pytest.mark.parametrize(
    "index",
    [
        "nexus/pypi/simple",
        "ftp://nexus/pypi/simple",
        "http://user:password@nexus/pypi/simple",
        "http://nexus:invalid/pypi/simple",
        "http://nexus/pypi/simple?token=value",
        "http://nexus/pypi/simple#fragment",
        "http://bad host/pypi/simple",
    ],
)
def test_sandbox_options_reject_invalid_pypi_index(index: str) -> None:
    with pytest.raises(ValueError, match="--sandbox-pypi-index"):
        _sandbox.SandboxOptions(network="agent-egress", pypi_index=index)


def test_https_pypi_index_does_not_disable_certificate_checks() -> None:
    options = _sandbox.SandboxOptions(network="agent-egress", pypi_index="https://nexus/pypi/simple")

    assert "PIP_TRUSTED_HOST" not in options.environment()
    assert "UV_INSECURE_HOST" not in options.environment()


@pytest.mark.parametrize(
    "index",
    ["sparse+http://user:password@nexus/cargo/", "sparse+ftp://nexus/cargo/", "sparse+http://nexus/cargo"],
)
def test_sandbox_options_reject_invalid_cargo_index(index: str) -> None:
    with pytest.raises(ValueError, match="--sandbox-cargo-index"):
        _sandbox.SandboxOptions(network="agent-egress", cargo_index=index)


def test_sandbox_options_build_client_environment(tmp_path: Path) -> None:
    ca_certificate = tmp_path / "egress-ca.pem"
    ca_certificate.write_text("certificate", encoding="utf-8")
    options = _sandbox.SandboxOptions(
        network="agent-egress",
        proxy="http://mitmania:8080",
        no_proxy="nexus,datasets",
        pypi_index="http://nexus:8081/pypi/simple",
        npm_registry="http://nexus/npm/",
        cargo_index="sparse+http://nexus/cargo/",
        go_proxy="http://nexus/go",
        go_sumdb="sum.golang.org https://nexus/sumdb/sum.golang.org",
        ca_certificate=ca_certificate,
    )

    assert options.environment() == {
        "HTTP_PROXY": "http://mitmania:8080",
        "HTTPS_PROXY": "http://mitmania:8080",
        "http_proxy": "http://mitmania:8080",
        "https_proxy": "http://mitmania:8080",
        "NO_PROXY": "nexus,datasets",
        "no_proxy": "nexus,datasets",
        "UV_DEFAULT_INDEX": "http://nexus:8081/pypi/simple",
        "UV_INSECURE_HOST": "nexus:8081",
        "PIP_INDEX_URL": "http://nexus:8081/pypi/simple",
        "PIP_TRUSTED_HOST": "nexus:8081",
        "NPM_CONFIG_REGISTRY": "http://nexus/npm/",
        "CARGO_HOME": _sandbox.CARGO_HOME,
        "GOPROXY": "http://nexus/go",
        "GOSUMDB": "sum.golang.org https://nexus/sumdb/sum.golang.org",
        "SSL_CERT_FILE": _sandbox.EGRESS_CA_PATH,
        "REQUESTS_CA_BUNDLE": _sandbox.EGRESS_CA_PATH,
        "CURL_CA_BUNDLE": _sandbox.EGRESS_CA_PATH,
        "GIT_SSL_CAINFO": _sandbox.EGRESS_CA_PATH,
        "NODE_EXTRA_CA_CERTS": _sandbox.EGRESS_CA_PATH,
        "CARGO_HTTP_CAINFO": _sandbox.EGRESS_CA_PATH,
    }
    assert options.read_only_volumes()[_sandbox.EGRESS_CA_PATH] == str(ca_certificate)
    assert options.cargo_config() == (
        '[source.crates-io]\nreplace-with = "axio-mirror"\n\n'
        '[source.axio-mirror]\nregistry = "sparse+http://nexus/cargo/"\n'
    )


@pytest.mark.asyncio
async def test_restricted_options_do_not_fall_back_to_host(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_sandbox, "docker_available", lambda: False)
    options = _sandbox.SandboxOptions(network="agent-egress")

    async with AsyncExitStack() as stack:
        with pytest.raises(RuntimeError, match="require Docker"):
            await _sandbox.build_tools(stack, [], "auto", "img", Path("/tmp"), options)


@pytest.mark.asyncio
async def test_explicit_host_mode_rejects_nondefault_resources() -> None:
    options = _sandbox.SandboxOptions(memory="1g")

    async with AsyncExitStack() as stack:
        with pytest.raises(RuntimeError, match="require Docker"):
            await _sandbox.build_tools(stack, [], "none", "img", Path("/tmp"), options)


@pytest.mark.asyncio
async def test_docker_mode_passes_restricted_configuration(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    datasets = tmp_path / "datasets"
    datasets.mkdir()
    captured: dict[str, Any] = {}

    class FakeDockerSandbox:
        def __init__(self, **kwargs: Any) -> None:
            captured.update(kwargs)
            self.tools: list[Tool[Any]] = []

        async def __aenter__(self) -> Self:
            return self

        async def __aexit__(self, *exc: object) -> None:
            return None

        async def exec(self, command: str, timeout: float = 30) -> str:
            return ""

    monkeypatch.setattr(docker_sandbox, "DockerSandbox", FakeDockerSandbox)
    options = _sandbox.SandboxOptions(
        network="agent-egress",
        memory="4g",
        cpus="2.0",
        proxy="http://mitmania:8080",
        cargo_index="sparse+http://nexus/cargo/",
        datasets=datasets,
    )

    async with AsyncExitStack() as stack:
        _, description, _, _ = await _sandbox.build_tools(stack, [], "docker", "img", tmp_path, options)

        read_only_volumes = captured["read_only_volumes"]
        cargo_config_path = Path(read_only_volumes[_sandbox.CARGO_CONFIG_PATH])
        assert cargo_config_path.parent != tmp_path
        assert cargo_config_path.read_text(encoding="utf-8") == options.cargo_config()

    assert captured["network"] == "agent-egress"
    assert captured["require_internal_network"] is True
    assert captured["memory"] == "4g"
    assert captured["cpus"] == "2.0"
    assert captured["env"]["HTTPS_PROXY"] == "http://mitmania:8080"
    assert read_only_volumes[_sandbox.DATASETS_DIR] == str(datasets)
    assert "internal network agent-egress" in description


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
    assert "uv add/sync" in offline
    online = await _sandbox.describe_environment(_StubSandbox(["python3"]), "img", networking=True)
    assert "uv add/sync" not in online


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
