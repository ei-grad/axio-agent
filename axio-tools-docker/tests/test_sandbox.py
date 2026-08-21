"""Tests for DockerSandbox."""

from __future__ import annotations

import asyncio
import io
import tarfile
from collections.abc import AsyncGenerator
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock, patch

import aiodocker
import pytest
from axio.exceptions import HandlerError
from axio.tool import CONTEXT

from axio_tools_docker import sandbox as sandbox_module
from axio_tools_docker.sandbox import DockerSandbox, ImageNotAvailableError, parse_cpus, parse_device, parse_memory

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_tar_bytes(filename: str, content: bytes) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:") as tar:
        info = tarfile.TarInfo(name=filename)
        info.size = len(content)
        tar.addfile(info, io.BytesIO(content))
    return buf.getvalue()


def make_tar_file(filename: str, content: bytes) -> tarfile.TarFile:
    return tarfile.open(fileobj=io.BytesIO(make_tar_bytes(filename, content)))


def make_listing_output(*entries: tuple[str, int, int, str]) -> bytes:
    fields = ["AXIO_LIST_V3"]
    if entries:
        fields.extend(("AXIO_LIST_BATCH", str(len(entries)), *(name for _, _, _, name in entries)))
    metadata = "".join(f"{mode} {size} {mtime}\n" for mode, size, mtime, _ in entries)
    return (("\0".join(fields) + "\0") + metadata).encode()


def mock_docker_factory(
    exec_messages: list[tuple[int, bytes]] | None = None,
    exec_exit_code: int = 0,
    archive_content: tarfile.TarFile | None = None,
    shell_paths: tuple[str, ...] = ("/bin/sh",),
    login_profile_noise: bytes | None = None,
) -> tuple[MagicMock, MagicMock, MagicMock]:
    """Return (mock_docker_class, mock_client, mock_container)."""
    if exec_messages is None:
        exec_messages = [(1, b"hello\n")]

    messages = list(exec_messages)

    async def read_out() -> Any:
        if messages:
            stream_type, data = messages.pop(0)
            msg = MagicMock()
            msg.stream = stream_type
            msg.data = data
            return msg
        return None

    mock_stream = MagicMock()
    mock_stream.read_out = read_out
    mock_stream.close = AsyncMock()

    mock_exec = MagicMock()
    mock_exec.start = MagicMock(return_value=mock_stream)
    mock_exec.inspect = AsyncMock(return_value={"ExitCode": exec_exit_code})

    mock_container = MagicMock()
    mock_container.start = AsyncMock()
    mock_container.delete = AsyncMock()
    mock_container.command_exec = mock_exec

    async def exec_command(*, cmd: list[str], **_: Any) -> MagicMock:
        if cmd[1:] == ["-c", "exit 0"]:
            probe_stream = MagicMock()
            probe_stream.read_out = AsyncMock(return_value=None)
            probe_stream.close = AsyncMock()
            probe_exec = MagicMock()
            probe_exec.start = MagicMock(return_value=probe_stream)
            probe_exec.inspect = AsyncMock(return_value={"ExitCode": 0 if cmd[0] in shell_paths else 127})
            return probe_exec
        if login_profile_noise is not None and "l" in cmd[1]:
            noise_messages = [(2, login_profile_noise)]

            async def read_noise() -> Any:
                if noise_messages:
                    stream_type, data = noise_messages.pop(0)
                    msg = MagicMock()
                    msg.stream = stream_type
                    msg.data = data
                    return msg
                return None

            noise_stream = MagicMock()
            noise_stream.read_out = read_noise
            noise_stream.close = AsyncMock()
            noise_exec = MagicMock()
            noise_exec.start = MagicMock(return_value=noise_stream)
            noise_exec.inspect = AsyncMock(return_value={"ExitCode": 0})
            return noise_exec
        return cast(MagicMock, mock_container.command_exec)

    mock_container.exec = AsyncMock(side_effect=exec_command, return_value=mock_exec)
    mock_container.show = AsyncMock(
        return_value={
            "State": {"Running": True},
            "Config": {"Env": [f"PATH={sandbox_module._DEFAULT_CONTAINER_PATH}"]},
        }
    )
    mock_container.put_archive = AsyncMock()
    mock_container.get_archive = AsyncMock(
        return_value=archive_content if archive_content is not None else make_tar_file("file.txt", b"content")
    )

    mock_containers = MagicMock()
    captured_config: list[dict[str, Any]] = []

    async def create_container(config: dict[str, Any], **_: Any) -> MagicMock:
        captured_config.append(config)
        return mock_container

    mock_containers.create = create_container
    # Default: no named container exists - callers can override per test.
    mock_containers.get = AsyncMock(side_effect=aiodocker.exceptions.DockerError(404, "Not found"))

    mock_images = MagicMock()
    mock_images.inspect = AsyncMock()  # image present by default - no pull needed
    mock_images.pull = AsyncMock()

    mock_system = MagicMock()
    mock_system.info = AsyncMock(return_value={})  # daemon available by default

    mock_network = MagicMock()
    mock_network.show = AsyncMock(return_value={"Internal": True, "Id": "verified-network-id"})
    mock_networks = MagicMock()
    mock_networks.get = AsyncMock(return_value=mock_network)

    mock_client = MagicMock()
    mock_client.containers = mock_containers
    mock_client.images = mock_images
    mock_client.system = mock_system
    mock_client.networks = mock_networks
    mock_client.close = AsyncMock()
    mock_client._captured_config = captured_config

    mock_docker_class = MagicMock(return_value=mock_client)

    return mock_docker_class, mock_client, mock_container


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------


async def test_daemon_unavailable_raises() -> None:
    cls, client, container = mock_docker_factory()
    client.system.info = AsyncMock(side_effect=OSError("connection refused"))
    with patch("axio_tools_docker.sandbox.aiodocker.Docker", cls):
        with pytest.raises(RuntimeError, match="Docker daemon not available"):
            async with DockerSandbox():
                pass


async def test_daemon_unavailable_closes_client() -> None:
    cls, client, container = mock_docker_factory()
    client.system.info = AsyncMock(side_effect=OSError("connection refused"))
    with patch("axio_tools_docker.sandbox.aiodocker.Docker", cls):
        with pytest.raises(RuntimeError):
            async with DockerSandbox():
                pass
    client.close.assert_awaited_once()


async def test_context_manager_creates_and_starts() -> None:
    cls, client, container = mock_docker_factory()
    with patch("axio_tools_docker.sandbox.aiodocker.Docker", cls):
        async with DockerSandbox(image="alpine:latest"):
            pass
    container.start.assert_awaited_once()


async def test_missing_local_only_image_is_not_pulled() -> None:
    cls, client, container = mock_docker_factory()
    client.images.inspect = AsyncMock(side_effect=aiodocker.exceptions.DockerError(404, "Not found"))
    with patch("axio_tools_docker.sandbox.aiodocker.Docker", cls):
        with pytest.raises(ImageNotAvailableError, match="not available locally"):
            async with DockerSandbox(image="local-only:latest", pull_missing=False):
                pass
    client.images.pull.assert_not_awaited()
    client.close.assert_awaited_once()


async def test_image_inspection_error_is_not_reported_as_missing() -> None:
    cls, client, container = mock_docker_factory()
    client.images.inspect = AsyncMock(side_effect=aiodocker.exceptions.DockerError(500, "daemon error"))
    with patch("axio_tools_docker.sandbox.aiodocker.Docker", cls):
        with pytest.raises(aiodocker.exceptions.DockerError, match="daemon error"):
            async with DockerSandbox(image="local-only:latest", pull_missing=False):
                pass
    client.images.pull.assert_not_awaited()
    client.close.assert_awaited_once()


async def test_missing_image_is_pulled_by_default() -> None:
    cls, client, container = mock_docker_factory()
    client.images.inspect = AsyncMock(side_effect=aiodocker.exceptions.DockerError(404, "Not found"))
    with patch("axio_tools_docker.sandbox.aiodocker.Docker", cls):
        async with DockerSandbox(image="public:latest"):
            pass
    client.images.pull.assert_awaited_once_with("public:latest")


async def test_context_manager_deletes_on_exit() -> None:
    cls, client, container = mock_docker_factory()
    with patch("axio_tools_docker.sandbox.aiodocker.Docker", cls):
        async with DockerSandbox():
            pass
    container.delete.assert_awaited_once_with(force=True)


async def test_context_manager_deletes_on_error() -> None:
    cls, client, container = mock_docker_factory()
    with patch("axio_tools_docker.sandbox.aiodocker.Docker", cls):
        with pytest.raises(RuntimeError):
            async with DockerSandbox():
                raise RuntimeError("boom")
    container.delete.assert_awaited_once_with(force=True)


async def test_client_closed_on_exit() -> None:
    cls, client, container = mock_docker_factory()
    with patch("axio_tools_docker.sandbox.aiodocker.Docker", cls):
        async with DockerSandbox():
            pass
    client.close.assert_awaited_once()


async def test_named_existing_container_attaches() -> None:
    """name= reuses an existing container - no create, no start."""
    cls, client, container = mock_docker_factory()
    client.containers.get = AsyncMock(return_value=container)
    with patch("axio_tools_docker.sandbox.aiodocker.Docker", cls):
        async with DockerSandbox(name="my-sandbox"):
            pass
    client.containers.get.assert_awaited_once_with("my-sandbox")
    container.start.assert_not_awaited()


async def test_named_existing_stopped_container_starts_before_attach() -> None:
    cls, client, container = mock_docker_factory()
    client.containers.get = AsyncMock(return_value=container)
    container.show = AsyncMock(return_value={"State": {"Running": False}})
    with patch("axio_tools_docker.sandbox.aiodocker.Docker", cls):
        async with DockerSandbox(name="my-sandbox"):
            pass
    container.start.assert_awaited_once()


async def test_named_existing_container_not_deleted() -> None:
    """Attached container is never removed even with remove=True."""
    cls, client, container = mock_docker_factory()
    client.containers.get = AsyncMock(return_value=container)
    with patch("axio_tools_docker.sandbox.aiodocker.Docker", cls):
        async with DockerSandbox(name="my-sandbox", remove=True):
            pass
    container.delete.assert_not_awaited()


async def test_named_missing_container_creates_new() -> None:
    """If no container with the name exists, a new one is created."""
    cls, client, container = mock_docker_factory()
    # mock_docker_factory already sets get to raise DockerError by default
    with patch("axio_tools_docker.sandbox.aiodocker.Docker", cls):
        async with DockerSandbox(name="new-sandbox"):
            pass
    container.start.assert_awaited_once()


# ---------------------------------------------------------------------------
# tools property
# ---------------------------------------------------------------------------

EXPECTED_TOOL_NAMES = {"shell", "write_file", "read_file", "list_files", "run_python", "patch_file"}


async def test_tools_returns_six_tools() -> None:
    cls, client, container = mock_docker_factory()
    with patch("axio_tools_docker.sandbox.aiodocker.Docker", cls):
        async with DockerSandbox() as sb:
            tools = sb.tools
    assert len(tools) == 6
    assert {t.name for t in tools} == EXPECTED_TOOL_NAMES


async def test_tools_names_match_axio_tools_local() -> None:
    """Tool names must be identical to axio-tools-local for drop-in use."""
    cls, client, container = mock_docker_factory()
    with patch("axio_tools_docker.sandbox.aiodocker.Docker", cls):
        async with DockerSandbox() as sb:
            names = {t.name for t in sb.tools}
    assert names == EXPECTED_TOOL_NAMES


async def test_patch_file_tool_schema_exposes_bounded_optional_indent() -> None:
    cls, client, container = mock_docker_factory()
    with patch("axio_tools_docker.sandbox.aiodocker.Docker", cls):
        async with DockerSandbox() as sb:
            tool = next(item for item in sb.tools if item.name == "patch_file")

    prop = tool.schema["properties"]["first_line_indent"]
    assert prop["type"] == "integer"
    assert prop["minimum"] == 0
    assert prop["maximum"] == 256
    assert prop["default"] == 0
    assert "never inferred" in prop["description"]
    assert "first_line_indent" not in tool.schema["required"]


async def test_shell_discovery_prefers_bash_and_builds_runtime_schema() -> None:
    cls, client, container = mock_docker_factory(
        shell_paths=("/usr/bin/bash", "/bin/sh", "/usr/local/bin/zsh", "/usr/bin/dash")
    )
    with patch("axio_tools_docker.sandbox.aiodocker.Docker", cls):
        async with DockerSandbox() as sb:
            assert sb.available_shells == ("bash", "sh", "zsh", "dash")
            tool = next(item for item in sb.tools if item.name == "shell")

    shell_schema = tool.input_schema["properties"]["shell"]
    assert shell_schema["anyOf"][0]["enum"] == ["bash", "sh", "zsh", "dash"]
    assert "Omit shell to use bash" in shell_schema["description"]
    assert "Docker-observed" in tool.description


async def test_shell_discovery_falls_back_to_sh_and_is_cached() -> None:
    cls, client, container = mock_docker_factory(shell_paths=("/bin/sh",))
    with patch("axio_tools_docker.sandbox.aiodocker.Docker", cls):
        async with DockerSandbox() as sb:
            assert sb.available_shells == ("sh",)
            probe_count = sum(call.kwargs["cmd"][1:] == ["-c", "exit 0"] for call in container.exec.await_args_list)
            await sb.exec("true")
            await sb.exec("true")
            assert (
                sum(call.kwargs["cmd"][1:] == ["-c", "exit 0"] for call in container.exec.await_args_list)
                == probe_count
            )

    command_calls = [call.kwargs["cmd"] for call in container.exec.await_args_list if call.kwargs["cmd"][2] == "true"]
    assert command_calls == [["/bin/sh", "-c", "true"], ["/bin/sh", "-c", "true"]]


async def test_shell_explicit_selection_uses_cached_path() -> None:
    cls, client, container = mock_docker_factory(shell_paths=("/usr/bin/bash", "/bin/sh"))
    with patch("axio_tools_docker.sandbox.aiodocker.Docker", cls):
        async with DockerSandbox() as sb:
            await sb.exec("printf selected", shell="sh")

    assert container.exec.await_args.kwargs["cmd"] == ["/bin/sh", "-c", "printf selected"]


async def test_shell_invalid_selection_cannot_become_argv() -> None:
    cls, client, container = mock_docker_factory(shell_paths=("/usr/bin/bash",))
    with patch("axio_tools_docker.sandbox.aiodocker.Docker", cls):
        async with DockerSandbox() as sb:
            command_count = len(container.exec.await_args_list)
            with pytest.raises(HandlerError, match=r"available shells: bash"):
                await sb.exec("true", shell="bash; touch /tmp/injected")
            assert len(container.exec.await_args_list) == command_count


async def test_shell_no_discovered_shell_is_expected_failure() -> None:
    cls, client, container = mock_docker_factory(shell_paths=())
    with patch("axio_tools_docker.sandbox.aiodocker.Docker", cls):
        async with DockerSandbox() as sb:
            assert sb.available_shells == ()
            with pytest.raises(HandlerError, match="No supported shell found in the container PATH"):
                await sb.exec("true")


async def test_tools_raises_outside_context() -> None:
    sb = DockerSandbox()
    with pytest.raises(RuntimeError, match="async context manager"):
        _ = sb.tools


async def test_container_id_inside_context() -> None:
    cls, client, container = mock_docker_factory()
    container.id = "abc123def456"
    with patch("axio_tools_docker.sandbox.aiodocker.Docker", cls):
        async with DockerSandbox() as sb:
            cid = sb.container_id
    assert cid == "abc123def456"


async def test_container_id_raises_outside_context() -> None:
    sb = DockerSandbox()
    with pytest.raises(RuntimeError, match="async context manager"):
        _ = sb.container_id


async def test_tools_unavailable_after_exit() -> None:
    cls, client, container = mock_docker_factory()
    with patch("axio_tools_docker.sandbox.aiodocker.Docker", cls):
        async with DockerSandbox() as sb:
            pass
    with pytest.raises(RuntimeError):
        _ = sb.tools


async def test_shell_discovery_cancellation_cleans_up_container_and_client() -> None:
    cls, client, container = mock_docker_factory()
    discovery_started = asyncio.Event()
    cleanup_started = asyncio.Event()
    allow_cleanup = asyncio.Event()

    async def blocked_discovery(self: DockerSandbox) -> tuple[sandbox_module._ShellExecutable, ...]:
        discovery_started.set()
        await asyncio.Event().wait()
        return ()

    async def blocked_delete(*, force: bool) -> None:
        assert force is True
        cleanup_started.set()
        await allow_cleanup.wait()

    container.delete = AsyncMock(side_effect=blocked_delete)
    with (
        patch("axio_tools_docker.sandbox.aiodocker.Docker", cls),
        patch.object(DockerSandbox, "_discover_shells", blocked_discovery),
    ):
        sandbox = DockerSandbox()
        enter_task = asyncio.create_task(sandbox.__aenter__())
        await discovery_started.wait()
        enter_task.cancel()
        await cleanup_started.wait()
        enter_task.cancel()
        allow_cleanup.set()
        with pytest.raises(asyncio.CancelledError):
            await enter_task

    container.delete.assert_awaited_once_with(force=True)
    client.close.assert_awaited_once()
    assert sandbox._container is None
    assert sandbox._client is None


@pytest.mark.parametrize(
    ("status", "message"),
    [(404, "No such container"), (409, "Container is not running")],
)
async def test_shell_discovery_container_lifecycle_failure_is_not_absent_shell(status: int, message: str) -> None:
    cls, client, container = mock_docker_factory()
    stopped = aiodocker.exceptions.DockerError(status, message)
    container.exec = AsyncMock(side_effect=stopped)
    sandbox = DockerSandbox()

    with patch("axio_tools_docker.sandbox.aiodocker.Docker", cls):
        with pytest.raises(aiodocker.exceptions.DockerError) as raised:
            await sandbox.__aenter__()

    assert raised.value is stopped
    container.delete.assert_awaited_once_with(force=True)
    client.close.assert_awaited_once()
    assert sandbox._container is None
    assert sandbox._client is None


@pytest.mark.parametrize("failure_type", [KeyboardInterrupt, SystemExit])
async def test_shell_discovery_base_exception_preserved_and_cleaned_up(
    failure_type: type[BaseException],
) -> None:
    cls, client, container = mock_docker_factory()
    failure = failure_type("startup stopped")

    async def failed_discovery(self: DockerSandbox) -> tuple[sandbox_module._ShellExecutable, ...]:
        raise failure

    sandbox = DockerSandbox()
    caught: BaseException | None = None
    with (
        patch("axio_tools_docker.sandbox.aiodocker.Docker", cls),
        patch.object(DockerSandbox, "_discover_shells", failed_discovery),
    ):
        try:
            await sandbox.__aenter__()
        except BaseException as exc:
            caught = exc

    assert caught is failure
    container.delete.assert_awaited_once_with(force=True)
    client.close.assert_awaited_once()
    assert sandbox._container is None
    assert sandbox._client is None


# ---------------------------------------------------------------------------
# exec
# ---------------------------------------------------------------------------


async def test_exec_stdout() -> None:
    cls, client, container = mock_docker_factory(exec_messages=[(1, b"hello\n")])
    with patch("axio_tools_docker.sandbox.aiodocker.Docker", cls):
        async with DockerSandbox() as sb:
            result = await sb.exec("echo hello")
    assert result == "hello"


async def test_exec_stderr() -> None:
    cls, client, container = mock_docker_factory(exec_messages=[(2, b"oops\n")])
    with patch("axio_tools_docker.sandbox.aiodocker.Docker", cls):
        async with DockerSandbox() as sb:
            result = await sb.exec("bad_cmd")
    assert "[stderr]" in result
    assert "oops" in result


async def test_exec_nonzero_exit_code() -> None:
    cls, client, container = mock_docker_factory(exec_messages=[(2, b"fail\n")], exec_exit_code=1)
    with patch("axio_tools_docker.sandbox.aiodocker.Docker", cls):
        async with DockerSandbox() as sb:
            result = await sb.exec("false")
    assert "[exit code: 1]" in result


async def test_exec_no_output() -> None:
    cls, client, container = mock_docker_factory(exec_messages=[])
    with patch("axio_tools_docker.sandbox.aiodocker.Docker", cls):
        async with DockerSandbox() as sb:
            result = await sb.exec("true")
    assert result == "(no output)"


async def test_exec_timeout() -> None:
    async def hanging_read_out() -> Any:
        await asyncio.sleep(999)

    cls, client, container = mock_docker_factory()
    mock_stream_slow = MagicMock()
    mock_stream_slow.read_out = hanging_read_out
    mock_stream_slow.close = AsyncMock()
    mock_exec_slow = MagicMock()
    mock_exec_slow.start = MagicMock(return_value=mock_stream_slow)
    mock_exec_slow.inspect = AsyncMock(return_value={"ExitCode": 0})
    container.command_exec = mock_exec_slow

    with patch("axio_tools_docker.sandbox.aiodocker.Docker", cls):
        async with DockerSandbox() as sb:
            result = await sb.exec("sleep 999", timeout=0.01)
    assert "[timeout after 0.01s]" in result


async def test_exec_stream_yields_stdout_stderr_and_exit_status() -> None:
    cls, client, container = mock_docker_factory(
        exec_messages=[(1, b"first\n"), (2, b"warning\n"), (1, b"second\n")],
        exec_exit_code=7,
    )
    with patch("axio_tools_docker.sandbox.aiodocker.Docker", cls):
        async with DockerSandbox() as sb:
            chunks = [chunk async for chunk in sb.exec_stream("mixed")]

    assert chunks == [
        ("stdout", "first\n"),
        ("stderr", "warning\n"),
        ("stdout", "second\n"),
        ("stderr", "[exit code: 7]"),
    ]
    container.exec.return_value.start.return_value.close.assert_awaited_once()


async def test_exec_stream_preserves_utf8_split_across_frames() -> None:
    cls, client, container = mock_docker_factory(exec_messages=[(1, b"price: \xe2"), (1, b"\x82\xac\n")])
    with patch("axio_tools_docker.sandbox.aiodocker.Docker", cls):
        async with DockerSandbox() as sb:
            chunks = [chunk async for chunk in sb.exec_stream("unicode")]

    assert chunks == [("stdout", "price: "), ("stdout", "€\n")]


async def test_exec_stream_timeout_does_not_cancel_slow_consumer() -> None:
    calls = 0

    async def read_out() -> Any:
        nonlocal calls
        calls += 1
        if calls == 1:
            msg = MagicMock()
            msg.stream = 1
            msg.data = b"started\n"
            return msg
        await asyncio.sleep(999)

    cls, client, container = mock_docker_factory()
    stream = container.exec.return_value.start.return_value
    stream.read_out = read_out
    with patch("axio_tools_docker.sandbox.aiodocker.Docker", cls):
        async with DockerSandbox() as sb:
            chunks = sb.exec_stream("slow consumer", timeout=0.01)
            assert await anext(chunks) == ("stdout", "started\n")
            await asyncio.sleep(0.02)
            assert await anext(chunks) == ("stderr", "[timeout after 0.01s]")
            with pytest.raises(StopAsyncIteration):
                await anext(chunks)

    stream.close.assert_awaited_once()


async def test_shell_tool_streams_and_preserves_final_format() -> None:
    cls, client, container = mock_docker_factory(
        exec_messages=[(1, b"first\n"), (2, b"warning\n"), (1, b"second\n")],
        exec_exit_code=7,
    )
    with patch("axio_tools_docker.sandbox.aiodocker.Docker", cls):
        async with DockerSandbox() as sb:
            tool = next(tool for tool in sb.tools if tool.name == "shell")
            assert tool.supports_streaming
            chunks = [chunk async for chunk in tool.call_streaming(command="mixed")]
            records = [(float(index), key, text) for index, (key, text) in enumerate(chunks)]
            result = tool.format_stream_result(records)

    assert chunks == [
        ("stdout", "first\n"),
        ("stderr", "warning\n"),
        ("stdout", "second\n"),
        ("stderr", "[exit code: 7]"),
    ]
    assert result == "first\n\n[stderr]\nwarning\n\n[stdout]\nsecond\n\n[exit code: 7]"


def test_shell_final_format_preserves_leading_whitespace_and_only_trims_final_newlines() -> None:
    result = sandbox_module._format_shell_records(
        [
            (0.0, "stdout", "        first\n"),
            (0.1, "stderr", "  warning  \n"),
        ]
    )

    assert result == "        first\n\n[stderr]\n  warning  "
    assert sandbox_module._format_shell_records([(0.0, "stdout", " \t\n")]) == "(no output)"


async def test_exec_stdin_writes_temp_file() -> None:
    """When stdin is provided, a temp file is written before the command."""
    cls, client, container = mock_docker_factory()
    with patch("axio_tools_docker.sandbox.aiodocker.Docker", cls):
        async with DockerSandbox() as sb:
            await sb.exec("cat", stdin="hello stdin")
    # put_archive must have been called at least once (for the stdin temp file)
    assert container.put_archive.await_count >= 1


# ---------------------------------------------------------------------------
# write_file
# ---------------------------------------------------------------------------


async def test_write_file_calls_put_archive() -> None:
    cls, client, container = mock_docker_factory()
    with patch("axio_tools_docker.sandbox.aiodocker.Docker", cls):
        async with DockerSandbox() as sb:
            result = await sb.write_file("/workspace/hello.py", "print('hi')")
    assert "hello.py" in result
    container.put_archive.assert_awaited()


async def test_write_file_tar_contains_content() -> None:
    cls, client, container = mock_docker_factory()
    with patch("axio_tools_docker.sandbox.aiodocker.Docker", cls):
        async with DockerSandbox() as sb:
            await sb.write_file("/workspace/hello.py", "print('hi')")

    call_kwargs = container.put_archive.call_args
    tar_bytes: bytes = call_kwargs.kwargs.get("data") or call_kwargs.args[1]
    with tarfile.open(fileobj=io.BytesIO(tar_bytes), mode="r:") as tar:
        member = tar.next()
        assert member is not None
        assert member.name == "hello.py"
        f = tar.extractfile(member)
        assert f is not None
        assert f.read() == b"print('hi')"


async def test_write_file_tar_uses_numeric_runtime_owner() -> None:
    cls, client, container = mock_docker_factory()
    with patch("axio_tools_docker.sandbox.aiodocker.Docker", cls):
        async with DockerSandbox(user="1234:5678") as sb:
            await sb.write_file("/workspace/hello.py", "print('hi')")

    call_kwargs = container.put_archive.call_args
    tar_bytes: bytes = call_kwargs.kwargs.get("data") or call_kwargs.args[1]
    with tarfile.open(fileobj=io.BytesIO(tar_bytes), mode="r:") as tar:
        member = tar.next()
        assert member is not None
        assert member.uid == 1234
        assert member.gid == 5678


async def test_write_file_correct_parent_dir() -> None:
    cls, client, container = mock_docker_factory()
    with patch("axio_tools_docker.sandbox.aiodocker.Docker", cls):
        async with DockerSandbox() as sb:
            await sb.write_file("/workspace/hello.py", "x")

    call_kwargs = container.put_archive.call_args
    path_arg: str = call_kwargs.kwargs.get("path") or call_kwargs.args[0]
    assert path_arg == "/workspace"


# ---------------------------------------------------------------------------
# read_file_bytes
# ---------------------------------------------------------------------------


async def test_read_file_bytes_extracts_content() -> None:
    tar_file = make_tar_file("hello.py", b"print('hi')")
    cls, client, container = mock_docker_factory(archive_content=tar_file)
    with patch("axio_tools_docker.sandbox.aiodocker.Docker", cls):
        async with DockerSandbox() as sb:
            result = await sb.read_file_bytes("/workspace/hello.py")
    assert result == b"print('hi')"


# ---------------------------------------------------------------------------
# get_archive - missing path is FileNotFoundError, other failures propagate
# ---------------------------------------------------------------------------


async def test_get_archive_404_raises_file_not_found() -> None:
    cls, client, container = mock_docker_factory()
    container.get_archive = AsyncMock(side_effect=aiodocker.exceptions.DockerError(404, "Not found"))
    with patch("axio_tools_docker.sandbox.aiodocker.Docker", cls):
        async with DockerSandbox() as sb:
            with pytest.raises(FileNotFoundError, match="/workspace/missing.txt"):
                await sb.get_archive("/workspace/missing.txt")


async def test_get_archive_non_404_propagates_as_docker_error() -> None:
    cls, client, container = mock_docker_factory()
    container.get_archive = AsyncMock(side_effect=aiodocker.exceptions.DockerError(500, "daemon error"))
    with patch("axio_tools_docker.sandbox.aiodocker.Docker", cls):
        async with DockerSandbox() as sb:
            with pytest.raises(aiodocker.exceptions.DockerError, match="daemon error"):
                await sb.get_archive("/workspace/whatever.txt")


# ---------------------------------------------------------------------------
# Tool handlers - expected failures raise HandlerError, not raw exceptions
# ---------------------------------------------------------------------------


def _bind_context(sb: DockerSandbox) -> Any:
    """Set CONTEXT the same way Tool.__call__ does, so handlers can be called
    directly without going through Tool's own catch-all exception wrapping.
    """
    return CONTEXT.set(sb)


async def test_read_file_handler_missing_path_raises_handler_error() -> None:
    cls, client, container = mock_docker_factory()
    container.get_archive = AsyncMock(side_effect=aiodocker.exceptions.DockerError(404, "Not found"))
    with patch("axio_tools_docker.sandbox.aiodocker.Docker", cls):
        async with DockerSandbox() as sb:
            token = _bind_context(sb)
            try:
                with pytest.raises(HandlerError, match="missing.txt"):
                    await sandbox_module.read_file(path="missing.txt")
            finally:
                CONTEXT.reset(token)


async def test_list_files_handler_missing_path_raises_handler_error() -> None:
    cls, client, container = mock_docker_factory(exec_messages=[(1, b"AXIO_LIST_ERROR\0missing\0")])
    with patch("axio_tools_docker.sandbox.aiodocker.Docker", cls):
        async with DockerSandbox() as sb:
            token = _bind_context(sb)
            try:
                with pytest.raises(HandlerError, match="missing_dir"):
                    await sandbox_module.list_files(path="missing_dir")
            finally:
                CONTEXT.reset(token)


async def test_list_files_uses_depth_one_exec_without_archive() -> None:
    output = make_listing_output(
        ("81a4", 3, 1_700_000_000, "/workspace/project/z.txt"),
        ("41ed", 4096, 1_700_000_001, "/workspace/project/subdir"),
        ("81a4", 2, 1_700_000_002, "/workspace/project/.hidden"),
    )
    cls, client, container = mock_docker_factory(exec_messages=[(1, output)])
    with patch("axio_tools_docker.sandbox.aiodocker.Docker", cls):
        async with DockerSandbox(workdir="/workspace/project") as sb:
            token = _bind_context(sb)
            try:
                result = await sandbox_module.list_files()
            finally:
                CONTEXT.reset(token)

    assert result.index("subdir/") < result.index(".hidden") < result.index("z.txt")
    assert "drwxr-xr-x" in result
    assert "-rw-r--r--" in result
    container.get_archive.assert_not_awaited()
    command = container.exec.await_args.kwargs["cmd"][2]
    assert command.count("stat -c") == 1
    assert 'stat -c "%f %s %Y" -- "$@"' in command
    assert 'cd "$target"' in command
    assert "find . ! -path ." in command
    assert "-prune -exec" in command


async def test_list_files_protocol_does_not_load_noisy_login_profile() -> None:
    output = make_listing_output(("81a4", 3, 1_700_000_000, "/workspace/project/result.txt"))
    cls, client, container = mock_docker_factory(
        exec_messages=[(1, output)],
        shell_paths=("/usr/bin/bash",),
        login_profile_noise=b"login-profile-noise\n",
    )
    with patch("axio_tools_docker.sandbox.aiodocker.Docker", cls):
        async with DockerSandbox(workdir="/workspace/project") as sandbox:
            token = _bind_context(sandbox)
            try:
                result = await sandbox_module.list_files()
            finally:
                CONTEXT.reset(token)

    assert "result.txt" in result
    assert container.exec.await_args.kwargs["cmd"][0:2] == ["/usr/bin/bash", "-c"]


async def test_list_files_empty_directory() -> None:
    cls, client, container = mock_docker_factory(exec_messages=[(1, make_listing_output())])
    with patch("axio_tools_docker.sandbox.aiodocker.Docker", cls):
        async with DockerSandbox() as sb:
            token = _bind_context(sb)
            try:
                result = await sandbox_module.list_files(path="empty")
            finally:
                CONTEXT.reset(token)
    assert result == "(empty directory)"


async def test_list_files_malformed_metadata_raises_handler_error() -> None:
    output = make_listing_output(("not-a-mode", 3, 1_700_000_000, "/workspace/project/broken"))
    cls, client, container = mock_docker_factory(exec_messages=[(1, output)])
    with patch("axio_tools_docker.sandbox.aiodocker.Docker", cls):
        async with DockerSandbox() as sb:
            token = _bind_context(sb)
            try:
                with pytest.raises(HandlerError, match="parse directory listing"):
                    await sandbox_module.list_files(path="broken")
            finally:
                CONTEXT.reset(token)


async def test_list_files_reports_stderr_instead_of_parsing_it() -> None:
    output = make_listing_output(("81a4", 3, 1_700_000_000, "/workspace/project/vanished"))
    cls, client, container = mock_docker_factory(
        exec_messages=[(1, output), (2, b"stat: cannot stat 'vanished': No such file\n")],
        exec_exit_code=1,
    )
    with patch("axio_tools_docker.sandbox.aiodocker.Docker", cls):
        async with DockerSandbox() as sb:
            token = _bind_context(sb)
            try:
                with pytest.raises(HandlerError, match="cannot stat 'vanished'"):
                    await sandbox_module.list_files(path="changing")
            finally:
                CONTEXT.reset(token)


async def test_list_files_reports_timeout_instead_of_parsing_partial_output() -> None:
    cls, client, container = mock_docker_factory()
    with patch("axio_tools_docker.sandbox.aiodocker.Docker", cls):
        async with DockerSandbox() as sb:

            async def timed_out_stream(
                command: str,
                timeout: float = 30,
                stdin: str | None = None,
                shell: str | None = None,
            ) -> AsyncGenerator[tuple[str, str], None]:
                yield "stdout", make_listing_output(("81a4", 3, 1_700_000_000, "/workspace/project/partial")).decode()
                yield "stderr", sandbox_module._ShellControl("[timeout after 30s]")

            sb.exec_stream = timed_out_stream  # type: ignore[method-assign]
            token = _bind_context(sb)
            try:
                with pytest.raises(HandlerError, match="timeout after 30s"):
                    await sandbox_module.list_files(path="wide")
            finally:
                CONTEXT.reset(token)


async def test_patch_file_handler_missing_path_raises_handler_error() -> None:
    cls, client, container = mock_docker_factory()
    container.get_archive = AsyncMock(side_effect=aiodocker.exceptions.DockerError(404, "Not found"))
    with patch("axio_tools_docker.sandbox.aiodocker.Docker", cls):
        async with DockerSandbox() as sb:
            token = _bind_context(sb)
            try:
                with pytest.raises(HandlerError, match="missing.txt"):
                    await sandbox_module.patch_file(path="missing.txt", from_line=1, to_line=1, content="x")
            finally:
                CONTEXT.reset(token)


async def test_read_file_handler_negative_max_chars_raises_handler_error() -> None:
    cls, client, container = mock_docker_factory()
    with patch("axio_tools_docker.sandbox.aiodocker.Docker", cls):
        async with DockerSandbox() as sb:
            token = _bind_context(sb)
            try:
                with pytest.raises(HandlerError, match="max_chars"):
                    await sandbox_module.read_file(path="file.txt", max_chars=-1)
            finally:
                CONTEXT.reset(token)


async def test_read_file_line_metadata_is_distinct_from_exact_selected_source() -> None:
    tar_file = make_tar_file("lines.txt", b"one\n    two\n\tthree\nfour\n")
    cls, client, container = mock_docker_factory(archive_content=tar_file)
    with patch("axio_tools_docker.sandbox.aiodocker.Docker", cls):
        async with DockerSandbox() as sb:
            token = _bind_context(sb)
            try:
                result = await sandbox_module.read_file(
                    path="lines.txt",
                    start_line=2,
                    end_line=3,
                    line_numbers=True,
                )
            finally:
                CONTEXT.reset(token)

    assert result == "L2│    two\nL3│\tthree\n"


async def test_read_file_handler_non_utf8_without_hex_raises_handler_error() -> None:
    tar_file = make_tar_file("bin.dat", b"\x80\x81\xff")
    cls, client, container = mock_docker_factory(archive_content=tar_file)
    with patch("axio_tools_docker.sandbox.aiodocker.Docker", cls):
        async with DockerSandbox() as sb:
            token = _bind_context(sb)
            try:
                with pytest.raises(HandlerError, match="not valid UTF-8"):
                    await sandbox_module.read_file(path="bin.dat", binary_as_hex=False)
            finally:
                CONTEXT.reset(token)


async def test_patch_file_handler_non_utf8_target_raises_handler_error() -> None:
    tar_file = make_tar_file("bin.dat", b"\x80\x81\xff")
    cls, client, container = mock_docker_factory(archive_content=tar_file)
    with patch("axio_tools_docker.sandbox.aiodocker.Docker", cls):
        async with DockerSandbox() as sb:
            token = _bind_context(sb)
            try:
                with pytest.raises(HandlerError, match="not valid UTF-8"):
                    await sandbox_module.patch_file(path="bin.dat", from_line=1, to_line=1, content="x")
            finally:
                CONTEXT.reset(token)


async def test_patch_file_handler_reports_compact_path_free_diff() -> None:
    """The result keeps exact changes without repeating the tool input path."""
    cls, client, container = mock_docker_factory(
        archive_content=make_tar_file("patch_me.txt", b"line1\nline2\nline3\n")
    )
    with patch("axio_tools_docker.sandbox.aiodocker.Docker", cls):
        async with DockerSandbox() as sb:
            token = _bind_context(sb)
            try:
                result = await sandbox_module.patch_file(
                    path="/workspace/patch_me.txt", from_line=2, to_line=2, content="REPLACED"
                )
            finally:
                CONTEXT.reset(token)

    assert result.startswith("+1 -1\n@@ -1,3 +1,3 @@\n")
    assert "/workspace/patch_me.txt" not in result
    assert "Wrote" not in result
    assert "---" not in result
    assert "+++" not in result
    assert "-line2\n" in result
    assert "+REPLACED\n" in result
    assert " line1\n" in result
    written = container.put_archive.call_args.kwargs["data"]
    member = tarfile.open(fileobj=io.BytesIO(written)).extractfile("patch_me.txt")
    assert member is not None
    assert member.read() == b"line1\nREPLACED\nline3\n"


async def test_patch_file_handler_preserves_leading_spaces_and_tabs_exactly() -> None:
    cls, client, container = mock_docker_factory(
        archive_content=make_tar_file("patch_me.txt", b"before\nold one\nold two\nafter\n")
    )
    replacement = "        first\n\tsecond"
    with patch("axio_tools_docker.sandbox.aiodocker.Docker", cls):
        async with DockerSandbox() as sb:
            token = _bind_context(sb)
            try:
                await sandbox_module.patch_file(
                    path="patch_me.txt",
                    from_line=2,
                    to_line=3,
                    content=replacement,
                )
            finally:
                CONTEXT.reset(token)

    written = container.put_archive.call_args.kwargs["data"]
    member = tarfile.open(fileobj=io.BytesIO(written)).extractfile("patch_me.txt")
    assert member is not None
    assert member.read() == b"before\n        first\n\tsecond\nafter\n"


async def test_patch_file_handler_indents_only_first_line_and_preserves_following_whitespace() -> None:
    cls, client, container = mock_docker_factory(
        archive_content=make_tar_file("patch_me.txt", b"before\nold one\nold two\nafter\n")
    )
    with patch("axio_tools_docker.sandbox.aiodocker.Docker", cls):
        async with DockerSandbox() as sb:
            token = _bind_context(sb)
            try:
                await sandbox_module.patch_file(
                    path="patch_me.txt",
                    from_line=2,
                    to_line=3,
                    content="first\n\t second",
                    first_line_indent=6,
                )
            finally:
                CONTEXT.reset(token)

    written = container.put_archive.call_args.kwargs["data"]
    member = tarfile.open(fileobj=io.BytesIO(written)).extractfile("patch_me.txt")
    assert member is not None
    assert member.read() == b"before\n      first\n\t second\nafter\n"


async def test_patch_file_handler_zero_indent_preserves_content_exactly() -> None:
    cls, client, container = mock_docker_factory(archive_content=make_tar_file("patch_me.txt", b"old\n"))
    replacement = " \tfirst\n\t second\n"
    with patch("axio_tools_docker.sandbox.aiodocker.Docker", cls):
        async with DockerSandbox() as sb:
            token = _bind_context(sb)
            try:
                await sandbox_module.patch_file(
                    path="patch_me.txt",
                    from_line=1,
                    to_line=1,
                    content=replacement,
                    first_line_indent=0,
                )
            finally:
                CONTEXT.reset(token)

    written = container.put_archive.call_args.kwargs["data"]
    member = tarfile.open(fileobj=io.BytesIO(written)).extractfile("patch_me.txt")
    assert member is not None
    assert member.read() == replacement.encode()


@pytest.mark.parametrize("content", [" already indented", "\talready indented"])
async def test_patch_file_handler_rejects_double_first_line_indent_without_writing(content: str) -> None:
    cls, client, container = mock_docker_factory(archive_content=make_tar_file("patch_me.txt", b"old\n"))
    with patch("axio_tools_docker.sandbox.aiodocker.Docker", cls):
        async with DockerSandbox() as sb:
            token = _bind_context(sb)
            try:
                with pytest.raises(HandlerError, match="already begins with whitespace"):
                    await sandbox_module.patch_file(
                        path="patch_me.txt",
                        from_line=1,
                        to_line=1,
                        content=content,
                        first_line_indent=4,
                    )
            finally:
                CONTEXT.reset(token)

    container.put_archive.assert_not_awaited()


async def test_patch_file_handler_rejects_indent_for_empty_deletion_without_writing() -> None:
    cls, client, container = mock_docker_factory(archive_content=make_tar_file("patch_me.txt", b"old\n"))
    with patch("axio_tools_docker.sandbox.aiodocker.Docker", cls):
        async with DockerSandbox() as sb:
            token = _bind_context(sb)
            try:
                with pytest.raises(HandlerError, match="empty content"):
                    await sandbox_module.patch_file(
                        path="patch_me.txt",
                        from_line=1,
                        to_line=1,
                        content="",
                        first_line_indent=4,
                    )
            finally:
                CONTEXT.reset(token)

    container.put_archive.assert_not_awaited()


@pytest.mark.parametrize(
    ("before", "from_line", "to_line", "content", "after", "expected_result"),
    [
        (
            b"same",
            1,
            1,
            "same\n",
            b"same\n",
            "+1 -1\n@@ -1 +1 @@\n-same\n\\ No newline at end of file\n+same\n",
        ),
        (
            b"same\n",
            1,
            1,
            "same",
            b"same",
            "+1 -1\n@@ -1 +1 @@\n-same\n+same\n\\ No newline at end of file\n",
        ),
        (
            b"one\ntwo",
            2,
            2,
            "two\n",
            b"one\ntwo\n",
            "+1 -1\n@@ -1,2 +1,2 @@\n one\n-two\n\\ No newline at end of file\n+two\n",
        ),
        (
            b"one\ntwo\n",
            2,
            2,
            "two",
            b"one\ntwo",
            "+1 -1\n@@ -1,2 +1,2 @@\n one\n-two\n+two\n\\ No newline at end of file\n",
        ),
        (b"", 1, 0, "\n", b"\n", "+1 -0\n@@ -0,0 +1 @@\n+\n"),
        (b"\n", 1, 1, "", b"", "+0 -1\n@@ -1 +0,0 @@\n-\n"),
    ],
)
async def test_patch_file_handler_reports_final_newline_changes_exactly(
    before: bytes,
    from_line: int,
    to_line: int,
    content: str,
    after: bytes,
    expected_result: str,
) -> None:
    cls, client, container = mock_docker_factory(archive_content=make_tar_file("f.txt", before))
    with patch("axio_tools_docker.sandbox.aiodocker.Docker", cls):
        async with DockerSandbox() as sb:
            token = _bind_context(sb)
            try:
                result = await sandbox_module.patch_file(
                    path="f.txt",
                    from_line=from_line,
                    to_line=to_line,
                    content=content,
                )
            finally:
                CONTEXT.reset(token)

    assert result == expected_result
    written = container.put_archive.call_args.kwargs["data"]
    member = tarfile.open(fileobj=io.BytesIO(written)).extractfile("f.txt")
    assert member is not None
    assert member.read() == after


async def test_write_file_handler_diffs_only_when_replacing_text() -> None:
    """Replacing text shows a diff; a new or binary target still gets written.

    The container round-trip that answers "what did it hold" is also the one
    that answers "does it exist", so all three cases share a single fetch.
    """
    cls, client, container = mock_docker_factory(archive_content=make_tar_file("data.txt", b"old line\n"))
    with patch("axio_tools_docker.sandbox.aiodocker.Docker", cls):
        async with DockerSandbox() as sb:
            token = _bind_context(sb)
            try:
                replaced = await sandbox_module.write_file(path="/workspace/data.txt", content="new line\n")

                container.get_archive = AsyncMock(side_effect=aiodocker.exceptions.DockerError(404, "Not found"))
                created = await sandbox_module.write_file(path="/workspace/fresh.txt", content="brand new\n")

                container.get_archive = AsyncMock(return_value=make_tar_file("bin.dat", b"\xff\xfe\x00binary"))
                overwritten_binary = await sandbox_module.write_file(path="/workspace/bin.dat", content="text\n")
            finally:
                CONTEXT.reset(token)

    assert replaced.startswith("Wrote 9 bytes to /workspace/data.txt\nChanged /workspace/data.txt:\n")
    assert "-old line\n" in replaced
    assert "+new line\n" in replaced
    assert created == "Wrote 10 bytes to /workspace/fresh.txt"
    assert overwritten_binary == "Wrote 5 bytes to /workspace/bin.dat"


# ---------------------------------------------------------------------------
# Container config
# ---------------------------------------------------------------------------


async def test_volumes_binds() -> None:
    cls, client, container = mock_docker_factory()
    with patch("axio_tools_docker.sandbox.aiodocker.Docker", cls):
        async with DockerSandbox(volumes={"/container/path": "/host/path"}):
            pass
    config = client._captured_config[0]
    assert "/host/path:/container/path" in config["HostConfig"]["Binds"]


async def test_read_only_volumes_binds() -> None:
    cls, client, container = mock_docker_factory()
    with patch("axio_tools_docker.sandbox.aiodocker.Docker", cls):
        async with DockerSandbox(read_only_volumes={"/datasets": "/host/datasets"}):
            pass
    config = client._captured_config[0]
    assert "/host/datasets:/datasets:ro" in config["HostConfig"]["Binds"]


async def test_named_volumes_binds() -> None:
    cls, client, container = mock_docker_factory()
    with patch("axio_tools_docker.sandbox.aiodocker.Docker", cls):
        async with DockerSandbox(named_volumes={"/data": "myvolume"}):
            pass
    config = client._captured_config[0]
    assert "myvolume:/data" in config["HostConfig"]["Binds"]


async def test_named_volumes_combined_with_bind_mounts() -> None:
    cls, client, container = mock_docker_factory()
    with patch("axio_tools_docker.sandbox.aiodocker.Docker", cls):
        async with DockerSandbox(
            volumes={"/app": "/host/app"},
            named_volumes={"/data": "myvolume"},
        ):
            pass
    config = client._captured_config[0]
    binds = config["HostConfig"]["Binds"]
    assert "/host/app:/app" in binds
    assert "myvolume:/data" in binds


async def test_volumes_remove_deletes_named_volumes_on_exit() -> None:
    cls, client, container = mock_docker_factory()
    mock_volume = MagicMock()
    mock_volume.delete = AsyncMock()
    client.volumes = MagicMock()
    client.volumes.get = AsyncMock(return_value=mock_volume)
    with patch("axio_tools_docker.sandbox.aiodocker.Docker", cls):
        async with DockerSandbox(named_volumes={"/data": "myvolume"}, volumes_remove=True):
            pass
    client.volumes.get.assert_awaited_once_with("myvolume")
    mock_volume.delete.assert_awaited_once()


async def test_volumes_remove_not_called_when_attached() -> None:
    cls, client, container = mock_docker_factory()
    client.containers.get = AsyncMock(return_value=container)
    mock_volume = MagicMock()
    mock_volume.delete = AsyncMock()
    client.volumes = MagicMock()
    client.volumes.get = AsyncMock(return_value=mock_volume)
    with patch("axio_tools_docker.sandbox.aiodocker.Docker", cls):
        async with DockerSandbox(
            name="existing",
            named_volumes={"/data": "myvolume"},
            volumes_remove=True,
        ):
            pass
    mock_volume.delete.assert_not_awaited()


async def test_volumes_remove_false_does_not_delete() -> None:
    cls, client, container = mock_docker_factory()
    mock_volume = MagicMock()
    mock_volume.delete = AsyncMock()
    client.volumes = MagicMock()
    client.volumes.get = AsyncMock(return_value=mock_volume)
    with patch("axio_tools_docker.sandbox.aiodocker.Docker", cls):
        async with DockerSandbox(named_volumes={"/data": "myvolume"}, volumes_remove=False):
            pass
    mock_volume.delete.assert_not_awaited()


async def test_network_mode_none_when_disabled() -> None:
    cls, client, container = mock_docker_factory()
    with patch("axio_tools_docker.sandbox.aiodocker.Docker", cls):
        async with DockerSandbox(network=False):
            pass
    config = client._captured_config[0]
    assert config["HostConfig"]["NetworkMode"] == "none"


async def test_network_mode_absent_when_true() -> None:
    cls, client, container = mock_docker_factory()
    with patch("axio_tools_docker.sandbox.aiodocker.Docker", cls):
        async with DockerSandbox(network=True):
            pass
    config = client._captured_config[0]
    assert "NetworkMode" not in config["HostConfig"]


async def test_network_mode_string() -> None:
    cls, client, container = mock_docker_factory()
    with patch("axio_tools_docker.sandbox.aiodocker.Docker", cls):
        async with DockerSandbox(network="my-project_default"):
            pass
    config = client._captured_config[0]
    assert config["HostConfig"]["NetworkMode"] == "my-project_default"


async def test_required_internal_network_is_verified() -> None:
    cls, client, container = mock_docker_factory()
    with patch("axio_tools_docker.sandbox.aiodocker.Docker", cls):
        async with DockerSandbox(network="agent-egress", require_internal_network=True):
            pass
    client.networks.get.assert_awaited_once_with("agent-egress")
    assert client._captured_config[0]["HostConfig"]["NetworkMode"] == "verified-network-id"


async def test_required_internal_network_unavailable_closes_client() -> None:
    cls, client, container = mock_docker_factory()
    client.networks.get = AsyncMock(side_effect=aiodocker.exceptions.DockerError(404, "not found"))
    with patch("axio_tools_docker.sandbox.aiodocker.Docker", cls):
        with pytest.raises(RuntimeError, match="is unavailable"):
            async with DockerSandbox(network="missing", require_internal_network=True):
                pass
    client.close.assert_awaited_once()
    container.start.assert_not_awaited()


async def test_required_internal_network_create_race_closes_client() -> None:
    cls, client, container = mock_docker_factory()
    client.containers.create = AsyncMock(side_effect=aiodocker.exceptions.DockerError(404, "network gone"))
    with patch("axio_tools_docker.sandbox.aiodocker.Docker", cls):
        with pytest.raises(aiodocker.exceptions.DockerError):
            async with DockerSandbox(network="agent-egress", require_internal_network=True):
                pass
    client.close.assert_awaited_once()
    container.start.assert_not_awaited()


@pytest.mark.parametrize("network_id", [None, "", "   ", " id-with-spaces ", 42])
async def test_required_internal_network_rejects_missing_stable_id(network_id: object) -> None:
    cls, client, container = mock_docker_factory()
    docker_network = await client.networks.get("agent-egress")
    docker_network.show = AsyncMock(return_value={"Internal": True, "Id": network_id})
    client.networks.get.reset_mock()
    with patch("axio_tools_docker.sandbox.aiodocker.Docker", cls):
        with pytest.raises(RuntimeError, match="no stable ID"):
            async with DockerSandbox(network="agent-egress", require_internal_network=True):
                pass
    client.close.assert_awaited_once()
    container.start.assert_not_awaited()


@pytest.mark.parametrize("internal", [False, None, "true", 1])
async def test_required_internal_network_rejects_routed_or_malformed_network(internal: object) -> None:
    cls, client, container = mock_docker_factory()
    docker_network = await client.networks.get("agent-egress")
    docker_network.show = AsyncMock(return_value={"Internal": internal, "Id": "routed-network-id"})
    client.networks.get.reset_mock()
    with patch("axio_tools_docker.sandbox.aiodocker.Docker", cls):
        with pytest.raises(RuntimeError, match="is not internal"):
            async with DockerSandbox(network="agent-egress", require_internal_network=True):
                pass
    client.close.assert_awaited_once()
    container.start.assert_not_awaited()


def test_required_internal_network_needs_named_network() -> None:
    with pytest.raises(ValueError, match="named Docker network"):
        DockerSandbox(network=False, require_internal_network=True)


def test_required_internal_network_rejects_named_container_reuse() -> None:
    with pytest.raises(ValueError, match="named-container reuse"):
        DockerSandbox(network="agent-egress", name="persistent", require_internal_network=True)


async def test_network_mode_host() -> None:
    cls, client, container = mock_docker_factory()
    with patch("axio_tools_docker.sandbox.aiodocker.Docker", cls):
        async with DockerSandbox(network="host"):
            pass
    config = client._captured_config[0]
    assert config["HostConfig"]["NetworkMode"] == "host"


async def test_init_always_true() -> None:
    cls, client, container = mock_docker_factory()
    with patch("axio_tools_docker.sandbox.aiodocker.Docker", cls):
        async with DockerSandbox():
            pass
    config = client._captured_config[0]
    assert config["HostConfig"]["Init"] is True


async def test_memory_parsed() -> None:
    cls, client, container = mock_docker_factory()
    with patch("axio_tools_docker.sandbox.aiodocker.Docker", cls):
        async with DockerSandbox(memory="256m"):
            pass
    config = client._captured_config[0]
    assert config["HostConfig"]["Memory"] == 256 * 1024 * 1024


async def test_cpus_as_nanocpus() -> None:
    cls, client, container = mock_docker_factory()
    with patch("axio_tools_docker.sandbox.aiodocker.Docker", cls):
        async with DockerSandbox(cpus="1.0"):
            pass
    config = client._captured_config[0]
    assert config["HostConfig"]["NanoCPUs"] == 1_000_000_000


async def test_custom_url() -> None:
    cls, client, container = mock_docker_factory()
    with patch("axio_tools_docker.sandbox.aiodocker.Docker", cls) as docker_cls:
        async with DockerSandbox("tcp://localhost:2375"):
            pass
    docker_cls.assert_called_once_with(url="tcp://localhost:2375")


async def test_env_vars() -> None:
    cls, client, container = mock_docker_factory()
    with patch("axio_tools_docker.sandbox.aiodocker.Docker", cls):
        async with DockerSandbox(env={"FOO": "bar", "BAZ": "qux"}):
            pass
    config = client._captured_config[0]
    assert "FOO=bar" in config["Env"]
    assert "BAZ=qux" in config["Env"]


async def test_no_env_by_default() -> None:
    cls, client, container = mock_docker_factory()
    with patch("axio_tools_docker.sandbox.aiodocker.Docker", cls):
        async with DockerSandbox():
            pass
    config = client._captured_config[0]
    assert config["Env"] == []


async def test_user_set() -> None:
    cls, client, container = mock_docker_factory()
    with patch("axio_tools_docker.sandbox.aiodocker.Docker", cls):
        async with DockerSandbox(user="nobody"):
            pass
    config = client._captured_config[0]
    assert config["User"] == "nobody"


async def test_user_absent_by_default() -> None:
    cls, client, container = mock_docker_factory()
    with patch("axio_tools_docker.sandbox.aiodocker.Docker", cls):
        async with DockerSandbox():
            pass
    config = client._captured_config[0]
    assert "User" not in config


async def test_supplementary_groups_set() -> None:
    cls, client, container = mock_docker_factory()
    with patch("axio_tools_docker.sandbox.aiodocker.Docker", cls):
        async with DockerSandbox(group_add=["20", "998"]):
            pass
    config = client._captured_config[0]
    assert config["HostConfig"]["GroupAdd"] == ["20", "998"]


async def test_container_name_passed() -> None:
    cls, client, container = mock_docker_factory()

    captured_kwargs: list[dict[str, Any]] = []

    original_create = client.containers.create

    async def create_with_capture(**kwargs: Any) -> Any:
        captured_kwargs.append(kwargs)
        return await original_create(**kwargs)

    client.containers.create = create_with_capture

    with patch("axio_tools_docker.sandbox.aiodocker.Docker", cls):
        async with DockerSandbox(name="my-sandbox"):
            pass
    assert any(kw.get("name") == "my-sandbox" for kw in captured_kwargs)


async def test_no_name_by_default() -> None:
    cls, client, container = mock_docker_factory()

    captured_kwargs: list[dict[str, Any]] = []

    original_create = client.containers.create

    async def create_with_capture(**kwargs: Any) -> Any:
        captured_kwargs.append(kwargs)
        return await original_create(**kwargs)

    client.containers.create = create_with_capture

    with patch("axio_tools_docker.sandbox.aiodocker.Docker", cls):
        async with DockerSandbox():
            pass
    assert all("name" not in kw for kw in captured_kwargs)


async def test_remove_true_deletes_container() -> None:
    cls, client, container = mock_docker_factory()
    with patch("axio_tools_docker.sandbox.aiodocker.Docker", cls):
        async with DockerSandbox(remove=True):
            pass
    container.delete.assert_awaited_once_with(force=True)


async def test_remove_false_keeps_container() -> None:
    cls, client, container = mock_docker_factory()
    with patch("axio_tools_docker.sandbox.aiodocker.Docker", cls):
        async with DockerSandbox(remove=False):
            pass
    container.delete.assert_not_awaited()


# ---------------------------------------------------------------------------
# read_only
# ---------------------------------------------------------------------------


async def test_read_only_sets_flag() -> None:
    cls, client, container = mock_docker_factory()
    with patch("axio_tools_docker.sandbox.aiodocker.Docker", cls):
        async with DockerSandbox(read_only=True):
            pass
    config = client._captured_config[0]
    assert config["HostConfig"]["ReadonlyRootfs"] is True


async def test_read_only_absent_by_default() -> None:
    cls, client, container = mock_docker_factory()
    with patch("axio_tools_docker.sandbox.aiodocker.Docker", cls):
        async with DockerSandbox():
            pass
    config = client._captured_config[0]
    assert "ReadonlyRootfs" not in config["HostConfig"]


# ---------------------------------------------------------------------------
# shm_size
# ---------------------------------------------------------------------------


async def test_shm_size_parsed() -> None:
    cls, client, container = mock_docker_factory()
    with patch("axio_tools_docker.sandbox.aiodocker.Docker", cls):
        async with DockerSandbox(shm_size="64m"):
            pass
    config = client._captured_config[0]
    assert config["HostConfig"]["ShmSize"] == 64 * 1024 * 1024


async def test_shm_size_absent_by_default() -> None:
    cls, client, container = mock_docker_factory()
    with patch("axio_tools_docker.sandbox.aiodocker.Docker", cls):
        async with DockerSandbox():
            pass
    config = client._captured_config[0]
    assert "ShmSize" not in config["HostConfig"]


# ---------------------------------------------------------------------------
# cap_add / cap_drop
# ---------------------------------------------------------------------------


async def test_cap_add() -> None:
    cls, client, container = mock_docker_factory()
    with patch("axio_tools_docker.sandbox.aiodocker.Docker", cls):
        async with DockerSandbox(cap_add=["NET_ADMIN", "SYS_PTRACE"]):
            pass
    config = client._captured_config[0]
    assert config["HostConfig"]["CapAdd"] == ["NET_ADMIN", "SYS_PTRACE"]


async def test_cap_drop() -> None:
    cls, client, container = mock_docker_factory()
    with patch("axio_tools_docker.sandbox.aiodocker.Docker", cls):
        async with DockerSandbox(cap_drop=["ALL"]):
            pass
    config = client._captured_config[0]
    assert config["HostConfig"]["CapDrop"] == ["ALL"]


async def test_cap_absent_by_default() -> None:
    cls, client, container = mock_docker_factory()
    with patch("axio_tools_docker.sandbox.aiodocker.Docker", cls):
        async with DockerSandbox():
            pass
    config = client._captured_config[0]
    assert "CapAdd" not in config["HostConfig"]
    assert "CapDrop" not in config["HostConfig"]


# ---------------------------------------------------------------------------
# privileged
# ---------------------------------------------------------------------------


async def test_privileged() -> None:
    cls, client, container = mock_docker_factory()
    with patch("axio_tools_docker.sandbox.aiodocker.Docker", cls):
        async with DockerSandbox(privileged=True):
            pass
    config = client._captured_config[0]
    assert config["HostConfig"]["Privileged"] is True


async def test_privileged_absent_by_default() -> None:
    cls, client, container = mock_docker_factory()
    with patch("axio_tools_docker.sandbox.aiodocker.Docker", cls):
        async with DockerSandbox():
            pass
    config = client._captured_config[0]
    assert "Privileged" not in config["HostConfig"]


# ---------------------------------------------------------------------------
# ulimits
# ---------------------------------------------------------------------------


async def test_ulimits_single_value() -> None:
    """A plain int means soft == hard."""
    cls, client, container = mock_docker_factory()
    with patch("axio_tools_docker.sandbox.aiodocker.Docker", cls):
        async with DockerSandbox(ulimits={"nofile": 1024}):
            pass
    config = client._captured_config[0]
    assert config["HostConfig"]["Ulimits"] == [{"Name": "nofile", "Soft": 1024, "Hard": 1024}]


async def test_ulimits_tuple() -> None:
    """A (soft, hard) tuple sets them independently."""
    cls, client, container = mock_docker_factory()
    with patch("axio_tools_docker.sandbox.aiodocker.Docker", cls):
        async with DockerSandbox(ulimits={"nofile": (1024, 65536)}):
            pass
    config = client._captured_config[0]
    assert config["HostConfig"]["Ulimits"] == [{"Name": "nofile", "Soft": 1024, "Hard": 65536}]


async def test_ulimits_multiple() -> None:
    cls, client, container = mock_docker_factory()
    with patch("axio_tools_docker.sandbox.aiodocker.Docker", cls):
        async with DockerSandbox(ulimits={"nofile": (1024, 65536), "nproc": 512}):
            pass
    config = client._captured_config[0]
    entries = {e["Name"]: e for e in config["HostConfig"]["Ulimits"]}
    assert entries["nofile"] == {"Name": "nofile", "Soft": 1024, "Hard": 65536}
    assert entries["nproc"] == {"Name": "nproc", "Soft": 512, "Hard": 512}


async def test_ulimits_absent_by_default() -> None:
    cls, client, container = mock_docker_factory()
    with patch("axio_tools_docker.sandbox.aiodocker.Docker", cls):
        async with DockerSandbox():
            pass
    config = client._captured_config[0]
    assert "Ulimits" not in config["HostConfig"]


# ---------------------------------------------------------------------------
# tmpfs
# ---------------------------------------------------------------------------


async def test_tmpfs() -> None:
    cls, client, container = mock_docker_factory()
    with patch("axio_tools_docker.sandbox.aiodocker.Docker", cls):
        async with DockerSandbox(tmpfs={"/tmp": "size=128m,mode=1777"}):
            pass
    config = client._captured_config[0]
    assert config["HostConfig"]["Tmpfs"] == {"/tmp": "size=128m,mode=1777"}


async def test_tmpfs_absent_by_default() -> None:
    cls, client, container = mock_docker_factory()
    with patch("axio_tools_docker.sandbox.aiodocker.Docker", cls):
        async with DockerSandbox():
            pass
    config = client._captured_config[0]
    assert "Tmpfs" not in config["HostConfig"]


# ---------------------------------------------------------------------------
# ports
# ---------------------------------------------------------------------------


async def test_ports_bindings() -> None:
    cls, client, container = mock_docker_factory()
    with patch("axio_tools_docker.sandbox.aiodocker.Docker", cls):
        async with DockerSandbox(ports={8080: 8080, 5432: 15432}):
            pass
    config = client._captured_config[0]
    bindings = config["HostConfig"]["PortBindings"]
    assert bindings["8080/tcp"] == [{"HostPort": "8080"}]
    assert bindings["5432/tcp"] == [{"HostPort": "15432"}]
    exposed = config["ExposedPorts"]
    assert "8080/tcp" in exposed
    assert "5432/tcp" in exposed


async def test_ports_absent_by_default() -> None:
    cls, client, container = mock_docker_factory()
    with patch("axio_tools_docker.sandbox.aiodocker.Docker", cls):
        async with DockerSandbox():
            pass
    config = client._captured_config[0]
    assert "PortBindings" not in config["HostConfig"]
    assert "ExposedPorts" not in config


# ---------------------------------------------------------------------------
# platform
# ---------------------------------------------------------------------------


async def test_platform() -> None:
    cls, client, container = mock_docker_factory()
    with patch("axio_tools_docker.sandbox.aiodocker.Docker", cls):
        async with DockerSandbox(platform="linux/amd64"):
            pass
    config = client._captured_config[0]
    assert config["Platform"] == "linux/amd64"


async def test_platform_absent_by_default() -> None:
    cls, client, container = mock_docker_factory()
    with patch("axio_tools_docker.sandbox.aiodocker.Docker", cls):
        async with DockerSandbox():
            pass
    config = client._captured_config[0]
    assert "Platform" not in config


# ---------------------------------------------------------------------------
# extra_hosts
# ---------------------------------------------------------------------------


async def test_extra_hosts() -> None:
    cls, client, container = mock_docker_factory()
    with patch("axio_tools_docker.sandbox.aiodocker.Docker", cls):
        async with DockerSandbox(extra_hosts={"myhost": "1.2.3.4", "other": "5.6.7.8"}):
            pass
    config = client._captured_config[0]
    assert "myhost:1.2.3.4" in config["HostConfig"]["ExtraHosts"]
    assert "other:5.6.7.8" in config["HostConfig"]["ExtraHosts"]


async def test_extra_hosts_absent_by_default() -> None:
    cls, client, container = mock_docker_factory()
    with patch("axio_tools_docker.sandbox.aiodocker.Docker", cls):
        async with DockerSandbox():
            pass
    config = client._captured_config[0]
    assert "ExtraHosts" not in config["HostConfig"]


# ---------------------------------------------------------------------------
# devices
# ---------------------------------------------------------------------------


async def test_devices_full_format() -> None:
    cls, client, container = mock_docker_factory()
    with patch("axio_tools_docker.sandbox.aiodocker.Docker", cls):
        async with DockerSandbox(devices=["/dev/sda:/dev/xvda:r"]):
            pass
    config = client._captured_config[0]
    assert config["HostConfig"]["Devices"] == [
        {"PathOnHost": "/dev/sda", "PathInContainer": "/dev/xvda", "CgroupPermissions": "r"}
    ]


async def test_devices_short_format() -> None:
    """Just the host path - maps to same container path with rwm."""
    cls, client, container = mock_docker_factory()
    with patch("axio_tools_docker.sandbox.aiodocker.Docker", cls):
        async with DockerSandbox(devices=["/dev/net/tun"]):
            pass
    config = client._captured_config[0]
    assert config["HostConfig"]["Devices"] == [
        {"PathOnHost": "/dev/net/tun", "PathInContainer": "/dev/net/tun", "CgroupPermissions": "rwm"}
    ]


async def test_devices_multiple() -> None:
    cls, client, container = mock_docker_factory()
    with patch("axio_tools_docker.sandbox.aiodocker.Docker", cls):
        async with DockerSandbox(devices=["/dev/sda", "/dev/sdb:/dev/xvdb:rw"]):
            pass
    config = client._captured_config[0]
    assert len(config["HostConfig"]["Devices"]) == 2


async def test_devices_absent_by_default() -> None:
    cls, client, container = mock_docker_factory()
    with patch("axio_tools_docker.sandbox.aiodocker.Docker", cls):
        async with DockerSandbox():
            pass
    config = client._captured_config[0]
    assert "Devices" not in config["HostConfig"]


# ---------------------------------------------------------------------------
# dns
# ---------------------------------------------------------------------------


async def test_dns() -> None:
    cls, client, container = mock_docker_factory()
    with patch("axio_tools_docker.sandbox.aiodocker.Docker", cls):
        async with DockerSandbox(dns=["8.8.8.8", "1.1.1.1"]):
            pass
    config = client._captured_config[0]
    assert config["HostConfig"]["Dns"] == ["8.8.8.8", "1.1.1.1"]


async def test_dns_absent_by_default() -> None:
    cls, client, container = mock_docker_factory()
    with patch("axio_tools_docker.sandbox.aiodocker.Docker", cls):
        async with DockerSandbox():
            pass
    config = client._captured_config[0]
    assert "Dns" not in config["HostConfig"]


# ---------------------------------------------------------------------------
# parse_device unit tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (
            "/dev/sda",
            {"PathOnHost": "/dev/sda", "PathInContainer": "/dev/sda", "CgroupPermissions": "rwm"},
        ),
        (
            "/dev/sda:/dev/xvda",
            {"PathOnHost": "/dev/sda", "PathInContainer": "/dev/xvda", "CgroupPermissions": "rwm"},
        ),
        (
            "/dev/sda:/dev/xvda:r",
            {"PathOnHost": "/dev/sda", "PathInContainer": "/dev/xvda", "CgroupPermissions": "r"},
        ),
    ],
)
def test_parse_device(value: str, expected: dict[str, str]) -> None:
    assert parse_device(value) == expected


# ---------------------------------------------------------------------------
# parse_memory / parse_cpus unit tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("256m", 256 * 1024**2),
        ("1g", 1024**3),
        ("512k", 512 * 1024),
        ("1048576", 1048576),
    ],
)
def test_parse_memory(value: str, expected: int) -> None:
    assert parse_memory(value) == expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("1.0", 1_000_000_000),
        ("0.5", 500_000_000),
        ("2.0", 2_000_000_000),
    ],
)
def test_parse_cpus(value: str, expected: int) -> None:
    assert parse_cpus(value) == expected
