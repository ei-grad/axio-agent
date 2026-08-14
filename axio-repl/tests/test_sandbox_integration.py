from __future__ import annotations

import os
from contextlib import AsyncExitStack
from pathlib import Path

import aiodocker
import pytest
from axio import Tool

from axio_repl import _sandbox


async def _noop() -> str:
    return ""


async def _exercise_runtime_identity(tmp_path: Path, image: str) -> None:
    host_file = tmp_path / "written-by-tool.txt"
    async with AsyncExitStack() as stack:
        tools, _description, tool_root, _note = await _sandbox.build_tools(
            stack,
            [Tool(name="shell", handler=_noop), Tool(name="write_file", handler=_noop)],
            "docker",
            image,
            tmp_path,
        )
        assert tool_root == tmp_path
        shell_tool = next(tool for tool in tools if tool.name == "shell")
        write_tool = next(tool for tool in tools if tool.name == "write_file")
        result = await shell_tool(
            command='printf \'%s\\n\' "$(id -u)" "$(id -g)" "$(pwd -P)" "$HOME"; '
            'getent passwd "$(id -u)" >/dev/null; getent group "$(id -g)" >/dev/null; '
            'touch "$HOME/cli-state"'
        )
        assert result.splitlines() == [str(os.getuid()), str(os.getgid()), str(tmp_path), _sandbox.SANDBOX_HOME]
        await write_tool(path=host_file.name, content="owned by the invoking user\n")

    assert host_file.read_text(encoding="utf-8") == "owned by the invoking user\n"
    assert host_file.stat().st_uid == os.getuid()
    assert host_file.stat().st_gid == os.getgid()


async def _local_image_is_present(image: str) -> bool:
    async with aiodocker.Docker() as client:
        try:
            await client.images.inspect(image)
        except aiodocker.exceptions.DockerError as exc:
            if exc.status == 404:
                return False
            raise
    return True


@pytest.mark.asyncio
async def test_repl_sandbox_preserves_local_identity_and_project_path(tmp_path: Path) -> None:
    if not _sandbox.docker_available():
        pytest.skip("Docker daemon is unavailable")
    if not await _local_image_is_present("python:3.12-slim"):
        pytest.skip("python:3.12-slim is not available locally")
    await _exercise_runtime_identity(tmp_path, "python:3.12-slim")


@pytest.mark.asyncio
async def test_local_default_image_runtime_identity_when_built(tmp_path: Path) -> None:
    if not _sandbox.docker_available():
        pytest.skip("Docker daemon is unavailable")
    if not await _local_image_is_present(_sandbox.DEFAULT_SANDBOX_IMAGE):
        pytest.skip("local default sandbox image is not built")
    await _exercise_runtime_identity(tmp_path, _sandbox.DEFAULT_SANDBOX_IMAGE)
