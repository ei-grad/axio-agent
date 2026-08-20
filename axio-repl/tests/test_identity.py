from __future__ import annotations

import json
from datetime import datetime
from types import SimpleNamespace

import pytest

from axio_repl._identity import (
    RUNTIME_METADATA_KEY,
    append_runtime_identity_metadata,
    resolve_effective_username,
)
from axio_repl._panel import make_prompt_factory


def test_effective_username_uses_nss_record_not_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("USER", "forged-environment-name")
    username = resolve_effective_username(
        geteuid=lambda: 1001,
        getpwuid=lambda uid: SimpleNamespace(pw_name=f"nss-{uid}"),
    )

    assert username == "nss-1001"


def test_effective_username_falls_back_to_numeric_uid_when_nss_fails() -> None:
    def missing(_uid: int) -> SimpleNamespace:
        raise KeyError

    assert resolve_effective_username(geteuid=lambda: 4242, getpwuid=missing) == "4242"


def test_effective_username_strips_terminal_controls_and_empty_values() -> None:
    hostile = "alice\nadmin\033]0;owned\007"
    username = resolve_effective_username(
        geteuid=lambda: 1001,
        getpwuid=lambda _uid: SimpleNamespace(pw_name=hostile),
    )

    assert username == "alice admin"
    assert "\033" not in username
    assert "\n" not in username
    assert (
        resolve_effective_username(
            geteuid=lambda: 1001,
            getpwuid=lambda _uid: SimpleNamespace(pw_name="\033]0;owned\007"),
        )
        == "1001"
    )


def test_runtime_identity_metadata_is_one_compact_json_data_record() -> None:
    system = append_runtime_identity_metadata("base system", 'alice"\nadmin')
    payload = json.loads(system.rsplit("\n", 1)[-1])

    assert system.startswith("base system\n\n")
    assert system.count(RUNTIME_METADATA_KEY) == 1
    assert payload == {
        RUNTIME_METADATA_KEY: {
            "effective_username": 'alice" admin',
            "kind": "data",
        }
    }


def test_prompt_and_system_metadata_share_the_same_resolved_identity() -> None:
    from prompt_toolkit.formatted_text import to_formatted_text

    username = resolve_effective_username(
        geteuid=lambda: 1001,
        getpwuid=lambda _uid: SimpleNamespace(pw_name="shared-user"),
    )
    prompt = make_prompt_factory(
        username,
        now_provider=lambda: datetime(2026, 8, 20, 9, 7),
    )()
    system = append_runtime_identity_metadata("system", username)

    assert to_formatted_text(prompt) == [("class:repl-prompt", "09:07 shared-user> ")]
    assert system.count('"effective_username":"shared-user"') == 1
