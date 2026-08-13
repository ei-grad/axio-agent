from __future__ import annotations

from axio.events import TextDelta, ToolInputDelta, ToolUseStart
from axio_tools_agents.runtime import (
    AgentEventEnvelope,
    ExecutionMode,
    RuntimeEvent,
    SessionEventHub,
    new_turn_identity,
)

from axio_repl import ReplRenderer, render_runtime_event
from axio_repl._multiplexer import DisplayMode


async def test_display_filtering_does_not_change_hub_events() -> None:
    hub = SessionEventHub(session_id="session")
    renderer = ReplRenderer(display_mode=DisplayMode.ACTIVE_ONLY)
    observed: list[AgentEventEnvelope] = []

    async def render(envelope: AgentEventEnvelope) -> None:
        await render_runtime_event(renderer, envelope)

    async def semantic_subscriber(envelope: AgentEventEnvelope) -> None:
        observed.append(envelope)

    hub.subscribe(render)
    hub.subscribe(semantic_subscriber)
    identity = new_turn_identity(
        agent_id="hidden",
        parent_agent_id="main",
        execution_mode=ExecutionMode.BACKGROUND,
    )

    events: list[RuntimeEvent] = [
        TextDelta(index=0, delta="hidden prose"),
        ToolUseStart(index=0, tool_use_id="call", name="shell"),
        ToolInputDelta(index=0, tool_use_id="call", partial_json="{}"),
    ]
    for event in events:
        await hub.publish_for(identity, event)

    assert [envelope.seq for envelope in observed] == [1, 2, 3]
    assert [envelope.event for envelope in observed] == events
