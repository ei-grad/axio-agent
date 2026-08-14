"""Settings screen for the llama.cpp transport."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Static

from axio_transport_openai.llamacpp import LLAMA_CPP_BASE_URL


class LlamaCppSettingsScreen(ModalScreen[dict[str, str] | None]):
    """Editable llama.cpp URL and optional API key."""

    BINDINGS = [Binding("escape", "cancel", "Cancel")]
    CSS = """
    LlamaCppSettingsScreen { align: center middle; }
    #llamacpp-settings {
        width: 70; height: auto; border: heavy $accent;
        background: $panel; padding: 1 2;
    }
    .field-label { margin-top: 1; }
    .settings-buttons { height: auto; margin-top: 1; }
    .settings-buttons Button { margin: 0 1; }
    """
    DEFAULT_BASE_URL = LLAMA_CPP_BASE_URL

    def __init__(self, settings: dict[str, str]) -> None:
        super().__init__()
        self._settings = settings

    def compose(self) -> ComposeResult:
        with Container(id="llamacpp-settings"):
            yield Static("[bold]llama.cpp Settings[/]")
            yield Static("OpenAI-compatible base URL:", classes="field-label")
            yield Input(
                value=self._settings.get("base_url", self.DEFAULT_BASE_URL),
                id="base-url",
            )
            yield Static("API Key (optional; defaults to LLAMA_API_KEY):", classes="field-label")
            yield Input(
                value=self._settings.get("api_key", ""),
                id="api-key",
                password=True,
            )
            with Horizontal(classes="settings-buttons"):
                yield Button("Save", id="btn-save", variant="primary")
                yield Button("Cancel", id="btn-cancel")

    def on_mount(self) -> None:
        self.query_one("#base-url", Input).focus()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "btn-save":
            result = {
                "base_url": self.query_one("#base-url", Input).value.strip(),
                "api_key": self.query_one("#api-key", Input).value.strip(),
            }
            self.dismiss(result)
        else:
            self.dismiss(None)

    def action_cancel(self) -> None:
        self.dismiss(None)
