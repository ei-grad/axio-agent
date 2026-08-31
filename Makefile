PACKAGES := axio axio-audio axio-context-sqlite axio-repl axio-responses axio-sse \
            axio-tools-agents axio-tools-docker axio-tools-local axio-tools-mcp \
            axio-transport-anthropic axio-transport-codex \
            axio-transport-google axio-transport-openai \
            axio-tui axio-tui-guards examples/gas_town examples/agent_swarm \
            examples/realtime_smoke examples/realtime_chat
SANDBOX_IMAGE ?= axio-agent-sandbox:standard
SANDBOX_RUNTIME_USER ?= 1000:1000

.PHONY: $(PACKAGES) all pytest linter typing test tests test-docs test-tutorial docs-html \
        sandbox-image sandbox-image-smoke

all: linter typing pytest test-docs test-tutorial docs-html

linter:
	@for pkg in $(PACKAGES); do uv run --directory $$pkg ruff check . && uv run --directory $$pkg ruff format --check . || exit 1; done
	@uv run --directory docs ruff check ../examples/tutorial
	@uv run --directory docs ruff format --check ../examples/tutorial

typing pytest: $(PACKAGES)

$(PACKAGES):
	@uv run --directory $@ mypy .
	@uv run --directory $@ pytest -q

test-docs:
	@uv run --directory docs pytest -q .

test-tutorial:
	@uv run --directory docs pytest -q ../examples/tutorial

docs-html:
	@$(MAKE) -C docs check-html

test: pytest
tests: pytest

sandbox-image:
	docker build --tag $(SANDBOX_IMAGE) docker/agent-sandbox

sandbox-image-smoke: sandbox-image
	docker run --rm --network none --cap-drop ALL --security-opt no-new-privileges \
		--user $(SANDBOX_RUNTIME_USER) --env HOME=/tmp/axio-browser-home \
		$(SANDBOX_IMAGE) browser-smoke
	AXIO_STANDARD_SANDBOX_IMAGE=$(SANDBOX_IMAGE) \
		uv run --directory axio-tools-docker pytest -q tests/test_integration.py -k standard_image_defaults_to_bash
