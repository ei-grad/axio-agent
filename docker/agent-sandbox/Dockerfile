# syntax=docker/dockerfile:1.7

ARG BASE_IMAGE=mcr.microsoft.com/devcontainers/base:trixie
ARG NODE_IMAGE=node:24-bookworm-slim
ARG GO_IMAGE=golang:1.25-bookworm
ARG RUST_IMAGE=rust:1-bookworm
ARG GLAB_IMAGE=registry.gitlab.com/gitlab-org/cli:latest

FROM ${NODE_IMAGE} AS node-runtime
FROM ${GO_IMAGE} AS go-runtime
FROM ${RUST_IMAGE} AS rust-runtime
FROM ${GLAB_IMAGE} AS glab-runtime

FROM ${BASE_IMAGE}

ARG TARGETARCH

ENV DEBIAN_FRONTEND=noninteractive \
    UV_TOOL_BIN_DIR=/usr/local/bin \
    UV_TOOL_DIR=/opt/uv/tools \
    RUSTUP_HOME=/usr/local/rustup \
    CARGO_HOME=/usr/local/cargo \
    GOROOT=/usr/local/go \
    GOPATH=/opt/go \
    JAVA_HOME=/usr/local/openjdk \
    CHROME_BIN=/usr/bin/chromium \
    CHROMIUM_PATH=/usr/bin/chromium \
    PLAYWRIGHT_SKIP_BROWSER_DOWNLOAD=1 \
    PUPPETEER_SKIP_DOWNLOAD=true \
    HF_HUB_DISABLE_TELEMETRY=1 \
    HF_HUB_DISABLE_UPDATE_CHECK=1 \
    DO_NOT_TRACK=1 \
    PATH=/usr/local/cargo/bin:/usr/local/go/bin:/opt/go/bin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update \
    && apt-get install -y --no-install-recommends \
        bash-completion \
        bat \
        build-essential \
        ca-certificates \
        clang \
        cmake \
        chromium \
        chromium-driver \
        chromium-sandbox \
        curl \
        diffutils \
        dnsutils \
        fd-find \
        ffmpeg \
        file \
        fonts-liberation \
        fonts-noto-color-emoji \
        gdb \
        git \
        git-lfs \
        gnupg \
        graphviz \
        ghostscript \
        imagemagick \
        jq \
        less \
        libmagic1 \
        lsof \
        make \
        maven \
        netcat-openbsd \
        ninja-build \
        openjdk-21-jdk-headless \
        openssh-client \
        pandoc \
        patch \
        pkg-config \
        poppler-utils \
        postgresql-client \
        procps \
        psmisc \
        python3 \
        python3-dev \
        python3-pip \
        python3-venv \
        qpdf \
        redis-tools \
        ripgrep \
        rsync \
        shellcheck \
        sqlite3 \
        strace \
        tar \
        tesseract-ocr \
        tree \
        unzip \
        vim-tiny \
        wget \
        xz-utils \
        zip \
        zstd

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg \
        -o /usr/share/keyrings/githubcli-archive-keyring.gpg \
    && chmod go+r /usr/share/keyrings/githubcli-archive-keyring.gpg \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" \
        > /etc/apt/sources.list.d/github-cli.list \
    && apt-get update \
    && apt-get install -y --no-install-recommends gh

COPY --from=node-runtime /usr/local/bin/node /usr/local/bin/node
COPY --from=node-runtime /usr/local/lib/node_modules /usr/local/lib/node_modules
COPY --from=go-runtime /usr/local/go /usr/local/go
COPY --from=rust-runtime /usr/local/cargo /usr/local/cargo
COPY --from=rust-runtime /usr/local/rustup /usr/local/rustup
COPY --from=glab-runtime /usr/bin/glab /usr/local/bin/glab

RUN ln -sf ../lib/node_modules/npm/bin/npm-cli.js /usr/local/bin/npm \
    && ln -sf ../lib/node_modules/npm/bin/npx-cli.js /usr/local/bin/npx \
    && ln -sf ../lib/node_modules/corepack/dist/corepack.js /usr/local/bin/corepack \
    && ln -sf /usr/bin/fdfind /usr/local/bin/fd \
    && ln -sf /usr/bin/batcat /usr/local/bin/bat \
    && ln -sf /usr/lib/jvm/java-21-openjdk-${TARGETARCH} /usr/local/openjdk

RUN --mount=type=cache,target=/root/.npm,sharing=locked \
    npm install --global typescript

RUN --mount=type=bind,from=ghcr.io/astral-sh/uv:latest,source=/uv,target=/bin/uv \
    --mount=type=cache,target=/root/.cache/uv,sharing=locked \
    uv tool install ast-grep-cli \
    && uv tool install csvkit \
    && uv tool install hf \
    && uv tool install kaggle \
    && uv tool install mypy \
    && uv tool install pre-commit \
    && uv tool install pytest \
    && uv tool install ruff

RUN --mount=type=bind,from=ghcr.io/astral-sh/uv:latest,source=/uv,target=/bin/uv \
    --mount=type=cache,target=/root/.cache/uv,sharing=locked \
    uv init --bare --python /usr/bin/python3 /opt/agent-python \
    && uv add --project /opt/agent-python \
        duckdb \
        ipython \
        jupyterlab \
        matplotlib \
        numpy \
        openpyxl \
        pandas \
        pdfplumber \
        polars \
        pyarrow \
        pymupdf \
        pypdf \
        scikit-learn \
        seaborn

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/
COPY python-data /usr/local/bin/python-data
COPY browser-smoke /usr/local/bin/browser-smoke

RUN chmod 0755 /usr/local/bin/browser-smoke /usr/local/bin/python-data \
    && ln -sf /opt/agent-python/.venv/bin/ipython /usr/local/bin/ipython \
    && ln -sf /opt/agent-python/.venv/bin/jupyter /usr/local/bin/jupyter \
    && command -v git gh glab python3 python-data uv node npm go rustc cargo java javac mvn \
        chromium chromedriver pdftotext qpdf tesseract jq rg ast-grep kaggle hf >/dev/null \
    && test -u /usr/lib/chromium/chrome-sandbox \
    && chromium --version \
    && chromedriver --version \
    && python-data -c "import duckdb, pandas, pdfplumber, polars, pyarrow, pymupdf, pypdf, sklearn"

WORKDIR /workspace
