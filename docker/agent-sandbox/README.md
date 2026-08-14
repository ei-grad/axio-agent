# Standard agent sandbox image

This directory builds a local agent environment. It is not published by the
project, so choose the local tag explicitly when starting `axio-repl`. The
current build has been smoke-tested on `linux/amd64`; other platforms are not
claimed as tested.

```bash
make sandbox-image
axio-repl --sandbox docker \
  --sandbox-image axio-agent-sandbox:standard \
  --sandbox-memory 4g \
  --sandbox-cpus 2
```

The image is based on `mcr.microsoft.com/devcontainers/base:3-noble` and adds:

- Python 3 and uv, plus a data-analysis environment exposed as `python-data`,
  IPython, and Jupyter;
- Node.js/npm, Go, Rust/Cargo, OpenJDK 21, and Maven;
- Git, Git LFS, GitHub CLI (`gh`), and GitLab CLI (`glab`);
- compilers, CMake/Ninja, ripgrep, ast-grep, jq, ShellCheck, database clients,
  and common archive/process/network diagnostics;
- Poppler, qpdf, Ghostscript, Tesseract, Pandoc, ImageMagick, FFmpeg, PyMuPDF,
  pypdf, and pdfplumber;
- Kaggle (`kaggle`) and Hugging Face (`hf`) CLIs.

Ubuntu's system Python is externally managed under PEP 668. Use `uv add`, `uv
sync`, or `uvx` rather than global `pip install`. `python-data` selects the
baked analysis environment without changing the system interpreter.

Base/runtime images are build arguments so an operator can pin them to immutable
digests without editing the Dockerfile:

```bash
docker build \
  --build-arg BASE_IMAGE=mcr.microsoft.com/devcontainers/base@sha256:... \
  --build-arg NODE_IMAGE=node@sha256:... \
  --build-arg GO_IMAGE=golang@sha256:... \
  --build-arg RUST_IMAGE=rust@sha256:... \
  --build-arg GLAB_IMAGE=registry.gitlab.com/gitlab-org/cli@sha256:... \
  -t axio-agent-sandbox:standard docker/agent-sandbox
```

The defaults intentionally track supported runtime lines rather than claiming a
reproducible release. Production builds should pin every image argument and
record the resulting image digest and SBOM in the deployment system.
