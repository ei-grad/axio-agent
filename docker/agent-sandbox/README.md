# Standard agent sandbox image

This directory builds the local default environment for `axio-repl`. It is not
published by the project. The current build has been smoke-tested on
`linux/amd64`; other platforms are not claimed as tested.

```bash
make sandbox-image
axio-repl --sandbox docker \
  --sandbox-memory 4g \
  --sandbox-cpus 2
```

The REPL does not try to pull the default `axio-agent-sandbox:standard` tag. If
it has not been built locally, startup reports the `make sandbox-image` command.
Explicit `--sandbox-image` alternatives retain pull-on-missing behavior.

The image is based on the Debian Trixie variant of
`mcr.microsoft.com/devcontainers/base` and adds:

- Python 3 and uv, plus a data-analysis environment exposed as `python-data`,
  IPython, and Jupyter;
- Node.js/npm, Go, Rust/Cargo, OpenJDK 21, and Maven;
- Debian Chromium, its matching `chromium-driver` and setuid sandbox, plus
  common web fonts;
- Git, Git LFS, GitHub CLI (`gh`), and GitLab CLI (`glab`);
- compilers, CMake/Ninja, ripgrep, ast-grep, jq, ShellCheck, database clients,
  and common archive/process/network diagnostics;
- Poppler, qpdf, Ghostscript, Tesseract, Pandoc, ImageMagick, FFmpeg, PyMuPDF,
  pypdf, and pdfplumber;
- Kaggle (`kaggle`) and Hugging Face (`hf`) CLIs.

Debian's system Python is externally managed under PEP 668. Use `uv add`, `uv
sync`, or `uvx` rather than global `pip install`. `python-data` selects the
baked analysis environment without changing the system interpreter.

Base/runtime images are build arguments so an operator can pin them to immutable
digests without editing the Dockerfile:

```bash
docker build \
  --build-arg BASE_IMAGE=mcr.microsoft.com/devcontainers/base:trixie@sha256:... \
  --build-arg NODE_IMAGE=node@sha256:... \
  --build-arg GO_IMAGE=golang@sha256:... \
  --build-arg RUST_IMAGE=rust@sha256:... \
  --build-arg GLAB_IMAGE=registry.gitlab.com/gitlab-org/cli@sha256:... \
  -t axio-agent-sandbox:standard docker/agent-sandbox
```

The defaults intentionally track supported runtime lines rather than claiming a
reproducible release. Production builds should pin every image argument and
record the resulting image digest and SBOM in the deployment system.

## Browser automation

`chromium` and `chromedriver` are on `PATH`; `CHROME_BIN` and `CHROMIUM_PATH`
both point to `/usr/bin/chromium`. `PLAYWRIGHT_SKIP_BROWSER_DOWNLOAD=1` and
`PUPPETEER_SKIP_DOWNLOAD=true` are also set so installing those libraries does
not silently fetch a second browser.

Use the system executable explicitly when a framework does not honor those
variables:

```javascript
const browser = await playwright.chromium.launch({
  executablePath: process.env.CHROMIUM_PATH,
  args: ["--no-sandbox", "--disable-dev-shm-usage"],
});
```

```python
import os
from selenium import webdriver

options = webdriver.ChromeOptions()
options.binary_location = os.environ["CHROME_BIN"]
options.add_argument("--no-sandbox")
options.add_argument("--disable-dev-shm-usage")
driver = webdriver.Chrome(options=options)  # finds /usr/bin/chromedriver on PATH
```

For Cypress, select it with `npx cypress run --browser "$CHROME_BIN"`.

`make sandbox-image-smoke` builds the image, prints both installed versions,
renders a local `data:` page headlessly as fixed non-root UID 1000, and verifies
that the Docker shell tool defaults to Bash. Docker's default seccomp profile
blocks the namespace transition required by Chromium's nested sandbox, while
the smoke container's `no-new-privileges` setting independently prevents the
installed setuid sandbox from elevating. The browser process therefore uses
`--no-sandbox`; the surrounding test container still has no network and drops
all capabilities. Do not use that flag for untrusted pages: provide a container
security profile that supports Chromium's installed sandbox instead. The smoke
also uses `--disable-dev-shm-usage` because Docker's default `/dev/shm` is
small; give browser-heavy tests a larger `shm_size` and omit that performance
tradeoff.
