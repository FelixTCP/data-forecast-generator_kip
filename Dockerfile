# ─────────────────────────────────────────────────────────────────────────────
# data-forecast-generator
# Base image: python:3.12-slim  (works on amd64 and arm64; Podman-compatible)
# ─────────────────────────────────────────────────────────────────────────────
FROM python:3.12-slim

# ── System packages + Node.js 20 LTS (needed for GitHub Copilot CLI) ─────────
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        git \
 && curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
 && apt-get install -y --no-install-recommends nodejs \
 && curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg \
      | dd of=/usr/share/keyrings/githubcli-archive-keyring.gpg \
 && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" \
      > /etc/apt/sources.list.d/github-cli.list \
 && apt-get update \
 && apt-get install -y --no-install-recommends gh \
 && rm -rf /var/lib/apt/lists/*

# ── GitHub Copilot CLI ────────────────────────────────────────────────────────
# Pin global npm prefix to /usr/local so the binary lands in /usr/local/bin
# (predictable, guaranteed to be in PATH). Authentication via GITHUB_TOKEN.
RUN npm config set prefix /usr/local \
 && npm install -g @github-copilot/cli \
 && ls /usr/local/bin/copilot \
 && echo "copilot installed: $(copilot --version 2>&1 || true)"

# ── uv (fast Python package manager) ─────────────────────────────────────────
RUN pip install --no-cache-dir uv

WORKDIR /app

# ── Python dependencies — install before copying full source for layer cache ─
COPY pyproject.toml uv.lock ./

# UV_SYSTEM_PYTHON=1  → install into the system Python (no venv needed)
# --no-dev            → skip pytest / ruff / mypy dev extras
# --no-install-project→ only deps, skip installing the package itself
#                       (there is no src/data_forecast_generator in the image)
ENV UV_SYSTEM_PYTHON=1
RUN uv sync --no-dev --no-install-project

# uv sync creates a venv at /app/.venv — put its bin on PATH.
# Also spell out /usr/local/bin explicitly so copilot is always found
# regardless of how the process is launched (shell or exec form).
ENV PATH="/app/.venv/bin:/usr/local/bin:/usr/local/sbin:/usr/bin:/usr/sbin:/bin:/sbin"

# ── Application source ────────────────────────────────────────────────────────
COPY scripts/   ./scripts/
COPY docs/      ./docs/
COPY .github/   ./.github/

# ── Runtime directories (owned by the app user; mounted as volumes at runtime)
RUN mkdir -p artifacts/ui_uploads output

# ── Non-root user for Podman rootless compatibility ───────────────────────────
RUN useradd --uid 1000 --create-home appuser \
 && chown -R appuser:appuser /app
USER appuser

# ── Ports ─────────────────────────────────────────────────────────────────────
# 8501 → Streamlit agent pipeline app
# 8502 → Streamlit inference / XAI app
EXPOSE 8501
EXPOSE 8502

# ── Default: run the single-agent pipeline UI ─────────────────────────────────
# Override with --command to run the inference app instead.
CMD ["streamlit", "run", "scripts/streamlit_single_agent_app.py", \
     "--server.address=0.0.0.0", "--server.port=8501"]
