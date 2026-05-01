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
 && rm -rf /var/lib/apt/lists/*

# ── GitHub Copilot CLI ────────────────────────────────────────────────────────
# nixpkgs 'github-copilot-cli' maps to the @github-copilot/cli npm package.
# Authentication requires GITHUB_TOKEN (with Copilot permission) at runtime.
RUN npm install -g @github-copilot/cli \
 && copilot --version || true

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

# uv sync creates a venv at /app/.venv — put its bin on PATH
ENV PATH="/app/.venv/bin:$PATH"

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
