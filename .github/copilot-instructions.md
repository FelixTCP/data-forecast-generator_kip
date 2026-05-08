# Copilot Instructions for Data Forecast Generator

## Project Overview

This project runs a Docker-first CSV forecasting workflow. Users interact with a
Streamlit UI; the UI invokes GitHub Copilot CLI inside the container; Copilot
runs the custom `Single Agent Pipeline` agent and writes run artifacts.

Core workflow:

```text
Browser -> Streamlit -> Copilot CLI -> Single Agent Pipeline -> output/<RUN_ID>/
```

## Runtime Model

- Docker Compose is the standard development and runtime entrypoint.
- The training UI lives in `scripts/streamlit_single_agent_app.py`.
- The inference/XAI UI lives in `scripts/streamlit_inference_app.py`.
- The custom agent lives in `.github/agents/Single Agent Pipeline.agent.md`.
- Pipeline contracts and step specs live under `docs/`.
- Generated runtime code is written per run to `output/<RUN_ID>/code/`.
- There is no versioned `src/data_forecast_generator` package or local Python CLI.

## Build and Run Commands

```bash
cp .env.example .env
docker compose up --build
```

Open:

- Training UI: `http://localhost:8501`
- Inference UI: `http://localhost:8502`

For local setup convenience:

```bash
scripts/setup.sh
```

## Key Conventions

- Keep Streamlit as the user-facing orchestration layer.
- Keep Copilot CLI execution inside Docker for reproducibility.
- Do not add a monolithic checked-in pipeline implementation.
- Generated step scripts belong under `output/<RUN_ID>/code/`.
- Do not reintroduce the old `src/` package layout unless the project explicitly
  returns to a package-based architecture.

## Testing Strategy

- CI should verify that the Docker image builds and imports the runtime
  dependencies needed by the Streamlit apps.
- Do not run an authenticated Copilot pipeline in CI; that requires token access,
  network availability, and model usage.
