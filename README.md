# Data Forecast Generator

Automated discovery and development of regression forecasting models from customer CSV data using LLM-driven analysis.

## Quick Start

### Prerequisites

- NixOS with `nix flake` support
- Git

### Setup (NixOS)

```bash
# Enter development environment
nix flake update
nix develop

# Install project dependencies (Nix shell provides tooling)
uv pip install -e .

# Verify installation
pytest --version
ruff --version
```

### Setup (Other Systems)

Python 3.12+ with pip:

```bash
python -m venv .venv
source .venv/bin/activate
uv pip install -e ".[dev]"
```

## Development

### Running Tests

```bash
# All tests
pytest

# Specific test file
pytest tests/test_data_loader.py

# Specific test
pytest tests/test_data_loader.py::test_valid_csv

# With coverage
pytest --cov=src --cov-report=html
```

### Code Quality

```bash
# Lint
ruff check src tests

# Format
ruff format src tests

# Type check
mypy src
```

### Local Development Loop

```bash
# Make changes, then:
pytest tests/test_module.py -v
ruff format src tests
git add . && git commit -m "feat: ..."
```

### Run the MVP pipeline (`forecast`)

```bash
# Install editable package (once per env)
uv pip install -e ".[dev]"

# Execute full AGENTIC pipeline (Copilot CLI is called for each step)
uv run forecast --csv ./data/your_file.csv --target-column your_target --output-dir ./artifacts

# Optional split strategy
uv run forecast --csv ./data/your_file.csv --target-column your_target --split-mode auto

# Budget-friendly defaults (small/fast model)
uv run forecast --csv ./data/your_file.csv --target-column your_target --budget-mode low

# Explicit model override
uv run forecast --csv ./data/your_file.csv --target-column your_target --copilot-model gpt-5-mini --reasoning-effort low

# Reuse generated code from previous identical command fingerprint
uv run forecast --csv ./data/your_file.csv --target-column your_target --continue
```

This mode is instruction-first: step logic is executed by Copilot CLI prompts using
`docs/agentic-pipeline/` contracts and `docs/pipeline-framework/` step guidance.
Pipeline progress is visualized with `tqdm` while steps execute.
Required run artifacts now include `model.joblib` for the selected model.

### Verbose debugging artifacts
Each run now writes rich debug artifacts under `artifacts/<run_id>/debug/`:
- `run_context.json` (full runtime inputs/profile)
- `<step>/prompt.md` (exact prompt sent)
- `<step>/response.md` (raw Copilot response)
- `<step>/meta.json` (model/reasoning metadata)

It also writes `code_audit.json` and a persistent hashed code workspace:
- `artifacts/.agent_code/<command_hash>/workspace/` (generated code)
- `artifacts/.agent_code/<command_hash>/snapshots/` (pre-reset backups)

### Direct `copilot -p` workflow (without CLI orchestrator)
If you want to debug prompt-by-prompt manually:

```bash
scripts/run_agentic_prompts.sh \
  ./data/appliances_energy_prediction.csv \
  appliances \
  ./artifacts/manual_debug_run
```

Prompt templates are in `prompts/agentic/` and rendered prompts/responses are saved to `OUTPUT_DIR/debug/`.
Generated step code is stored under a hashed directory in `artifacts/.agent_code/<command_hash>/workspace`
and audited per run in `code_audit.json`.

### Quick inference check for `model.joblib`

```bash
# Predict with model artifact
uv run python scripts/infer_model.py \
  --model ./artifacts/<run_id>/model.joblib \
  --csv ./data/appliances_energy_prediction.csv \
  --target-column appliances \
  --output-csv ./artifacts/<run_id>/predictions.csv
```

## Architecture

See the session `plan.md` for the detailed implementation plan.

**Core modules**:

- `src/data/` - CSV loading & validation
- `src/analysis/` - LLM-based CSV analysis
- `src/pipeline/` - Feature engineering & regression models
- `src/evaluation/` - Model evaluation & metrics
- `src/artifacts/` - Model serialization

**Pipeline framework docs (Issue #8, sub-issues #10-#16):**

- `docs/agentic-pipeline/contracts.md` - Runtime contracts, file layout, and resume rules
- `docs/agentic-pipeline/step-prompts.md` - Runtime Reason→Code→Validate wrappers per step
- `docs/pipeline-framework/10-csv-read-cleansing.md` ... `16-result-presentation.md` - Canonical per-step logic

## Configuration

Copy `.env.example` to `.env` and update with your API keys:

```bash
cp .env.example .env
# Edit .env and add ANTHROPIC_API_KEY
```

## Project Phases

- **Phase 1 (MVP)**: Core pipeline - CSV → Analysis → Model → Artifact
- **Phase 2**: Quality metrics & LLM-as-Judge evaluation
- **Phase 3**: Full-stack system with FastAPI + Web UI

## CI/CD

GitHub Actions runs on every push:

- Lint with ruff
- Type check with mypy
- Test with pytest (Python 3.12)
- Coverage reporting to Codecov

See `.github/workflows/tests.yml`

## Contributing

1. Create feature branch: `git checkout -b feat/my-feature`
2. Implement with tests
3. Run `pytest` and `ruff format src tests`
4. Push and create PR
5. CI/CD validates automatically

## License

See LICENSE file
