# Quick Start

## 30-Second Setup

```bash
# Enter dev environment
nix develop

# Install runtime dependencies
uv sync --extra dev
uv pip install pandas statsmodels scipy pyarrow

# Verify everything works
pytest --version && ruff --version
```

Done! You now have:
- ✅ Python 3.12 environment
- ✅ All dependencies (scikit-learn, polars, anthropic, etc.)
- ✅ Testing framework (pytest)
- ✅ Linter (ruff) + formatter
- ✅ Run-based pipeline tooling

## Common Commands

```bash
# Testing
pytest                                    # Run all tests
pytest tests/test_data_loader.py         # Specific file
pytest tests/test_data_loader.py::test_valid_csv -v  # Specific test with verbose

# Code Quality
ruff check tests scripts                  # Lint
ruff format tests scripts                 # Auto-format
mypy .                                    # Type check

# Coverage / tests
pytest                                   # Run current tests
```

## Run the Pipeline

```bash
RUN_ID="singleagent_$(date -u +%Y%m%dT%H%M%SZ)"
OUT="output/$RUN_ID"
mkdir -p "$OUT/code"
cp output/manual_run_001/code/*.py "$OUT/code/"

uv run python "$OUT/code/orchestrator.py" \
  --csv-path data/appliances_energy_prediction.csv \
  --target-column appliances \
  --output-dir "$OUT" \
  --run-id "$RUN_ID" \
  --split-mode auto
```

For the full current workflow, read `CURRENT_SYSTEM_DOCUMENTATION.md`.

## Development Cycle

```bash
# Feature branch
git checkout -b feat/my-feature

# Code + test
# Edit the relevant step script under output/<RUN_ID>/code/

# Verify locally
pytest tests/test_module.py
ruff format tests scripts

# Commit & push
git commit -m "feat: description"
git push origin feat/my-feature

# Create PR on GitHub
# → GitHub Actions runs automatically
# → Review & merge
```

## Files to Know

| File | Purpose |
|------|---------|
| `CURRENT_SYSTEM_DOCUMENTATION.md` | Current run-based system documentation |
| `IMPLEMENTATION_OUTLINE.md` | Historical implementation outline |
| `.github/copilot-instructions.md` | Architecture & conventions |
| `SETUP_SUMMARY.md` | Detailed setup reference |
| `.github/workflows/tests.yml` | CI/CD pipeline |
| `pyproject.toml` | Project configuration |
| `flake.nix` | NixOS dev environment |

## Example: Re-run Training Step

```bash
uv run python output/<RUN_ID>/code/step_13_training.py \
  --output-dir output/<RUN_ID> \
  --run-id <RUN_ID> \
  --split-mode auto \
  --target-column appliances
```

## Configuration

Set up your environment:
```bash
cp .env.example .env
# Edit .env and add ANTHROPIC_API_KEY from https://console.anthropic.com
```

## Getting Help

- `README.md` - Quick reference
- `.github/copilot-instructions.md` - Architecture details
- `CURRENT_SYSTEM_DOCUMENTATION.md` - Full current workflow
- `IMPLEMENTATION_OUTLINE.md` - Historical module outline
- Docstrings in code - Implementation details
