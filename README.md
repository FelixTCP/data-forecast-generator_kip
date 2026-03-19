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

## Architecture

See the session `plan.md` for the detailed implementation plan.

**Core modules**:
- `src/data/` - CSV loading & validation
- `src/analysis/` - LLM-based CSV analysis
- `src/pipeline/` - Feature engineering & regression models
- `src/evaluation/` - Model evaluation & metrics
- `src/artifacts/` - Model serialization

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
