# 🚀 Quick Start

## 30-Second Setup

```bash
cd /home/felix/Uni/data-forecast-generator_kip

# Enter dev environment
nix develop

# Install project dependencies (Nix shell provides tooling)
uv pip install -e .

# Verify everything works
pytest --version && ruff --version
```

Done! You now have:
- ✅ Python 3.12 environment
- ✅ All dependencies (scikit-learn, polars, anthropic, etc.)
- ✅ Testing framework (pytest)
- ✅ Linter (ruff) + formatter
- ✅ GitHub Copilot CLI

## Common Commands

```bash
# Testing
pytest                                    # Run all tests
pytest tests/test_data_loader.py         # Specific file
pytest tests/test_data_loader.py::test_valid_csv -v  # Specific test with verbose

# Code Quality
ruff check src tests                      # Lint
ruff format src tests                     # Auto-format
mypy src                                  # Type check

# Coverage
pytest --cov=src --cov-report=html       # Generate HTML coverage report
```

## Next: Start Implementing

1. **Read the plan**: `/home/felix/.copilot/session-state/0044af14-0c57-421e-b7b4-3e1da6914e94/plan.md`
2. **Check implementation outline**: `IMPLEMENTATION_OUTLINE.md` (in this repo)
3. **Start with Module 1**: `src/data/loader.py`
   - Write tests first in `tests/test_data_loader.py`
   - Implement classes according to outline
   - Run `pytest tests/test_data_loader.py`
   - Submit PR when complete

## Development Cycle

```bash
# Feature branch
git checkout -b feat/my-feature

# Code + test
# Edit src/module.py and tests/test_module.py

# Verify locally
pytest tests/test_module.py
ruff format src tests

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
| `plan.md` | Full implementation roadmap (15 modules) |
| `IMPLEMENTATION_OUTLINE.md` | Code structure & class signatures |
| `.github/copilot-instructions.md` | Architecture & conventions |
| `SETUP_SUMMARY.md` | Detailed setup reference |
| `.github/workflows/tests.yml` | CI/CD pipeline |
| `pyproject.toml` | Project configuration |
| `flake.nix` | NixOS dev environment |

## Example: First Implementation

**Goal**: Implement `DataLoader` class

```bash
# 1. Create the module
touch src/data/__init__.py src/data/loader.py

# 2. Write tests (TDD)
# Edit tests/test_data_loader.py with:
#   - test_valid_csv()
#   - test_missing_values()
#   - test_malformed_csv()
# Use fixtures from tests/fixtures/sample_data.py

# 3. Implement
# Edit src/data/loader.py:
#   - class LoadedDataFrame
#   - class DataLoader
#   - method load_csv()

# 4. Test locally
pytest tests/test_data_loader.py -v

# 5. Lint & format
ruff format src tests

# 6. Commit
git add .
git commit -m "feat: implement data loader with validation"

# 7. Push & PR
git push origin feat/data-loader
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
- `plan.md` - Full specifications
- `IMPLEMENTATION_OUTLINE.md` - Code structure & signatures
- Docstrings in code - Implementation details
