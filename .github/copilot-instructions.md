# Copilot Instructions for Data Forecast Generator

## Project Overview

This is a Python-based project that automates discovery of data analysis use cases through LLM analysis and implements regression forecasting pipelines. The system is designed in three phases:

1. **PHASE 1**: Template/scaffold generation for analysis projects
2. **PHASE 2**: Quality evaluation of generated MVPs using appropriate metrics
3. **PHASE 3**: Full-stack system with FastAPI server and integrated AI agent

The core workflow: CSV input → LLM analysis → Use case discovery → Regression pipeline development → Model fitting → Results evaluation and reporting.

## Build, Test, and Lint Commands

**Note**: Project is in early phase. Commands should be added as development progresses.

As the project develops, expected tools:
- **Testing**: pytest (for regression pipeline tests)
- **Linting**: black, flake8, or ruff (Python style)
- **ML Framework**: scikit-learn (regression pipelines)
- **LLM Integration**: Likely Claude API or similar

## Architecture & Key Components

### Expected Modules (PHASE 1 focus)

1. **CSV Analysis Module**
   - LLM-based analysis to identify potential regression use cases
   - Detection of time-series data, feature relationships, and business value signals

2. **Regression Pipeline**
   - Implemented with scikit-learn
   - Automated feature engineering and model selection
   - Training and evaluation on identified use cases

3. **Artifact Generation**
   - Fitted model serialization
   - Pipeline configuration export
   - Quality metrics and potential analysis reporting

4. **Quality Metrics & Evaluation** (PHASE 2)
   - Assess goodness-of-fit for regression models
   - Business impact estimation
   - Model quality assessment (possibly LLM-as-Judge)

### Directory Structure (Planned)

```
data-forecast-generator_kip/
├── .github/
│   └── copilot-instructions.md (this file)
├── src/                    # Main source code
│   ├── csv_analyzer/      # LLM-based CSV analysis
│   ├── pipeline/          # Regression pipeline implementation
│   ├── artifacts/         # Model serialization and export
│   └── evaluation/        # Quality metrics and assessment
├── tests/                 # Test suite (pytest)
├── examples/              # Example CSVs and outputs
├── project.toml           # Poetry/packaging config (to be created)
├── README.md
├── project-description.md
└── LICENSE
```

## Key Conventions & Patterns

### Language & Framework
- **Python 3.9+** (assumed for modern type hints)
- **scikit-learn** for regression pipelines
- **LLM API** integration (Claude or similar)
- **FastAPI** for PHASE 3 server (if reached)

### Code Organization
- Keep CSV analysis logic separate from ML pipeline logic
- Use scikit-learn pipelines for reproducibility
- Model artifacts should be versioned and serializable (pickle/joblib)

### CSV Input Handling
- Assume customer CSVs may have quality issues (missing values, wrong types, etc.)
- LLM should identify these issues and suggest solutions
- Document assumptions about expected CSV structure

### Regression Pipeline
- Should be configurable (feature selection, model type, hyperparameters)
- Comprehensive evaluation metrics (R², RMSE, MAE, cross-validation scores)
- Explainability focus (which features drive predictions)

### Testing Strategy
- Unit tests for pipeline components
- Integration tests for end-to-end CSV → Model workflow
- Fixture-based testing with synthetic CSVs for regression models

### Output/Artifacts
- Model should be JSON-serializable config + pickle model file
- Include full evaluation metrics with the artifact
- Generate human-readable analysis report alongside model

## Common Tasks

### When Adding New Analysis Features
1. Implement in `csv_analyzer` module
2. Add tests using synthetic test data
3. Update artifact schema if new metadata needed
4. Document assumptions about data format

### When Updating Regression Pipeline
1. Modify scikit-learn pipeline in `pipeline` module
2. Ensure backward compatibility with old artifacts
3. Add evaluation metrics to track improvement
4. Test on diverse synthetic datasets

### When Creating PHASE 2 Quality Metrics
1. Define metrics that correlate with business value
2. Implement as pluggable evaluation functions
3. Build tests around metric stability and reproducibility

## Important Notes

- This is a **template generation system**, not a one-off analysis tool—design for reusability
- Customer CSVs are the primary input; prioritize robustness and clear error messages
- Model quality is critical; always include cross-validation and residual analysis
- Document all assumptions about data structure and analysis approach
