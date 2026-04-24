# Implementation Outline - Phase 1 MVP

Historical module outline. The current working system is run-based and documented in
`CURRENT_SYSTEM_DOCUMENTATION.md`.

## 1. Data Loading & Validation
**File**: `src/data/loader.py`

```python
class DataLoader:
    def load_csv(filepath: str) -> LoadedDataFrame:
        """Load and validate CSV, return wrapped DataFrame with metadata"""
        
class LoadedDataFrame:
    df: pl.DataFrame
    shape: tuple
    dtypes: dict
    quality_score: float  # 0-100
    issues: list[str]
    suggestions: list[str]
```

**Tests**: `tests/test_data_loader.py`
- Valid CSV loading
- Missing values handling
- Type inference
- Malformed CSV rejection
- Edge cases (empty, single column, all NaN)

---

## 2. LLM-Based CSV Analysis
**File**: `src/analysis/csv_analyzer.py`

```python
class CSVAnalyzer:
    def __init__(api_key: str):
        self.client = Anthropic(api_key)
        
    def analyze(df: LoadedDataFrame, context: str = "") -> AnalysisReport:
        """Use Claude to analyze CSV and suggest regression targets"""
        
class UseCase:
    target_column: str
    target_type: str  # "numeric", "time_series", etc.
    potential_features: list[str]
    business_rationale: str
    confidence_score: float
    
class AnalysisReport:
    use_cases: list[UseCase]  # Ranked by business value
    data_issues: list[str]
    recommendations: list[str]
```

**Tests**: `tests/test_csv_analyzer.py`
- Mock Claude API responses
- UseCase ranking logic
- Report generation
- Error handling for API failures

---

## 3. Feature Engineering
**File**: `src/pipeline/feature_engineer.py`

```python
class FeatureEngineer:
    def engineer(
        df: pl.DataFrame,
        target: str,
        feature_config: dict = None
    ) -> tuple[pl.DataFrame, FeatureMetadata]:
        """Handle missing values, scaling, categorical encoding"""
        
class FeatureMetadata:
    feature_names: list[str]
    scaler_config: dict
    categorical_mapping: dict
    transformations_applied: list[str]
    
class FeatureSelector:
    def select_features(X: pl.DataFrame, y: pl.Series) -> list[str]:
        """Correlation-based or variance-based feature selection"""
```

**Tests**: `tests/test_feature_engineer.py`
- Handle NaN values
- Scale numerical features
- Encode categorical features
- Feature selection consistency
- Edge cases

---

## 4. Regression Model Pipeline
**File**: `src/pipeline/regressor.py`

```python
class RegressionPipeline:
    def __init__(model_type: str = "auto"):
        """Initialize with Linear, Ridge, Lasso, RF, or GBM"""
        
    def train(
        X: pl.DataFrame,
        y: pl.Series,
        test_size: float = 0.2
    ) -> TrainedModel:
        """Fit model with cross-validation"""
        
    def get_feature_importance(self) -> dict:
        """Return feature importance scores"""

class TrainedModel:
    model: sklearn model
    cv_scores: dict
    feature_importance: dict
    model_params: dict
    
class ModelComparison:
    def compare_models(X, y) -> list[TrainedModel]:
        """Train multiple models and rank by R²"""
```

**Tests**: `tests/test_regressor.py`
- Model training
- Cross-validation
- Feature importance extraction
- Prediction consistency

---

## 5. Model Evaluation
**File**: `src/evaluation/metrics.py`

```python
class ModelEvaluator:
    def evaluate(
        model: TrainedModel,
        X_test: pl.DataFrame,
        y_test: pl.Series
    ) -> EvaluationReport:
        """Calculate all metrics and diagnostics"""

class EvaluationReport:
    r_squared: float
    rmse: float
    mae: float
    mape: float
    cv_mean: float
    cv_std: float
    residuals: dict
    quality_score: float  # 0-100
    warnings: list[str]
    recommendations: list[str]
```

**Tests**: `tests/test_evaluation.py`
- Metric calculations
- Residual analysis
- Edge cases

---

## 6. Artifact Generation
**File**: `src/artifacts/generator.py`

```python
class ArtifactGenerator:
    def generate(
        model: TrainedModel,
        evaluation: EvaluationReport,
        metadata: dict,
        output_dir: str
    ) -> Artifact:
        """Save all model artifacts"""

class Artifact:
    path: str
    config: dict
    model_file: str
    metadata_file: str
    report_file: str
```

**Artifact Structure**:
```
artifact_2024-03-19_123456/
├── config.json          # Model params, features, scaling info
├── model.joblib         # Serialized sklearn model
├── metadata.json        # Target, features, data shape
├── evaluation.json      # R², RMSE, MAE, CV scores
├── report.md            # Human-readable summary
└── inference.py         # Helper script for predictions
```

**File**: `src/artifacts/loader.py`
```python
class ArtifactLoader:
    def load(artifact_path: str) -> LoadedArtifact:
        """Load artifact and prepare for inference"""
        
    def predict(new_data: pl.DataFrame) -> pl.Series:
        """Make predictions on new data"""
```

**Tests**: `tests/test_artifacts.py`
- Serialization round-trip
- Loading and prediction
- Version compatibility

---

## 7. Orchestration
**File**: `src/orchestrator.py`

```python
class Pipeline:
    def __init__(config_path: str):
        self.config = load_config(config_path)
        
    def run(csv_path: str, output_dir: str) -> PipelineResult:
        """Execute: Load → Analyze → Engineer → Train → Evaluate → Save"""
        
class PipelineResult:
    success: bool
    artifact_path: str
    metrics: dict
    errors: list[str]
    warnings: list[str]
    execution_time: float
```

**Tests**: `tests/test_orchestrator.py` (integration)
- End-to-end CSV → Artifact
- Error handling
- Logging verification

---

## 8. CLI Entry Point
**File**: `src/cli/main.py`

```bash
# Usage
forecast-gen analyze <csv_path> [--output-dir] [--model-type] [--api-key]

# Example
forecast-gen analyze data.csv --output-dir ./artifacts --model-type auto
```

```python
def main():
    args = parse_args()
    pipeline = Pipeline(config_file="./config.yaml")
    result = pipeline.run(args.csv, args.output_dir)
    
    if result.success:
        print(f"✅ Artifact saved: {result.artifact_path}")
        print(f"R²: {result.metrics['r_squared']:.3f}")
    else:
        print(f"❌ Error: {result.errors[0]}")
```

**Tests**: `tests/test_cli.py`
- Argument parsing
- Exit codes
- Output format

---

## Testing Fixtures

**File**: `tests/fixtures/sample_data.py`
```python
def synthetic_regression_data() -> tuple[pl.DataFrame, pl.Series]:
    """Generate synthetic CSV for testing"""
    
def synthetic_timeseries_data() -> pl.DataFrame:
    """Time series data for regression"""
```

**File**: `tests/fixtures/mock_responses.py`
```python
def mock_claude_analysis() -> dict:
    """Mock Claude API response for testing"""
```

---

## Development Checklist

- [ ] Implement DataLoader with 80%+ test coverage
- [ ] Implement CSVAnalyzer with mocked Claude
- [ ] Implement FeatureEngineer with test fixtures
- [ ] Implement RegressionPipeline with multiple models
- [ ] Implement ModelEvaluator with comprehensive metrics
- [ ] Implement ArtifactGenerator & Loader
- [ ] Implement Orchestrator (integration tests)
- [ ] Implement CLI interface
- [ ] Run `pytest --cov` - target 80%+ coverage
- [ ] Run `ruff check` - fix all linting issues
- [ ] Commit & push → GitHub Actions validates

---

## Key Imports (Standard Pattern)

```python
import polars as pl
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from pydantic import BaseModel, Field
import anthropic
import joblib
import json
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any
from dataclasses import dataclass
from datetime import datetime
import logging
```

---

## Git Workflow

```bash
# Feature development
git checkout -b feat/data-loader
# ... write code & tests ...
pytest tests/test_data_loader.py
ruff format src tests
git add .
git commit -m "feat: implement data loader with validation"

# Push & PR
git push origin feat/data-loader

# GitHub Actions runs:
# ✓ ruff check
# ✓ mypy check  
# ✓ pytest + coverage
# Review & merge
```

---

## Expected Timeline (Estimate)

- Modules 1-3: ~2-3 sessions
- Modules 4-5: ~2-3 sessions
- Modules 6-7: ~2 sessions
- Module 8 (CLI): ~1 session
- Testing & polish: ~1 session

**Total**: 8-13 development sessions for MVP

---

## Success Criteria

✅ All modules implemented with >80% test coverage
✅ CLI functional end-to-end (CSV → Artifact)
✅ GitHub Actions passing on main
✅ Artifact structure validated
✅ Type checking with mypy
✅ README with usage examples
✅ Ready for Phase 2 (quality metrics)
