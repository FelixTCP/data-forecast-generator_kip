# #12 Context Engineering: Feature Extraction

## Objective

Build stable, leakage-safe features and transformation metadata.

## Outputs

- train-ready feature matrix
- feature metadata (transformations, encoders, dropped columns)

## Guardrails

- No target leakage features.
- Fit transformations only on training partition.
- Keep transformation graph serializable.

## Copilot Prompt Snippet

```markdown
Implement `build_features(df, target_column, time_column=None, config=None)`.
Output `X`, `y`, and `feature_meta` with deterministic column ordering.
```

## Code Skeleton

```python
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def make_preprocessor(numeric_features: list[str], categorical_features: list[str]) -> ColumnTransformer:
    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([
        ("num", num_pipe, numeric_features),
        ("cat", cat_pipe, categorical_features),
    ])
```

## Tests

- unknown categories in validation/test
- missing numeric + categorical values
- deterministic feature order
