# Model Selection Report

**Run ID:** 20260622T174637Z

**Selected Model:** ridge

**Quality Flag:** acceptable

## Baselines

- Mean baseline R²: 0.0000
- Naive lag baseline R²: 0.9177358394764266

## Candidate Ranking

| Rank | Model | R² | RMSE | MAE | CV R² | Weighted Score | Eligible |
|------|-------|-----|------|-----|-------|----------------|----------|
| 1 | ridge | 0.9325 | 2.81 | 2.05 | 0.9290 | 0.9995 | ✓ |
| 2 | random_forest | 0.9255 | 2.95 | 2.20 | 0.9099 | 0.7566 | ✓ |
| 3 | gradient_boosting | 0.9149 | 3.15 | 2.48 | 0.8876 | 0.3832 | ✓ |
| 4 | xgboost | 0.9050 | 3.33 | 2.56 | 0.8645 | 0.0968 | ✓ |

## Rationale

ridge scored highest with a weighted score of 0.9995 (R²=0.9325, RMSE=2.81, MAE=2.05). It outperforms the naive lag baseline (R²=0.9177) and demonstrates stable cross-validation performance (CV R²=0.929). Lower-complexity models are preferred as tie-breakers to reduce overfitting risk.

## Candidate Analysis

**gradient_boosting**: R²=0.9149; beats mean baseline (R² > 0). lags naive baseline by 0.0028 R². RMSE=3.15. CV R²=0.888±0.040.

**random_forest**: R²=0.9255; beats mean baseline (R² > 0). beats naive baseline by 0.0077 R². RMSE=2.95. CV R²=0.910±0.011.

**ridge**: R²=0.9325; beats mean baseline (R² > 0). beats naive baseline by 0.0148 R². RMSE=2.81. CV R²=0.929±0.005.

**xgboost**: R²=0.9050; beats mean baseline (R² > 0). lags naive baseline by 0.0127 R². RMSE=3.33. CV R²=0.865±0.032.

