# Model Selection Report

## Baselines

| Baseline | R² | RMSE | MAE |
|----------|-----|------|-----|
| Mean Predictor | 0.000 | - | - |
| naive_persistence | 0.9177262617031104 | 3.099426121631291 | 2.255537547271745 |
| seasonal_naive | 0.7512210436763185 | 5.3896119119671635 | 4.175652800288132 |

## Candidate Rankings

| Rank | Model | Weighted Score | R² | RMSE | MAE | Status |
|------|-------|-----------------|-----|------|-----|---------|
| 1 | gradient_boosting | 1.0000 | 0.9294 | 2.8717 | 2.1032 | eligible |
| 2 | ridge | 0.6902 | 0.9269 | 2.9208 | 2.1574 | eligible |
| 3 | random_forest | 0.3975 | 0.9244 | 2.9709 | 2.2144 | eligible |
| 4 | elasticnet | 0.0516 | 0.9201 | 3.0540 | 2.2730 | eligible |

## Candidate Analysis

**gradient_boosting:** Beats naive persistence (R² 0.9294 vs 0.9177). Excellent CV stability (std < 0.01). Max residual (15.21) exceeds target std (10.81).

**ridge:** Beats naive persistence (R² 0.9269 vs 0.9177). Good CV stability. Max residual (15.67) exceeds target std (10.81).

**random_forest:** Beats naive persistence (R² 0.9244 vs 0.9177). Good CV stability. Max residual (15.58) exceeds target std (10.81).

**elasticnet:** Beats naive persistence (R² 0.9201 vs 0.9177). Good CV stability. Max residual (15.73) exceeds target std (10.81).


## Selection Rationale

Selected 'gradient_boosting' with weighted score 1.0000. This model achieves R² = 0.9294, outperforming the mean baseline and the naive lag baseline (R² = 0.9177). It offers a good balance between performance (R², RMSE, MAE) and cross-validation stability. The selected model is simpler than alternatives while maintaining strong predictive accuracy.

## Quality Assessment

**Quality Flag:** acceptable

