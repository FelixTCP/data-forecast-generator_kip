# Forecasting Report (2026-04-18T10:49:22.186258+00:00)

## 1. Problem + selected target
This run predicts 'appliances' from dataset 'appliances_energy_prediction.csv' using a per-step regression workflow.
Model selection quality flag: acceptable and selected model: ridge.

## 2. Data quality summary
Rows after cleansing: 19735; columns: 29; duplicates: 0.
Detected time column: date.
Top null-rate columns:
- date: 0.0000
- appliances: 0.0000
- lights: 0.0000
- t1: 0.0000
- rh_1: 0.0000
- t2: 0.0000
- rh_2: 0.0000
- t3: 0.0000
- rh_3: 0.0000
- t4: 0.0000

## 3. Candidate models + scores table
| Model | R2 | RMSE | MAE | CV Std | Ineligible |
|---|---:|---:|---:|---:|---:|
| ridge | 0.5669 | 59.5633 | 28.4129 | 0.0677 | False |
| gradient_boosting | 0.3247 | 74.3739 | 50.2406 | 0.0443 | False |
| random_forest | -0.0348 | 92.0673 | 60.5778 | 0.0978 | True |

## 4. Selected model rationale
ridge achieved the highest weighted score under the configured blend of R2, RMSE, MAE, and stability. The tie-breaker favors lower complexity, which helps reduce overfitting risk when scores are close.

## 5. Risks and caveats
- Holdout quality can degrade under distribution shift in weather and occupancy patterns.
- Lag features may leak weak short-term autocorrelation that does not generalize across seasons.
- Negative-R2 candidates were marked ineligible and excluded from winner consideration.
- If quality flag is subpar, downstream decisions should not rely on this model without retraining.

## 6. Next iteration recommendations
- Add richer exogenous features (holiday calendar, occupancy proxies, weather interactions).
- Run rolling-origin backtests to stress temporal robustness beyond one holdout split.
- Tune expanded candidates using constrained search spaces and monotonic constraints where valid.