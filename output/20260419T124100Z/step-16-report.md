# Forecasting Run Report

## 1. Problem + selected target
This pipeline run performs regression forecasting for target column 'appliances'.
Selected model: ridge.

## 2. Data quality summary
Rows after cleansing: 19735
Columns after cleansing: 29
Detected time column: date
Recommended features from exploration: 21
Low variance columns flagged: 0
Redundant columns flagged: 5

## 3. Candidate models + scores table
| model_name | status | eligible | r2 | rmse | mae | cv_std_r2 | weighted_score |
|---|---:|---:|---:|---:|---:|---:|---:|
| ridge | ok | True | 0.566883 | 59.563297 | 28.412928 | 0.067663 | 0.9 |
| random_forest | ok | False | -0.032898 | 91.982434 | 60.499806 | 0.098148 |  |
| gradient_boosting | ok | True | 0.379734 | 71.279549 | 45.498995 | 0.047929 | 0.1 |

## 4. Selected model rationale
Model 'ridge' achieved the highest weighted score (0.9000) under the configured multi-metric rule. Its holdout profile balances R2=0.5669, RMSE=59.5633, and MAE=28.4129 with stability bonus from CV variance.

## 5. Risks and caveats
Quality flag: acceptable
Target holdout mean/std/min/max: 96.07151914785696, 90.50573357075729, 20.0, 850.0
No major risk flags were raised beyond normal forecasting uncertainty.

## 6. Next iteration recommendations
- Review excluded features and MI thresholds from step 11 to recover potentially useful predictors.
- Run additional hyperparameter search for the top 2 ranked model families.
- Backtest across multiple chronological splits to validate stability over different periods.
