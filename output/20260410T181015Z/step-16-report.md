# Forecasting Report (2026-04-10T18:19:28.326614+00:00)

## 1. Problem + selected target
This run predicts 'appliances' from dataset 'appliances_energy_prediction.csv' using a multi-candidate regression workflow.

## 2. Data quality summary
Rows after cleansing: 19735; columns: 29; duplicates: 0.
Detected time column: date.
Top null rates:
- date: 0.000
- appliances: 0.000
- lights: 0.000
- t1: 0.000
- rh_1: 0.000
- t2: 0.000
- rh_2: 0.000
- t3: 0.000

## 3. Candidate models + scores table
| Model | R2 | RMSE | MAE | CV Std |
|---|---:|---:|---:|---:|
| ridge | 0.1147 | 85.6598 | 49.9705 | 0.0961 |
| random_forest | -4.6669 | 216.7202 | 176.1474 | 0.3133 |
| gradient_boosting | -6.8507 | 255.0817 | 198.2381 | 0.5658 |

## 4. Selected model rationale
Selected ridge with top weighted score 1.0000; tie-breaker favors lower complexity.

## 5. Risks and caveats
- Holdout score can vary under production data drift.
- Temporal behavior may shift due to weather/occupancy changes.
- Some engineered lag features can amplify noise spikes.

## 6. Next iteration recommendations
- Add event/holiday features and weather interaction terms.
- Add rolling backtests by month for stronger temporal confidence.
- Tune tree-based candidates with constrained depth and monotonic checks.