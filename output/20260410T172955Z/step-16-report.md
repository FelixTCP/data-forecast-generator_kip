# Forecasting Run Report (2026-04-10T17:42:45.764817+00:00)

## 1. Problem + selected target
Regression forecasting for target column `appliances` using CSV `appliances_energy_prediction.csv`.

## 2. Data quality summary
Rows: 19735, Columns: 29, Duplicate rows: 0.
Detected time column: date.

## 3. Candidate models + scores
- ridge: R2=0.1061, RMSE=86.0758, MAE=50.5136, CV_STD=0.0823
- gradient_boosting: R2=-9.7092, RMSE=297.9227, MAE=245.5456, CV_STD=0.3387
- random_forest: R2=-4.5300, RMSE=214.0864, MAE=192.5985, CV_STD=0.2941

## 4. Selected model rationale
Selected model: ridge (weighted_score=1.0000).
Selection used weighted normalized ranking with complexity tie-break.

## 5. Risks and caveats
- Holdout performance may shift on future data drift.
- If temporal ordering changes in source systems, retraining is required.
- Residual variance suggests periodic recalibration checks.

## 6. Next iteration recommendations
- Add domain-specific lag windows and holiday/event flags.
- Benchmark additional regularized and boosting variants.
- Add backtesting slices for stronger temporal robustness checks.
