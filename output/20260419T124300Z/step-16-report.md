# Regression Forecasting Pipeline Report

**Run ID:** 20260419T124300Z  
**Generated:** 2026-04-19T11:50:21.081516Z  
**Data Source:** data/appliances_energy_prediction.csv  

## 1. Problem & Target Definition

**Target Column:** `Appliances`
**Task:** Build a regression model to predict Appliances
**Dataset Size:** 19735 rows × 29 initial columns

## 2. Data Quality Summary

**Target Statistics:**
- Mean: 96.0249391727494
- Std Dev: 89.9397371151608
- Min: 20.0
- Max: 850.0

**Features Used:**
- Initial numeric columns: 19
- After MI filtering: 16
- Final engineered features: 24

**Data Quality:** ✓ No null values detected

## 3. Candidate Models & Evaluation Scores

| Model | R² | RMSE | MAE | CV R² | Status |
|-------|----|----|-----|------|--------|
| ridge | 0.5434 | 60.7733 | 27.0967 | 0.5530 | ✓ |
| random_forest | 0.1709 | 81.8957 | 49.0656 | 0.3923 | ✓ |
| gradient_boosting | -0.8045 | 120.8191 | 91.2439 | 0.1157 | ❌ Below baseline |

## 4. Selected Model Rationale

**Selected Model:** `ridge`
**Weighted Score:** 1.0000
**Quality Assessment:** ACCEPTABLE

**Rationale:**

ridge achieved the highest weighted score (1.0000) with R² = 0.5434, RMSE = 60.7733, and MAE = 27.0967. It outperformed random_forest by 1.0000 points.

## 5. Risks & Caveats

- ✓ No major risks identified. Model may be suitable for deployment with standard validation.

## 6. Next Iteration Recommendations

1. **Cross-validation:** Perform k-fold or time-series cross-validation on full dataset
2. **Production validation:** Test on recent unseen data to confirm generalization
3. **Monitoring:** Set up performance tracking post-deployment
4. **Retraining schedule:** Plan periodic model updates as new data becomes available

---

*Pipeline Version: 1.0.0*  
*Framework: data-forecast-generator*  
*Report generated automatically by agentic pipeline*