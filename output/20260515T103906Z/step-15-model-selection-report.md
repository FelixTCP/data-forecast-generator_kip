# Step 15 — Model Selection Report

**Run ID:** 20260515T103906Z

## Baseline Summary

| Baseline | R² | RMSE | MAE |
|----|----|----|-----|
| Mean baseline (R²=0) | 0.0000 | - | - |
| Naïve lag-1 | 0.9117 | 3.157 | 2.300 |

## Candidate Ranking

| Rank | Model | R² | RMSE | MAE | CV R² | Weighted Score | Status |
|----|----|----|----|----|----|----|-----|
| 1 | ridge | 0.9277 | 2.858 | 2.083 | 0.9196 | 0.9000 | eligible |
| 2 | random_forest | 0.9246 | 2.918 | 2.134 | 0.9262 | 0.5670 | eligible |
| 3 | hist_gbm | 0.9243 | 2.924 | 2.144 | 0.9215 | 0.5153 | eligible |
| 4 | elasticnet | 0.9212 | 2.982 | 2.189 | 0.9236 | 0.0840 | eligible |
| - | holt_winters | -11.6327 | 37.768 | 32.421 | -3.9709 | - | ineligible |
| - | naive_persistence | 0.9117 | 3.158 | 2.302 | - | - | benchmark |
| - | seasonal_naive | 0.7042 | 5.779 | 4.432 | - | - | benchmark |
| - | auto_arima_benchmark | -1.5189 | 16.865 | 14.171 | - | - | benchmark |
| - | ar1_benchmark | 0.0099 | 10.574 | 9.118 | - | - | benchmark |

## Selected Model

**ridge** — ridge achieved the highest weighted score (0.9000) with holdout R²=0.9277, RMSE=2.858°F, MAE=2.083°F. It beats the naïve lag-1 baseline (R²=0.9117) by Δ=+0.0160. Cross-validation R²=0.9196±0.0150 indicates consistent performance across temporal folds. Note: The improvement over the naïve persistence baseline is modest. Daily temperature autocorrelation is strong — any lag-1 model performs well. The value of this model lies in better handling of seasonal transitions and feature integration.

## Candidate Analysis

**elasticnet:** R²=0.9212 > 0 (beats mean baseline). Performance nearly identical to naïve lag-1 baseline (Δ=+0.0095). CV R²=0.9236 ≈ holdout R²=0.9212: good generalization. RMSE=2.98°F (27.3% of target std).

**hist_gbm:** R²=0.9243 > 0 (beats mean baseline). Beats naïve lag-1 by Δ=+0.0126 R². CV R²=0.9215 ≈ holdout R²=0.9243: good generalization. RMSE=2.92°F (26.8% of target std).

**random_forest:** R²=0.9246 > 0 (beats mean baseline). Beats naïve lag-1 by Δ=+0.0129 R². CV R²=0.9262 ≈ holdout R²=0.9246: good generalization. RMSE=2.92°F (26.7% of target std).

**ridge:** R²=0.9277 > 0 (beats mean baseline). Beats naïve lag-1 by Δ=+0.0160 R². CV R²=0.9196 ≈ holdout R²=0.9277: good generalization. RMSE=2.86°F (26.2% of target std).

**holt_winters:** R²=-11.6327 < 0 — model is worse than predicting the mean. Ineligible for selection.
