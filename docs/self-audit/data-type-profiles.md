# Data Type Profiles (Universal Detection)

The audit system adapts its checks and thresholds based on detected data characteristics. This document defines **universal profile detection that works for any dataset**.

---

## Profile Detection (Universal Algorithm)

The audit system automatically detects the data profile using these objective signals:

### Signal 1: Temporal Structure
- **Has time column?** (detected in step 01)
  - No → `static_regression` (atemporal features)
  - Yes → proceed to Signal 2

### Signal 2: Multi-Series Structure  
- **Duplicate timestamps?** (`n_unique(time_col) < n_rows`)
  - Yes → `multi_series_temporal` (multiple entities over time)
  - No → proceed to Signal 3

### Signal 3: Short-Term Temporal Patterns
- **Strong autocorrelation at short lags** (acf[1:7] > 0.4)?
  - Yes → `daily_cyclical_temporal` (daily/weekly patterns)
  - No → proceed to Signal 4

### Signal 4: Longer-Term Autocorrelation
- **Strong autocorrelation at longer lags** (acf[24:48] > 0.3 OR acf[7:30] > 0.3)?
  - Yes → `longer_period_temporal` (multi-day/weekly/monthly cycles)
  - No → `generic_temporal`

---

## Profile Definitions

### 1. Multi-Series Temporal Data (`multi_series_temporal`)

**Detection Criteria:**
- ✓ Time column present
- ✓ Duplicate timestamps (`n_unique(time_col) < n_rows`)
- ✓ Grouping column(s) with 3–100 unique values
- ✓ Variance between groups > variance within groups (ratio > 2.0)

**Characteristics:**
- Multiple independent entities (e.g., stocks, machines, sensors, locations, devices, users) measured over time
- Each entity has its own temporal trajectory
- Same timestamps appear multiple times (once per entity)
- **Real-world examples:**
  - Stock prices: 7 assets × 252 trading days = 1764 rows with 252 unique timestamps
  - Industrial sensors: 5 machines × 1000 hours = 5000 rows with 1000 unique timestamps  
  - Building sensors: 10 cities × 365 days = 3650 rows with 365 unique timestamps

**Expected Difficulty:** High (multi-series models are harder; entity-specific patterns)

**Audit Thresholds:**

| Check | Pass | Marginal | Fail |
|---|---|---|---|
| Multi-series detection | Per-entity models | Warnings if mixed | FAIL if mixed models |
| R² (per-entity) | ≥ 0.50 | 0.30–0.50 | < 0.30 |
| Feature-target MI | ≥ 0.10 avg | 0.05–0.10 | < 0.05 |
| Distribution drift (KS) | < 0.15 | 0.15–0.25 | ≥ 0.25 |
| Temporal consistency | Regular intervals | Gaps ≤ 10% | Gaps > 10% |

**Typical Issues & Remediation:**
- **Issue:** Model trained on mixed entities (FAIL in multi-series detection).
  - **Remedy:** Train separate models per entity. Expected improvement: +0.2–0.4 R² per entity.
- **Issue:** Low R² (0.25–0.35) despite good features.
  - **Remedy:** Extend lag window; add entity-specific features; ensemble predictions.
- **Issue:** Overfitting (holdout R² << CV R²).
  - **Remedy:** Increase regularization; reduce feature count; validate on out-of-sample entities/time.

---

### 2. Daily-Cyclical Temporal Data (`daily_cyclical_temporal`)

**Detection Criteria:**
- ✓ Time column present (no duplicate timestamps = single series)
- ✓ Strong autocorrelation at lags 1–7 (acf > 0.4)
- ✓ No multi-series grouping column detected
- ✓ Regular frequency (hourly, 10-min, daily, etc.)

**Characteristics:**
- Single entity with repeating daily/weekly patterns
- Values depend strongly on recent history (lag 1–7)
- Clear diurnal or weekly cycle
- **Real-world examples:**
  - Hourly electricity: repeats every 24 hours (acf[24] > 0.7)
  - Building occupancy: repeats daily pattern (acf[24] > 0.6)
  - Website traffic: repeats weekly (acf[7] > 0.5)
  - 10-minute weather: repeats daily (acf[144] > 0.6; 144 = 24h × 10-min)

**Expected Difficulty:** Moderate (good features capture patterns; beware temporal leakage)

**Audit Thresholds:**

| Check | Pass | Marginal | Fail |
|---|---|---|---|
| R² (holdout) | ≥ 0.55 | 0.35–0.55 | < 0.35 |
| Temporal consistency | Regular + no gaps | Gaps ≤ 5% | Gaps > 5% |
| Short-lag autocorr | acf[1:7] strong | acf[1:7] moderate | acf[1:7] weak |
| Distribution drift (KS) | < 0.15 | 0.15–0.25 | ≥ 0.25 |
| Seasonal features | Time-of-day/day-of-week | Partial | None |

**Typical Issues & Remediation:**
- **Issue:** Good CV R² (0.6+) but holdout R² much lower (< 0.4).
  - **Remedy:** Temporal leakage detected—model learned training-set seasonality. Use TimeSeriesSplit; validate on new time periods.
- **Issue:** Model captures average pattern but misses irregular spikes.
  - **Remedy:** Add external regressors (weather, holidays, events); use quantile regression.
- **Issue:** Low R² despite strong autocorrelation.
  - **Remedy:** Increase lag window; add rolling statistics (mean, std, min, max); consider non-linear models.

---

### 3. Longer-Period Temporal Data (`longer_period_temporal`)

**Detection Criteria:**
- ✓ Time column present (no duplicate timestamps)
- ✓ Weak autocorrelation at short lags (acf[1:7] ≤ 0.4)
- ✓ Strong autocorrelation at longer periods (acf[24:48] > 0.3 OR acf[7:30] > 0.3)
- ✓ Regular or semi-regular frequency

**Characteristics:**
- Single entity with longer-term trends or cycles (weeks, months, quarters, years)
- Daily or weekly frequency, but predictive power comes from multi-week/month patterns
- Patterns may have trend component
- **Real-world examples:**
  - Stock prices: weak daily acf (efficient market), but strong monthly/quarterly trends (acf[30] > 0.4)
  - Monthly sales: seasonal pattern at acf[12] (yearly seasonality)
  - Temperature data: strong monthly seasonality (acf[30] > 0.6 for daily data)
  - Quarterly economic indicators: acf[4] (quarterly cycle) > 0.5

**Expected Difficulty:** Moderate-to-Hard (longer lags needed; trend handling required)

**Audit Thresholds:**

| Check | Pass | Marginal | Fail |
|---|---|---|---|
| R² (holdout) | ≥ 0.50 | 0.25–0.50 | < 0.25 |
| Lag window | ≥ 30–60 observations | 15–30 observations | < 15 observations |
| Trend component | Identified + handled | Partial | Not handled |
| Distribution drift (KS) | < 0.20 | 0.20–0.30 | ≥ 0.30 |
| Stationarity | Yes (detrended) | Marginal | Non-stationary |

**Typical Issues & Remediation:**
- **Issue:** Low R² despite long lag history.
  - **Remedy:** Apply detrending/differencing; use trend-aware models (ARIMA, Prophet); add external signals.
- **Issue:** Model works well on training set but fails on new year/quarter.
  - **Remedy:** Validate on multiple out-of-sample periods; check for structural breaks; recommend frequent retraining.
- **Issue:** Seasonal patterns not captured or incorrect seasonality type.
  - **Remedy:** Add seasonal decomposition features; use multiplicative vs. additive seasonality.

---

### 4. Generic Temporal Data (`generic_temporal`)

**Detection Criteria:**
- ✓ Time column present
- ✓ No strong autocorrelation pattern (acf[1:7] ≤ 0.4 AND acf[24:48] ≤ 0.3)
- ✓ Single series (no duplicate timestamps)
- ✓ Irregular frequency or weak temporal structure

**Characteristics:**
- Temporal features present but weak predictive signal
- Prediction relies on external/contemporaneous features more than time dynamics
- May include noise or weak trend
- Time ordering exists but patterns are not clear
- **Real-world examples:**
  - Sparse transaction data (hours/days between records with gaps)
  - Noisy sensor readings (temporal signal buried in noise)
  - Mixed frequency data (sparse events combined with regular measurements)
  - Irregular event timestamps

**Expected Difficulty:** Moderate (temporal info marginal; feature engineering critical)

**Audit Thresholds:**

| Check | Pass | Marginal | Fail |
|---|---|---|---|
| R² (holdout) | ≥ 0.50 | 0.25–0.50 | < 0.25 |
| Feature count | ≥ 5 good features | 3–5 mediocre features | < 3 features |
| MI with target | ≥ 0.08 avg | 0.04–0.08 avg | < 0.04 avg |
| Distribution drift (KS) | < 0.15 | 0.15–0.25 | ≥ 0.25 |

**Typical Issues & Remediation:**
- **Issue:** Low MI across all features; weak predictability.
  - **Remedy:** Check data quality; add domain knowledge features; consider simpler baseline model; examine for missing predictors.
- **Issue:** Model overfits to noise despite weak temporal signal.
  - **Remedy:** Use cross-validation; increase regularization; reduce feature count; prefer simpler models.

---

### 5. Static Regression (Atemporal Data) (`static_regression`)

**Detection Criteria:**
- ✗ No time column detected (or time column is categorical/meaningless)
- ✓ Features are static/atemporal (no temporal dependencies)
- ✓ Rows are independent observations

**Characteristics:**
- Cross-sectional data: each row is an independent case
- No temporal ordering or causality
- Features describe intrinsic properties, not history
- **Real-world examples:**
  - Customer demographics → purchase amount
  - House features (square footage, rooms, age) → sale price
  - Employee attributes (education, experience, department) → salary
  - Product specs (material, size, color) → quality score

**Expected Difficulty:** Easiest (no temporal leakage; standard regression)

**Audit Thresholds:**

| Check | Pass | Marginal | Fail |
|---|---|---|---|
| R² (holdout) | ≥ 0.60 | 0.35–0.60 | < 0.35 |
| Feature count | ≥ 5 useful features | 3–5 features | < 3 features |
| Multicollinearity (VIF) | All ≤ 5 | Some 5–10 | Any > 10 |
| Distribution drift (KS) | < 0.10 | 0.10–0.20 | ≥ 0.20 |

**Typical Issues & Remediation:**
- **Issue:** High multicollinearity; model unstable across datasets.
  - **Remedy:** Remove redundant features; use Ridge/Lasso regularization; PCA.
- **Issue:** Low R² despite good features.
  - **Remedy:** Feature engineering (interactions, polynomial terms); check for non-linearity; ensemble models.
- **Issue:** Training R² >> holdout R² (overfitting).
  - **Remedy:** Increase regularization; cross-validation; reduce feature count.

---

## How to Use This Document in the Audit

1. **Profile Detection** runs **before** the 5 audit checks.
2. Detection outputs: `detected_profile`, `confidence` (0.0–1.0), `characteristics` (list).
3. Each check uses **profile-specific thresholds** from the tables above.
4. **If confidence < 0.70:** Audit logs "profile uncertain—using conservative thresholds" and recommends domain expert review.
5. **Remediation suggestions** are adapted per profile.

---

## Examples of Profile Detection

### Example 1: Multi-Series Temporal (Any number of entities)
```
Inputs: CSV with columns [timestamp, entity_id, feature_1, feature_2, target]
Detection:
  - has_time_col = True (timestamp detected)
  - n_unique(timestamp) = 252
  - n_rows = 1764 (252 × 7 entities)
  - n_unique(timestamp) < n_rows ✓
→ Profile: multi_series_temporal, confidence=0.95
```

### Example 2: Daily-Cyclical Temporal (Any frequency with daily pattern)
```
Inputs: CSV with columns [timestamp, value, external_feature]
Detection:
  - has_time_col = True (hourly or sub-daily)
  - n_unique(timestamp) = n_rows (no duplicates)
  - acf[24] = 0.85 (strong 24-hour cycle)
  - acf[1:7] range = [0.75, 0.92] (strong short-lag)
→ Profile: daily_cyclical_temporal, confidence=0.92
```

### Example 3: Static Regression (Any non-temporal dataset)
```
Inputs: CSV with columns [feature_1, feature_2, ..., target]
Detection:
  - has_time_col = False
→ Profile: static_regression, confidence=1.0
```

### Example 4: Longer-Period Temporal (Any frequency with month/year cycles)
```
Inputs: CSV with columns [date, value, external_features]
Detection:
  - has_time_col = True
  - acf[1:7] = 0.15 (weak short-lag)
  - acf[30] = 0.72 (strong 30-day/monthly cycle)
→ Profile: longer_period_temporal, confidence=0.88
```

---

## Summary Table: All 5 Profiles

| Profile | Requires Time? | Multi-Series? | Dominant Pattern | R² Pass Threshold | Typical Difficulty |
|---|---|---|---|---|---|
| `multi_series_temporal` | ✓ | ✓ | Duplicate timestamps | ≥ 0.50 | Hard |
| `daily_cyclical_temporal` | ✓ | ✗ | acf[1:7] > 0.4 | ≥ 0.55 | Moderate |
| `longer_period_temporal` | ✓ | ✗ | acf[24+] > 0.3 | ≥ 0.50 | Moderate-Hard |
| `generic_temporal` | ✓ | ✗ | Weak/irregular acf | ≥ 0.50 | Moderate |
| `static_regression` | ✗ | N/A | Cross-sectional | ≥ 0.60 | Easy |

---

## Key Principles (Universal Design)

1. **Universality:** Diese Profile basieren auf **objektiven, messbaren Merkmalen** (Autocorrelation, Duplikat-Timestamps, Häufigkeit), nicht auf spezifischen Datensätzen.

2. **Any Dataset Works:** Egal ob Stocks, Energie, Wetter, Verkauf, Immobilien, IoT, Sensoren — die Erkennung funktioniert für alle.

3. **Adaptive Thresholds:** Jedes Profil hat seine eigenen R²-, KS-, Lag- und Konsistenz-Schwellwerte.

4. **Confidence Scoring:** Wenn Konfidenz < 0.7, empfiehlt die Audit Überprüfung durch einen Domänen-Experten.

5. **No Dataset-Specific Names:** Profile heißen nach ihren **Charakteristiken** (multi_series, daily_cyclical, etc.), nicht nach spezifischen Beispielen.
