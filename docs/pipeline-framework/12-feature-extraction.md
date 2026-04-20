# Schritt 12 — Feature Extraction & Model Preselection

**Modul**: `src/data_forecast_generator/pipeline/feature_extraction.py`  
**Input**: Output von `data_exploration.py` (Schritt 11)  
**Output**: Feature-Matrix + Modell-Empfehlungen → `model_training.py` (Schritt 13)  
**Artefakte**: `outputs/step-12/features.parquet`, `outputs/step-12/step-12-features.json`

```
[10] csv_read_cleansing → [11] data_exploration → [12] feature_extraction → [13] model_training → ...
```

---

## Feature-Philosophie

| Prinzip | Regel |
|---|---|
| Kausales Rolling | `.shift(1)` vor jeder `rolling_*`-Berechnung — verhindert Look-ahead |
| Leakage → Hard Fail | `RuntimeError` wenn `\|r\| ≥ threshold` — kein Artefakt wird geschrieben |
| Leakage-Probe | Paarweise Pearson-Korrelation **und** Rekonstruktions-Probe (RF R² > 0.999) |
| Mindest-Features | Weniger als 2 Features nach Bereinigung → `ValueError` |

---

## Input-Vertrag (Output von Schritt 11)

Das Input für `feature_extraction.py` ist der direkte Output von `data_exploration.py`:

```python
exploration_output = {
    # === METADATEN ===
    "metadata": {
        "execution_id": "expl-2024-04-09-123",
        "module_name": "data_exploration",
        "timestamp_created": "2024-04-09T12:34:56Z",
        "source_file": "appliances_energy.csv",
        "n_rows": 19735,
        "n_cols": 29,
    },

    # === ROHDATEN ===
    "data": {
        "df": pl.DataFrame,                                    # Bereinigtes DataFrame (keine NaN in y!)
        "time_col": "date",                                    # Name der Zeitspalte
        "numeric_cols": ["Appliances", "lights", "T1", "RH_1", "T_out", ...],
        "categorical_cols": [],
        "target_column": null,                                 # Falls nicht explizit gesetzt
    },

    # === EXPLORATIONS-ERGEBNISSE ===
    "exploration": {
        "frequency": "10min",                                  # Erkannte Zeitreihen-Frequenz
        "stationarity_adf_pvalue": 0.23,                       # ADF-Test p-Wert
        "missing_fraction": 0.0,                               # Anteil fehlender Werte (vor Cleaning)
        "column_stats": {                                      # Deskriptive Statistik je Spalte
            "Appliances": {"mean": 93.7, "std": 102.3, "min": 0, "max": 2942},
            "lights": {"mean": 4.2, "std": 8.4, "min": 0, "max": 176},
        },
    },

    # === ANALYSE-ERGEBNISSE ===
    "analysis": {
        "mi_ranking": [                                        # Mutual Information Ranking (absteigend)
            {"feature": "T6", "mi_score": 0.42, "below_noise_baseline": false},
            {"feature": "T1", "mi_score": 0.35, "below_noise_baseline": false},
        ],
        "noise_mi_baseline": 0.005,                            # Durchschnitt der MI von Rausch-Spalten
        "redundant_columns": ["rv2"],                          # Redundante Features (high correlation)
        "correlation_matrix_summary": {
            "max_pair": ["rv1", "rv2"],
            "max_corr": 1.0
        },
        "low_variance_columns": [],                            # Low-Variance Features
        "recommended_features": ["T6", "T1", "RH_6", "lights", "T_out"],  # Empfohlene Features für Step 12
        "excluded_features": {
            "rv1": "below_noise_baseline",
            "rv2": "redundant"
        }
    },

    # === TIME-SERIES SPEZIFISCHE ANALYSE ===
    "time_series": {
        "time_series_detected": true,
        "significant_lags": [1, 3, 6],                         # Lags mit signifikantem Autocorrelation
        "useful_lag_features": [                               # Features mit signifikantem Cross-Correlation
            {"feature": "T1", "lag": 1, "xcorr": 0.23},
            {"feature": "RH_6", "lag": 3, "xcorr": 0.18},
        ]
    },

    # === FEHLER & WARNUNGEN ===
    "errors": [],
    "warnings": [],
}
```

**Kontrakt-Garantien:**
- `df` ist polars.DataFrame mit keinen NaN in der Zielvariable
- `numeric_cols` und `categorical_cols` decken alle Spalten ab
- `recommended_features` ist niemals leer (oder Fehler wird in Schritt 11 geworfen)
- `analysis.excluded_features` erklärt jeden ausgeschlossenen Feature

---

## Output-Vertrag (für Training-Modul)

```python
feature_output = {
    # === METADATEN ===
    "metadata": {
        "execution_id": "feat-2024-04-09-456",           # uuid4()[:8]
        "module_name": "feature_extraction",
        "timestamp_created": "2024-04-09T14:23:45Z",
        "runtime_seconds": 12.34,
    },

    # === INPUT-REFERENZ ===
    "input_ref": {
        "exploration_execution_id": "expl-2024-04-09-234",
        "source_file": "energy_demand.csv",
    },

    # === ZIELSPALTE ===
    "target_info": {
        "target_column": "energy",
        "detection_method": "highest_variance",           # "explicit" wenn via config übergeben
        "detection_info": {
            "candidates": ["energy", "temperature"],
            "scores": {"energy": 0.95, "temperature": 0.42},
            "reason": "energy hat höchste Varianz (0.95)",
        },
    },

    # === FEATURES ===
    "features": {
        "feature_matrix": pl.DataFrame,                  # Shape: (n_rows, n_features), nur numerisch, inkl. Target
        "target_series": pl.Series,                      # Zielvariable (in-memory für nächstes Modul)
        "parquet_path": "outputs/step-12/features.parquet",  # Pfad zur gespeicherten Feature-Matrix
        "feature_names": ["y_lag_1", "y_lag_24", "hour", "is_weekend"],
        "feature_count": 47,
        "rows_dropped_by_lags": 50,
        "final_row_count": 8650,
        "adaptive_features_added": ["y_diff_1", "embedding_matrices"],
    },

    # === ARTEFAKTE ===
    "artifacts": {
        "features_parquet": "outputs/step-12/features.parquet",  # Feature-Matrix (numerisch, inkl. Target)
        "audit_json": "outputs/step-12/step-12-features.json",   # Vollständiger Audit Trail
    },

    # === LEAKAGE ===
    "leakage": {
        "status": "pass",                # "pass" | "fail" — bei "fail" wird RuntimeError raised
        "leakage_candidates": [],        # Feature-Namen mit |r| >= threshold
        "correlations": {"y_lag_0": 1.0, "y_lag_1": 0.87},
        "threshold": 0.98,
        "reconstruction_probe_r2": None, # RF R² der Leakage-Probe (None wenn kein Kandidat)
    },

    # === SCALING-METADATEN ===
    "scaling_metadata": {
        "scaling_required": True,
        "per_model": {
            "Gradient Boosting": {"scaler": None,            "features": []},
            "SARIMA":            {"scaler": "StandardScaler", "features": ["y_lag_1", "hour"]},
        },
        "never_scale": ["is_weekend", "hour_of_day"],
    },

    # === ANALYSE-ERGEBNISSE ===
    "analysis": {
        "best_lags": [1, 2, 3, 24, 48],
        "seasonality": {
            "strength": 0.72,
            "label": "stark",
            "dominant_period": 24,
        },
        "target_distribution": {
            "mean": 23.1,
            "std": 14.2,
            "cv": 0.61,
            "tree_model_suitable": "bedingt",
        },
        "strata": {
            "active": ["hour_of_day", "day_of_week"],
            "variance_ratios": {"hour": 0.45, "day": 0.32},
        },
    },

    # === MODELL-EMPFEHLUNG ===
    "model_recommendations": {
        "recommendations": [
            {
                "model_type": str,
                "suitability": "sinnvoll" | "eher sinnvoll" | "eher nicht",
                "score": float,                          # 0.0–1.0
                "reasoning": str,
                "required_features": list[str],
                "required_features_missing": list[str],
                "why_good_fit": str,
            },
            # ...
        ],
        "top_recommendation": str,
        "top_3_recommendations": list[str],
        "adaptive_features_needed": dict[str, list[str]],
    },

    # === FEHLER & WARNUNGEN ===
    "errors": [],
    "warnings": [
        "Zeitreihe wurde um 50 Zeilen wegen Lag gekürzt",
        "State Space Embedding: Zeitreihe < 1 Jahr, skipped",
    ],
}
```

---

## Einstiegsfunktion

```python
# import json, logging, time, uuid
# from datetime import datetime, timezone
# from pathlib import Path
# import polars as pl

def run_analysis(
    exploration_output: dict,  # Output von Schritt 11 (siehe Input-Vertrag oben)
    config: dict | None = None,
    # config keys: max_lag=48, embedding_dim=3, use_state_space=True,
    #              rolling_windows=[7,14,30], output_dir="outputs/step-12",
    #              target_column=None  # optional: explizite Zielspalte, überspringt Auto-Detection
) -> dict:
    """
    **Input**: exploration_output vom Schritt 11
      - exploration_output["data"]["df"] wird als Input-DataFrame verwendet
      - exploration_output["data"]["numeric_cols"] definiert die Features
      - exploration_output["analysis"]["recommended_features"] lenkt Feature-Selektion
      - exploration_output["time_series"]["significant_lags"] lenkt Lag-Kreation

    **Output**: Feature-Extraktions-Output (siehe Output-Vertrag)
      - Schreibt features.parquet und step-12-features.json (nur wenn leakage.status="pass")
      - feature_matrix ist pl.DataFrame mit allen engineered Features + Target

    **Fehlerverarbeitung**:
      - ValueError bei ungültigem Input (zu wenig Features, leeres DataFrame, etc.)
      - RuntimeError wenn Leakage erkannt wird — kein Artefakt wird geschrieben
    """
```

---

## Funktions-Spezifikationen

### Z — Zielspalten-Bestimmung

```python
def auto_detect_target_column(
    df: pl.DataFrame,
    time_col: str,
    numeric_cols: list[str],
    heuristic: str = "highest_variance",
    explicit_target: str | None = None,
) -> tuple[str, dict]:
    """
    Gibt explizite Zielspalte zurück, wenn übergeben (nach Validierung gegen numeric_cols).
    Sonst: automatische Wahl aus numerischen Spalten.

    Heuristiken (nur bei Auto-Detection):
    - "highest_variance": Spalte mit größter Varianz
    - "most_correlated_with_self": höchste Autokorrelation

    Ausschlüsse (nur bei Auto-Detection): time_col, konstante Spalten (Varianz ≈ 0)

    Returns:
        (target_column_name, detection_info)
        detection_info: {"method": "explicit"|"highest_variance"|..., "candidates", "scores", "reason"}

    Raises:
        ValueError: Wenn explicit_target nicht in numeric_cols
    """
```

---

### A — Mutual Information

```python
def compute_lag_mutual_information(
    df: pl.DataFrame,
    target_col: str,
    max_lag: int = 48,
    n_neighbors: int = 5,
) -> pl.DataFrame:
    """
    Berechnet MI zwischen y(t) und y(t-lag) für lag=1..max_lag.

    Nutzt: sklearn.feature_selection.mutual_info_regression

    Returns:
        pl.DataFrame mit Spalten [lag, mutual_information], sortiert nach MI desc
    """
```

---

### B — Beste Lags (ACF + PACF + MI)

```python
def find_best_lags(
    df: pl.DataFrame,
    target_col: str,
    max_lag: int = 48,
    top_n: int = 10,
    method: str = "combined",
) -> dict:
    """
    Kombiniert ACF, PACF und MI zur Lag-Selektion.

    - statsmodels.tsa.stattools.acf / pacf
    - 95%-Konfidenzgrenzen: ±1.96 / √n

    Returns:
        {
            "best_lags_acf": [...],
            "best_lags_pacf": [...],
            "best_lags_mi": [...],
            "recommended_lags": [...],
            "acf_values": pl.DataFrame,
            "pacf_values": pl.DataFrame,
        }
    """
```

---

### C — Saisonalitätserkennung

```python
def detect_seasonality(
    df: pl.DataFrame,
    target_col: str,
    time_col: str,
    candidate_periods: list[int] | None = None,
) -> dict:
    """
    STL-Dekomposition + FFT + ACF-Peaks.

    Klassifikation:
    - < 0.2  → "keine klare Saisonalität"
    - 0.2–0.5 → "schwache Saisonalität"
    - > 0.5  → "starke Saisonalität"

    Bei erkannter Saisonalität: Fourier-Features erzeugen
        sin(2π·k·t/period), cos(2π·k·t/period) für k=1,2

    Returns:
        {"seasonality_strength", "seasonality_label", "dominant_period",
         "seasonal_features", ...}
    """
```

---

### D — Zielverteilung

```python
def analyze_target_distribution(
    df: pl.DataFrame,
    target_col: str,
) -> dict:
    """
    Analysiert die y-Verteilung für die Modellwahl.

    Metriken: min, max, mean, std, skewness, kurtosis, IQR, CV

    Tree-Model-Heuristik:
    - Ausreißeranteil < 5%: positiv
    - CV < 1.0: positiv
    - CV > 3.0: negativ
    - |skew| < 1: positiv

    Returns:
        {"min", "max", "mean", "std", "cv",
         "tree_model_suitable": "ja"|"bedingt"|"eher nicht",
         "tree_model_reasoning": str, ...}
    """
```

---

### E — State-Space Embedding

```python
def compute_state_space_embedding(
    series: pl.Series,
    embedding_dim: int = 3,
    delay: int | None = None,
    max_delay_search: int = 20,
) -> dict:
    """
    Delay Embedding mit automatischer Delay-Wahl (Takens-Theorem).

    Auto-Delay: erstes lokales Minimum der MI-Kurve
    Embedding-Matrix: [x(t), x(t-τ), x(t-2τ), ..., x(t-(d-1)τ)]

    Returns:
        {"embedding_matrix": np.ndarray,  # shape (n_valid, embedding_dim)
         "chosen_delay": int,
         "embedding_features": pl.DataFrame,
         "notes": str}
    """
```

---

### F — Strata-Features

```python
def create_strata_features(
    df: pl.DataFrame,
    time_col: str,
    target_col: str,
) -> dict:
    """
    Zeitbasierte Stratifikation mit ANOVA-F-Test.

    Aktivierung je nach Frequenz & Datenlänge:
    - hour_of_day  → wenn freq ≤ 1h
    - day_of_week  → immer wenn Zeitstempel vorhanden
    - month        → wenn Daten ≥ 60 Tage
    - season       → wenn Daten ≥ 180 Tage

    Nützlichkeitsprüfung: ANOVA F-Test, p < 0.05 → sinnvoll

    Returns:
        {"strata_features", "active_strata", "strata_usefulness",
         "variance_by_strata"}
    """
```

---

### G — Feature Engineering

```python
def engineer_timeseries_features(
    df: pl.DataFrame,
    target_col: str,
    time_col: str,
    lags: list[int] | None = None,
    rolling_windows: list[int] | None = None,
    include_calendar: bool = True,
    include_trend: bool = True,
    include_differences: bool = True,
) -> tuple[pl.DataFrame, dict]:
    """
    Erstellt die vollständige Feature-Matrix.

    Features:
    - Lag:       y_lag_{k}
    - Rolling:   y_rolling_mean_{w}, _std, _min, _max, _range
                 PFLICHT: `.shift(1).rolling_*(w)` — kein Look-ahead!
                 Verletzung → Feature wird ausgeschlossen + in warnings geloggt.
    - Diff:      y_diff_1, y_diff_2, y_pct_change_1
    - Trend:     t_index, t_index_squared, t_elapsed_days
    - Kalender:  hour, day_of_week, month, quarter, year, is_weekend

    NaN-Behandlung: Erste N Zeilen (= max Lag / Window) droppen.
    Weniger als 2 Features nach Bereinigung → ValueError.

    Returns:
        (feature_df, feature_metadata_dict)
    """
```

---

### H — Model Preselection

```python
def preselect_models(
    feature_matrix: pl.DataFrame,
    analysis_data: dict,
    best_lags: list[int],
) -> dict:
    """
    Bewertet Modelltypen anhand der tatsächlich vorhandenen Features.

    Modelltypen & typische Feature-Anforderungen:

    1. Gradient Boosting (XGBoost, LightGBM)
       ✓ Lags, Rolling, Kalender, Trend (tabellarisch)
    2. SARIMA
       ✓ y_diff_1/2 (Stationarisierung), dominanter Saisonperiod
       ✗ Zu viele Außenregressoren problematisch
    3. LSTM / Temporal CNN
       ? embedding_matrices / Sequenzen (ggf. adaptiv erzeugen)
       ✓ Lange Zeitreihe (> 500 Punkte)
    4. Linear Models (Ridge, Lasso)
       ✗ Hohes MI bei kleinem ACF → nichtlinear → ungeeignet
    5. State-Space (Kalman, Prophet)
       ✓ Trend + Saisonalitäts-Features, toleriert fehlende Werte
    6. Tree-Based (RF, ExtraTrees)
       ✓ Alle Features; bei sehr vielen Features Overfitting-Risiko

    Returns:
        {
            "recommendations": [{
                "model_type", "suitability", "score",
                "reasoning", "required_features",
                "required_features_missing", "why_good_fit"
            }, ...],
            "top_recommendation": str,
            "top_3_recommendations": list[str],
            "adaptive_features_needed": dict[str, list[str]],
        }
    """
```

---

### I — Adaptive Feature Engineering

```python
def add_features_for_models(
    feature_matrix: pl.DataFrame,
    target_col: str,
    recommended_models: list[str],
    analysis_data: dict,
) -> tuple[pl.DataFrame, list[str]]:
    """
    Ergänzt fehlende, modellspezifische Features nachträglich.

    Beispiele:
    - Für ARIMA/SARIMA: y_diff_1, y_diff_2 (Stationarisierung)
    - Für LSTM/Temporal CNN: State-Space Embedding Spalten
    - Für Tree-based: Interaktions-Features zwischen Top-Lags
    - Für State-Space: explizite Trend- und Saisonalitäts-Features

    Returns:
        (feature_matrix_extended, newly_added_feature_names)
    """
```

---

### J — Leakage-Erkennung

```python
def detect_feature_leakage(
    feature_matrix: pl.DataFrame,
    target_col: str,
    threshold: float = 0.98,
) -> dict:
    """
    Prüft ob Features die Zielspalte unnatürlich stark vorwegnehmen.

    Methode:
    1. Paarweise Pearson-Korrelation: |r| ≥ threshold → Leakage-Kandidat
    2. Rekonstruktions-Probe: RandomForestRegressor (n_estimators=10) auf Leakage-Kandidaten;
       R² > 0.999 → Leakage bestätigt
    - Sonderregel: target_col selbst und korrekte Lag-Features (y_lag_k) ausschließen

    Returns:
        {
            "status": "pass" | "fail",
            "leakage_candidates": list[str],         # Feature-Namen mit |r| >= threshold
            "correlations": dict[str, float],        # {feature: r} für alle Features
            "threshold": float,
            "reconstruction_probe_r2": float | None, # R² der RF-Probe (None wenn kein Kandidat)
        }

    Raises:
        ValueError: Wenn target_col nicht in feature_matrix
        RuntimeError: Wenn status == "fail" (wird von run_analysis weitergereicht)
    """
```

Bei `status == "fail"` raised `run_analysis` eine `RuntimeError` — kein Artefakt wird geschrieben.

---

### K — Feature-Scaling-Metadaten

```python
def compute_scaling_metadata(
    feature_matrix: pl.DataFrame,
    target_col: str,
    recommended_models: list[str],
) -> dict:
    """
    Bestimmt welche Features für welche Modelltypen skaliert werden müssen.
    Die Skalierung selbst erfolgt in Schritt 13; dieses Dict ist die Anweisung dafür.

    Regeln:
    - Linear Models / SARIMA / State-Space: alle numerischen Features → StandardScaler
    - Gradient Boosting / Tree-based / RF:  kein Scaler nötig
    - LSTM / Temporal CNN:                  alle Features → MinMaxScaler (0..1)
    - Binäre Features (0/1):               niemals skalieren

    Returns:
        {
            "scaling_required": bool,
            "per_model": {
                "Gradient Boosting": {"scaler": None,            "features": []},
                "SARIMA":            {"scaler": "StandardScaler", "features": ["y_lag_1", ...]},
                "LSTM":              {"scaler": "MinMaxScaler",   "features": ["y_lag_1", ...]},
                # ...
            },
            "never_scale": list[str],   # Binäre Features (is_weekend, etc.)
        }
    """
```

---

## Implementierungs-Checkliste

- [ ] Alle Funktionen (Z, A–K) vollständig implementiert
- [ ] `import json, logging, time, uuid` und `from pathlib import Path` vorhanden
- [ ] Kein `argparse`, kein CLI-Code, kein `print()`
- [ ] Input-Validierung mit sprechendem `ValueError`
- [ ] `execution_id` via `uuid.uuid4()[:8]`
- [ ] `runtime_seconds` via `time.time()`
- [ ] `features.parquet` wird via `feature_df.write_parquet(parquet_path)` geschrieben
- [ ] `step-12-features.json` enthält vollständigen Audit Trail (kein `pl.DataFrame`/`pl.Series` direkt — nur serialisierbare Typen; `default=str` als Fallback)
- [ ] `output_dir` wird mit `mkdir(parents=True, exist_ok=True)` angelegt
- [ ] Return-Dict enthält `artifacts` mit `features_parquet` und `audit_json` als Pfad-Strings
- [ ] Return-Dict enthält `features.feature_matrix` als `pl.DataFrame` (in-memory für nächstes Modul)
- [ ] Output-Dictionary entspricht exakt dem Output-Vertrag
- [ ] Alle Funktionen zustandslos (keine globalen Variablen)
- [ ] Type Hints vollständig
- [ ] `leakage`-Block im Output mit `status: "pass"|"fail"`; bei `"fail"` → `RuntimeError`, kein Artefakt
- [ ] Leakage-Probe: RandomForest R² > 0.999 gilt als bestätigtes Leakage
- [ ] Rolling-Features: `.shift(1)` vor jeder Rolling-Berechnung; Verletzung → Feature ausschließen
- [ ] `scaling_metadata`-Block im Output; binäre Features in `never_scale`
- [ ] Tests unter `tests/test_feature_extraction.py`
