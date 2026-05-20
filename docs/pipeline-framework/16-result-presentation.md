# #16 Context Engineering: Result Presentation

## Objective

Produce user-facing outputs that effectively communicate model results, business impact, and diagnostics to both technical and non-technical audiences.

If the upstream quality flag indicates `leakage_suspected`, the report must clearly state that metrics are invalid for production forecasting and must include remediation actions.

## Outputs

- `evaluation.json` (machine-readable)
- `report.md` (human-readable)
- optional lightweight plot assets

## Copilot Prompt Snippet

Implement `build_result_package(context, output_dir)`.
Generate a comprehensive summary detailing the chosen model, key metrics, strongest features, and actionable next steps. 
Crucially, begin the report with an "Executive Summary (ELI5)" that translates the project goal, results, and business impact into simple, jargon-free language suitable for a non-technical manager.
If leakage is suspected, do not present a "selected production model"; present diagnostics instead, and ensure the ELI5 section clearly explains *why* the current results cannot be trusted yet.

## Suggested `report.md` Sections & Expected Contents

**1. Executive Summary (ELI5)**
*   **Target Audience:** Non-technical managers / general business stakeholders.
*   **Expected Content:** 
    *   **The Goal:** What is the AI forecasting in plain English? (e.g., "Predicting how much inventory we will need for the next 30 days.")
    *   **The Result:** How close is the forecast to reality? (e.g., "Our predictions are typically within 5% of actual sales.")
    *   **The Business Impact:** How does this help? (e.g., "This prevents overstocking and reduces warehousing costs.")
    *   **Red Flags (If Applicable):** If `leakage_suspected`, explain the failure using a simple time-travel analogy (e.g., "The model accidentally peaked into the future while training. It looks artificially accurate, so we cannot use it yet.").

**2. Problem & Target Definition**
*   **Expected Content:**
    *   Formal framing: Time Series Regression.
    *   The exact target variable being forecasted.
    *   **Temporal Scope:** Specify the forecast horizon (e.g., predicting 7 days out) and granularity (e.g., hourly, daily).
    *   The primary evaluation metric (e.g., RMSE, MAE, MAPE, or WMAPE) and why it fits the business use case (e.g., "Using MAPE to understand the percentage error across differently sized regions").

**3. Data & Feature Summary**
*   **Expected Content:**
    *   Dataset timeframe (e.g., "Jan 2020 - Dec 2023") and frequency.
    *   **Top Features:** The most important drivers, specifically highlighting time-based features (e.g., lag variables, rolling averages, seasonality indicators, holiday flags).
    *   Noted data quality issues (e.g., missing timestamps, gaps in the series, or extreme outliers).

**4. Model Benchmarking (Candidates & Scores)**
*   **Expected Content:**
    *   A markdown table comparing all tested forecasting algorithms (e.g., ARIMA, Prophet, XGBoost, etc.).
    *   Primary and secondary validation metrics for each candidate over the temporal validation splits (time-series cross-validation).
    *   **Temporal Baselines:** Must include comparison against a naive baseline (e.g., "Naive" = same as yesterday, or "Seasonal Naive" = same time last week/year) to prove the model outperforms simple historical repetition.

**5. Selected Model Rationale (OR Diagnostic Analysis)**
*   **Expected Content:**
    *   **If valid:** Explain why the winning model was chosen (discussing trade-offs like error rates vs. ability to capture sudden trend changes vs. latency). 
    *   **If `leakage_suspected`:** Do NOT present a winner. Retitle this section to **Diagnostic Analysis** and explicitly detail the *lookahead bias*—which future data accidentally leaked into the training features and how to fix the pipeline.

**6. Risks, Caveats & Limitations**
*   **Expected Content:**
    *   Vulnerability to concept drift (e.g., "Model assumes historical seasonal patterns; sudden market shocks will degrade accuracy").
    *   Forecast degradation (how much worse the model gets at the end of the horizon vs. the beginning).
    *   *Mandatory disclosure:* If `quality_flag` is `leakage_suspected`, `subpar`, or `no_viable_candidate`, insert a bolded warning paragraph here.

**7. Next Steps & Recommendations**
*   **Expected Content:**
    *   **Go/No-Go:** Explicit statement on whether the model is ready for production forecasting.
    *   **Action Items:** Bulleted list of concrete next steps (e.g., "Shift the 7-day lag features to prevent lookahead bias," or "Add external weather data to handle holiday spikes").

## Mandatory Leakage Disclosure

- Include a dedicated, highly visible warning paragraph when `quality_flag` is `leakage_suspected`, `subpar`, or `no_viable_candidate`.
- Explicitly state whether the run is production-usable.
- If leakage is suspected, the ELI5 section must use an analogy to explain the issue (e.g., "The model accidentally saw the answer key before taking the test, so its high score is an illusion").