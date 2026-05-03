# Data Forecast Generator

Run-basierte Forecasting- und Regressionspipeline fuer CSV-Daten.

Der Data Forecast Generator verarbeitet Kundendaten als CSV, erzeugt daraus eine trainierbare Feature-Matrix, bewertet mehrere Regressionsmodelle und schreibt ein vollstaendiges Ergebnisartefakt mit Modell, Metriken und Markdown-Report.

## Motivation

Kunden liefern haeufig Produktions- oder Betriebsdaten als CSV und suchen Optimierungspotenzial, ohne den konkreten Analyse- oder Forecasting-Use-Case bereits klar benennen zu koennen.

Der Data Forecast Generator soll aus einer CSV-Datei, einer Zielspalte und Laufparametern automatisch eine Regression- bzw. Forecasting-Pipeline erzeugen, trainieren, bewerten und als wiederverwendbares Artefakt dokumentieren.

## Workflow

1. CSV-Cleansing
2. Datenexploration
3. Feature Engineering mit Leakage-Pruefung
4. Training mehrerer Modellkandidaten
5. Evaluation
6. Modellauswahl
7. Ergebnisreport

## Artefakte

Ein Run liegt unter `output/<RUN_ID>/` und enthaelt typischerweise:

- `progress.json`
- `cleaned.parquet`
- `features.parquet`
- `leakage_audit.json`
- `candidate-*.joblib`
- `model.joblib`
- `holdout.npz`
- `step-*.json`
- `step-16-report.md`
- `code_audit.json`

## Verifizierter Beispiel-Run

Der dokumentierte Referenzlauf nutzt:

- CSV: `data/appliances_energy_prediction.csv`
- Target: `appliances`
- Run: `output/singleagent_20260424T073352Z`
- ausgewaehltes Modell: `ridge`
- Qualitaetsflag: `acceptable`
- R2: `0.5668829594991238`
- RMSE: `59.56329686814976`
- MAE: `28.412928284580204`

## Modellartefakt pruefen

```bash
uv run --no-sync python - <<'PY'
import joblib

model = joblib.load("output/manual_run_001/model.joblib")
print(type(model))
print(hasattr(model, "predict"))
PY
```

## Streamlit-Apps

```bash
uv run streamlit run scripts/streamlit_single_agent_app.py
uv run streamlit run scripts/streamlit_inference_app.py
```

## Roadmap

- Phase 1: Agentische Pipeline fuer CSV-Cleansing, Exploration, Feature Engineering, Training, Evaluation und Reporting
- Phase 2: Qualitaetsbewertung mit robusten Metriken, Vergleichsbaselines und optionalem LLM-as-Judge
- Phase 3: Produktisierung mit Frontend, FastAPI-Server, Datei-Workspace und Sandbox-Ausfuehrung

## Dokumentation

- [SYSTEM-DOCUMENTATION.md](SYSTEM-DOCUMENTATION.md) - Systemarchitektur, Step-Ablauf und Validierungsgates
- [docs/agentic-pipeline/contracts.md](docs/agentic-pipeline/contracts.md) - Runtime-Vertraege
- [docs/agentic-pipeline/setup-prompt.md](docs/agentic-pipeline/setup-prompt.md) - Setup-Prompt fuer agentische Runs
- [docs/agentic-pipeline/step-prompts.md](docs/agentic-pipeline/step-prompts.md) - Runtime-Step-Prompts
- [docs/pipeline-framework/](docs/pipeline-framework/) - Step-Spezifikationen
