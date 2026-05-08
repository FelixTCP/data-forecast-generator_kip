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

## Docker Quickstart

Docker Compose ist der primaere Startweg. Die Training-App laeuft in Streamlit
und ruft innerhalb des Containers die GitHub Copilot CLI mit dem Custom Agent
`Single Agent Pipeline` auf.

```bash
cp .env.example .env
# GH_TOKEN in .env eintragen
mkdir -p output artifacts/ui_uploads
docker compose up --build
```

- Training UI: `http://localhost:8501`
- Inference UI: `http://localhost:8502`

Wenn Docker die Bind-Mount-Verzeichnisse vorher als `root` angelegt hat und
Uploads fehlschlagen, einmalig die Ownership korrigieren:

```bash
sudo chown -R "$USER":"$USER" artifacts output
```

Optional kann `COPILOT_COMPLETION_GRACE_SECONDS` in `.env` gesetzt werden. Wenn
`progress.json` bereits `completed` meldet, beendet Streamlit nach dieser Frist
einen noch laufenden Copilot-Wrapper-Prozess.

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
docker compose exec agent-app python - <<'PY'
import joblib

model = joblib.load("output/manual_run_001/model.joblib")
print(type(model))
print(hasattr(model, "predict"))
PY
```

## Lokaler Debug-Weg

Fuer lokale Entwicklung ohne Container kann `uv` weiterhin genutzt werden. Der
reproduzierbare Standardlauf bleibt Docker Compose.

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
