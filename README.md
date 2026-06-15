# Data Forecast Generator

Run-basierte Forecasting- und Regressionspipeline fuer CSV-Daten.

Der Data Forecast Generator verarbeitet Kundendaten als CSV, fuehrt einen agentischen 10-18 Step-Workflow aus, trainiert mehrere Regressionsmodelle und schreibt pro Run ein vollstaendiges Artefaktpaket mit Modell, Metriken, Audit und Bericht.

## Motivation

Kunden liefern haeufig Produktions- oder Betriebsdaten als CSV und suchen Optimierungspotenzial, ohne den konkreten Analyse- oder Forecasting-Use-Case bereits klar benennen zu koennen.

Der Data Forecast Generator soll aus CSV-Datei, Zielspalte und Laufparametern automatisch eine robuste Regression- bzw. Forecasting-Pipeline ausfuehren und reproduzierbar dokumentieren.

## Workflow

1. CSV Read/Cleansing (Step 10)
2. Datenexploration (Step 11)
3. Feature Extraction inkl. Leakage-Pruefung (Step 12)
4. Training mehrerer Modellkandidaten (Step 13)
5. Evaluation inkl. Baselines und Qualitaetsflag (Step 14)
6. Modellauswahl (Step 15)
7. Ergebnisreport (Step 16)
8. Kritischer Self-Audit (Step 17)
9. Executive Summary (Step 18)

Nach Step 18 kann die Streamlit-App optional den separaten Post Run Judge Agent starten. Der Judge ist kein Pipeline-Step, sondern bewertet den fertigen Run aus externer Sicht.

## Artefakte

Ein Run liegt unter `output/<RUN_ID>/` und enthaelt typischerweise:

- `progress.json`
- `cleaned.parquet`
- `features.parquet`
- `leakage_audit.json`
- `candidate-*.joblib`
- `model.joblib`
- `holdout.npz`
- `step-10-cleanse.json` bis `step-18-executive-summary.json`
- `step-16-report.md`
- `step-18-executive-summary.md`
- `code_audit.json`
- `step-17-audit.json`
- optional `judge.json` und `judge.md`

## Schnellstart

### Option A: Docker Compose (empfohlen)

1. `.env.example` nach `.env` kopieren und `GH_TOKEN` setzen.
2. Container starten:

```bash
docker compose up --build
```

Danach:

- Agent-App (Training/Run-Steuerung): `http://localhost:8501`
- Inference-App (Modell laden, Prognosen, XAI): `http://localhost:8502`

Wichtig: Fuer Agent-Runs ist ein gueltiger GitHub Token Pflicht. Ohne `GH_TOKEN` kann die Agent-App keine Pipeline-Schritte ueber Copilot CLI ausfuehren.

### Option B: Lokal mit uv

Abhaengigkeiten synchronisieren:

```bash
uv sync --no-install-project
```

Falls `uv` unter Windows nicht im PATH liegt, nutze stattdessen:

```powershell
.venv\Scripts\uv.exe sync --no-install-project
```

Apps starten:

```bash
uv run streamlit run scripts/streamlit_single_agent_app.py
uv run streamlit run scripts/streamlit_inference_app.py --server.port 8502
```

Auch lokal gilt: Fuer Runs ueber die Agent-App muss ein gueltiger GitHub Token verfuegbar sein (z. B. als `GH_TOKEN` in der Umgebung).

## Was beim Start ungefaehr ablaeuft

1. CSV hochladen oder vorhandene CSV aus `data/` verwenden.
2. Zielspalte waehlen (oder automatisch vorschlagen lassen).
3. Run starten (ueber Copilot CLI oder Codex CLI aus der Agent-App).
4. Die Pipeline schreibt Schritt fuer Schritt Artefakte nach `output/<RUN_ID>/`.
5. Nach Abschluss kannst du im gleichen Run die Ergebnisse pruefen:
	- Modell und Kandidaten
	- Metriken und Ranking
	- Markdown-Report
	- Self-Audit und optional Judge
6. In der Inference-App kannst du bestehende Runs laden und Vorhersagen/XAI analysieren.

## Modellartefakt pruefen

```bash
uv run --no-sync python -c "import joblib; m=joblib.load('output/<RUN_ID>/model.joblib'); print(type(m)); print(hasattr(m, 'predict'))"
```

## Roadmap

- Phase 1: Agentische Pipeline fuer CSV-Cleansing, Exploration, Feature Engineering, Training, Evaluation und Reporting
- Phase 2: Qualitaetsbewertung mit robusten Metriken, Vergleichsbaselines und Post-run Judge-Bewertung
- Phase 3: Produktisierung mit Frontend, FastAPI-Server, Datei-Workspace und Sandbox-Ausfuehrung

## Dokumentation

- [SYSTEM-DOCUMENTATION.md](SYSTEM-DOCUMENTATION.md) - Systemarchitektur, Step-Ablauf und Validierungsgates
- [docs/agentic-pipeline/contracts.md](docs/agentic-pipeline/contracts.md) - Runtime-Vertraege
- [docs/agentic-pipeline/setup-prompt.md](docs/agentic-pipeline/setup-prompt.md) - Setup-Prompt fuer agentische Runs
- [docs/agentic-pipeline/step-prompts.md](docs/agentic-pipeline/step-prompts.md) - Runtime-Step-Prompts
- [docs/pipeline-framework/](docs/pipeline-framework/) - Step-Spezifikationen
