# Projekt: Data Forecast Generator

## Projektbeschreibung
Kunden kommen oft mit Produktionsdaten (“ein csv“) und wünschen eine Optimierung ihres Systems ohne genauere Vorstellung.
Die Idee ist es ein LLM die csv Analysieren zu lassen, um darin etwaige Use Cases für Regression-Forecasts und Analysis zu entdecken.
Genauer:
Entdecken des “profitabelsten” Data-Analysis-Use Cases
Entwicklung einer Regression Pipeline (mit scikit-learn)
Implementieren von Tests
Ausführen der Pipeline auf csv
Auswerten des Ergebnis der Pipeline
Das Artefakt ist ein vollständiges Paket mit gefitteten Model-Artefakt, Auswertung und Potential-Analyse.

### PHASE 1: Claude-Template oder GitHub-Copilot-Template 
Single Prompt, Repo-Template (Readme.md, Ordnerstruktur, project.toml, code-snippets….)
### PHASE 2: Auswerten der Qualität der erzeugten MVPS’s/ Projekte mit geeigneten Metriken 
(wie gut passt der abgeleitete Solver auf die Daten und mögliche Business Use Cases), vielleicht sogar LLM-as-Judge?
### PHASE 3: Frontend+FAST-API Server mit eigenem Copilot 
(z.B. langgraph deepagents mit Filestyme und Sandbox)
