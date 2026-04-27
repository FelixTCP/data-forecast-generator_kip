# Skill: Step-Outputs Validieren

## Zweck

Dieser Skill beschreibt die Validierungsgates fuer die bestehende Step-Pipeline. Er ist eine Markdown-Spezifikation und keine neue Runtime.

Primaere Wahrheit ist `CURRENT_SYSTEM_DOCUMENTATION.md`.

## Allgemeine Schutzregeln

- Diese Skill-Datei beschreibt Verhalten fuer Phase 1.
- Phase 1 erzeugt keine neue Runtime und keine Python-Agenten.
- Die bestehende Pipeline wird nicht ersetzt.
- Bestehende Run-Verzeichnisse sind Audit-Artefakte.
- Aenderungen an Pipeline-Code oder Run-Artefakten erfolgen nur nach explizitem Auftrag.
- Keine Step-Skripte oder Run-Artefakte automatisch veraendern.
- Nicht von historischen Step-10- oder `src/`-Annahmen ausgehen.
- Gates pruefen vorhandene Artefakte und dokumentieren Befunde.

## Basischecks

| Check | Erwartung |
|---|---|
| Run-Verzeichnis | `output/<RUN_ID>/` existiert |
| Progress | `progress.json` ist parsebar |
| Step-Feld | Jede Step-JSON enthaelt ein plausibles `"step"` |
| Fehlerliste | `progress.json.errors` pruefen |
| Leakage | `leakage_audit.json` beachten |
| Completed steps | Bei abgeschlossenem Run enthaelt `progress.json.completed_steps` alle acht erwarteten Steps |
| Code-Audit | `code_audit.json` existiert bei vollstaendig abgeschlossenem Run oder die Abwesenheit wird als Hinweis dokumentiert |

Erwartete Steps bei abgeschlossenem Run:

1. `00-pre-exploration`
2. `01-csv-read-cleansing`
3. `11-data-exploration`
4. `12-feature-extraction`
5. `13-model-training`
6. `14-model-evaluation`
7. `15-model-selection`
8. `16-result-presentation`

## Step 00: Pre-Exploration

- `output/<RUN_ID>/step-00_profiler.json` existiert.
- `output/<RUN_ID>/step-00_data_profile_report.md` existiert.
- Header-/Spalteninformationen sind vorhanden.
- Beispielzeilen oder Row-Summary sind vorhanden.
- Grobe Typinformationen sind vorhanden.
- Auffaelligkeiten oder Profiling-Hinweise sind dokumentiert, falls erkannt.

## Step 01: CSV Read And Cleansing

- `output/<RUN_ID>/step-01-cleanse.json` existiert.
- `row_count_after > 0`.
- `target_column_normalized` ist gesetzt.
- `output/<RUN_ID>/cleaned.parquet` existiert.
- `artifacts.cleaned_parquet` verweist auf einen vorhandenen Pfad.

## Step 11: Data Exploration

- `output/<RUN_ID>/step-11-exploration.json` existiert.
- `numeric_columns` ist nicht leer.
- `mi_ranking` ist nicht leer.
- `recommended_features` ist nicht leer.
- `noise_mi_baseline` ist endlich.
- `excluded_features` ist nachvollziehbar dokumentiert.

## Step 12: Feature Extraction

- `output/<RUN_ID>/step-12-features.json` existiert.
- `features` ist nicht leer.
- `output/<RUN_ID>/features.parquet` existiert.
- `output/<RUN_ID>/leakage_audit.json` existiert.
- `leakage_audit.status = pass`.
- Ausgeschlossene Features aus Step 11 werden nicht wieder eingefuehrt.
- `split_strategy.resolved_mode` ist plausibel.

## Step 13: Model Training

- `output/<RUN_ID>/step-13-training.json` existiert.
- `output/<RUN_ID>/model.joblib` existiert und ist ladbar.
- `output/<RUN_ID>/holdout.npz` existiert.
- Mindestens ein Kandidat hat einen endlichen R2-Wert.
- `feature_names` ist vorhanden.
- Kandidatenartefakte aus `candidate-*.joblib` sind vorhanden, soweit in der JSON referenziert.
- Alle in `step-13-training.json` referenzierten Kandidatenartefakte existieren.
- `candidate-*.joblib` sind optional ladbar, wenn eine technische Pruefung moeglich ist.
- `holdout.npz` enthaelt erwartete Holdout-Daten, mindestens Test-Features und Test-Target.
- Falls konkrete Array-Namen im Holdout abweichen, wird dies dokumentiert statt stillschweigend angenommen.

## Step 14: Model Evaluation

- `output/<RUN_ID>/step-14-evaluation.json` existiert.
- Alle Kandidaten enthalten endliche `r2`, `rmse`, `mae`.
- `quality_assessment` ist gesetzt.
- Erlaubte Werte fuer `quality_assessment`: `acceptable`, `marginal`, `subpar`, `subpar_after_expansion`, `leakage_suspected`, falls vom Step gesetzt.
- `target_stats` ist gesetzt.
- Leakage-Probe beachten.
- Bei `quality_assessment = leakage_suspected` darf keine produktive Auswahl als gueltig gelten.
- Bei negativen R2-Werten muss die Einordnung als schlechter als Mean-Baseline beachtet werden.
- Bei `subpar` oder `subpar_after_expansion` muss `expansion_diagnosis` beruecksichtigt werden.

## Step 15: Model Selection

- `output/<RUN_ID>/step-15-selection.json` existiert.
- `quality_flag` ist gesetzt.
- `full_ranking` ist vorhanden.
- `selected_model` ist nur bei viablem Ergebnis gesetzt.
- Bei `quality_flag = no_viable_candidate` darf kein Gewinner behauptet werden.
- `quality_flag` kann auch einen Zustand wie `no_viable_candidate` ausdruecken, sofern kein Kandidat sinnvoll auswaehlbar ist.
- `step-15-model-selection-report.md` existiert.
- `step-15-model-selection-metrics.png` existiert, wenn Kandidaten vorhanden sind und Plot-Erzeugung moeglich war.
- `baselines` ist vorhanden.
- `candidate_analysis` ist vorhanden.

## Step 16: Result Presentation

- `output/<RUN_ID>/step-16-report.md` existiert.
- `step-16-report.md` ist mindestens 500 Bytes gross.
- Report enthaelt sechs Pflichtabschnitte:
  1. Problem + selected target
  2. Data quality summary
  3. Candidate models + scores table
  4. Selected model rationale
  5. Risks and caveats
  6. Next iteration recommendations
- `progress.json.status = completed`.
- `progress.json.current_step` ist plausibel, idealerweise `16-result-presentation`.

## Output Des Validierungs-Skills

Eine Validierungszusammenfassung soll enthalten:

- Run-ID.
- Gepruefte Artefakte.
- Bestanden / nicht bestanden je Step.
- Fehlende oder fehlerhafte Dateien.
- Leakage-Status.
- Modellartefakt-Status.
- Finale Einschaetzung: `complete`, `incomplete`, `failed`, `requires_new_run`.
