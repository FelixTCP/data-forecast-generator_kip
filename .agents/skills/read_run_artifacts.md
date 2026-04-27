# Skill: Run-Artefakte Lesen

## Zweck

Dieser Skill beschreibt, wie Agenten ein Run-Verzeichnis der bestehenden Pipeline lesen. Er erzeugt keine neue Runtime und ersetzt keine Pipeline.

Primaere Wahrheit ist `CURRENT_SYSTEM_DOCUMENTATION.md`.

## Schutzregeln

- Diese Skill-Datei beschreibt Verhalten fuer Phase 1.
- Phase 1 erzeugt keine neue Runtime und keine Python-Agenten.
- Die bestehende Step-Pipeline wird nicht ersetzt.
- Bestehende Run-Verzeichnisse sind Audit-Artefakte.
- Aenderungen an Pipeline-Code oder Run-Artefakten erfolgen nur nach explizitem Auftrag.
- Nur lesen, sofern kein expliziter Auftrag zum Erzeugen eines neuen Runs vorliegt.
- `output/manual_run_001/code/*.py` ist Referenz-Code und darf nicht automatisch veraendert werden.
- Nicht von historischen Step-10- oder `src/`-Annahmen ausgehen.

## Erwartete Run-Struktur

```text
output/<RUN_ID>/
├── code/
├── progress.json
├── cleaned.parquet
├── features.parquet
├── holdout.npz
├── model.joblib
├── candidate-*.joblib
├── leakage_audit.json
├── code_audit.json
├── step-*.json
└── step-16-report.md
```

## Zentrale Artefakte

| Artefakt | Bedeutung |
|---|---|
| `progress.json` | Run-Status, Steps, Fehler |
| `step-00_profiler.json` | CSV-Strukturprofil |
| `step-00_data_profile_report.md` | Profiling-Report |
| `step-01-cleanse.json` | Cleansing-Vertrag und Zielspalte |
| `cleaned.parquet` | Bereinigter Datensatz |
| `step-11-exploration.json` | Feature- und Target-Signale |
| `step-12-features.json` | Feature-Vertrag |
| `features.parquet` | Trainierbare Feature-Matrix |
| `leakage_audit.json` | Leakage-Pruefung |
| `step-13-training.json` | Training und Kandidaten |
| `holdout.npz` | Holdout-Daten |
| `model.joblib` | Ausgewaehltes Modell |
| `candidate-*.joblib` | Kandidatenmodelle |
| `step-14-evaluation.json` | Evaluation |
| `step-15-selection.json` | Auswahlentscheidung |
| `step-16-report.md` | Finaler Report |
| `code_audit.json` | Code-Inventar und Hashes |

## Empfohlene Lesereihenfolge

1. `CURRENT_SYSTEM_DOCUMENTATION.md`
2. `output/<RUN_ID>/progress.json`
3. `output/<RUN_ID>/code_audit.json`
4. `output/<RUN_ID>/step-00_profiler.json`
5. `output/<RUN_ID>/step-01-cleanse.json`
6. `output/<RUN_ID>/step-11-exploration.json`
7. `output/<RUN_ID>/step-12-features.json`
8. `output/<RUN_ID>/leakage_audit.json`
9. `output/<RUN_ID>/step-13-training.json`
10. `output/<RUN_ID>/step-14-evaluation.json`
11. `output/<RUN_ID>/step-15-selection.json`
12. `output/<RUN_ID>/step-16-report.md`

## Rollenbedarf

| Rolle | Wichtigste Artefakte |
|---|---|
| Conductor | `progress.json`, `step-*.json`, `leakage_audit.json`, `step-16-report.md` |
| Opportunity Analyst | `step-00_profiler.json`, `step-00_data_profile_report.md`, `step-01-cleanse.json`, `step-11-exploration.json` |
| Pipeline Builder | Alle Step-Artefakte, Parquet-Dateien, Joblib-Dateien, `holdout.npz`, `code_audit.json` |
| Result Interpreter | `step-11` bis `step-16`, Selection-Report, Leakage-Audit |

## Umgang Mit Fehlenden Artefakten

- Fehlende Artefakte immer dokumentieren.
- Nicht automatisch rekonstruieren oder ueberschreiben.
- Pruefen, ob der Run unvollstaendig, fehlgeschlagen oder historisch ist.
- Conductor informieren, wenn ein neuer Run sinnvoller ist als eine Interpretation des alten Zustands.

## Referenz-Code, Run-Code Und Run-Artefakte

| Kategorie | Beispiel | Regel |
|---|---|---|
| Referenz-Code | `output/manual_run_001/code/*.py` | Nicht automatisch aendern |
| Run-Code | `output/<RUN_ID>/code/*.py` | Teil eines konkreten Runs, nur mit Auftrag aendern |
| Run-Artefakte | `output/<RUN_ID>/step-*.json`, `model.joblib` | Audit- und Ergebnisartefakte, nicht ungefragt veraendern |
