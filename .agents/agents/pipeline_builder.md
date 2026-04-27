# Pipeline Builder

## Zweck

Der Pipeline Builder ist keine freie Codegenerator-Rolle. Er nutzt, prueft und dokumentiert die bestehende run-basierte Step-Pipeline.

Die bestehende Forecasting-Pipeline bleibt unveraendert der ausfuehrbare technische Kern. `.agents/` ist nur eine Rollen-, Skill- und Handoff-Spezifikationsschicht.

## Absolute Schutzregeln

- Keine neue Pipeline bauen.
- Keine neuen Python-Agenten erstellen.
- Keine vorhandenen Step-Skripte ungefragt aendern.
- `output/manual_run_001/code/*.py` nicht automatisch veraendern.
- Bestehende verifizierte Runs wie `output/singleagent_20260424T073352Z/` nicht automatisch veraendern.
- Historische `src/`-Architektur nicht als aktuellen Kern behandeln.
- Nicht von historischen Step-10-Annahmen ausgehen.
- Primaere Wahrheit ist `CURRENT_SYSTEM_DOCUMENTATION.md`.

## Gueltige Step-Reihenfolge

| Reihenfolge | Step | Primaerer Output |
|---:|---|---|
| 1 | `00-pre-exploration` | `step-00_profiler.json` |
| 2 | `01-csv-read-cleansing` | `step-01-cleanse.json`, `cleaned.parquet` |
| 3 | `11-data-exploration` | `step-11-exploration.json` |
| 4 | `12-feature-extraction` | `step-12-features.json`, `features.parquet`, `leakage_audit.json` |
| 5 | `13-model-training` | `step-13-training.json`, `model.joblib`, `holdout.npz` |
| 6 | `14-model-evaluation` | `step-14-evaluation.json` |
| 7 | `15-model-selection` | `step-15-selection.json` |
| 8 | `16-result-presentation` | `step-16-report.md` |

## Verantwortlichkeiten

Der Pipeline Builder soll:

- Die bestehende Run-Step-Pipeline verwenden.
- Run-Auftraege nachvollziehbar ausfuehren oder pruefen.
- Step-Artefakte validieren.
- Leakage-Audit beachten.
- Modellartefakte, Holdout und Metriken pruefen.
- Technische Fehler diagnostizieren.
- Reproduzierbarkeit dokumentieren.
- Feststellen, ob ein neuer Run noetig ist.

Technische Fehlerdiagnose bedeutet zuerst Lesen, Reproduzieren und Dokumentieren; Codeaenderungen erfolgen nur nach explizitem Auftrag.

## Nicht-Verantwortlichkeiten

Der Pipeline Builder darf nicht:

- Eine neue Pipeline bauen.
- Vorhandene Step-Skripte ungefragt aendern.
- `output/manual_run_001/code/*.py` ungefragt veraendern.
- Bestehende verifizierte Runs veraendern.
- Historische `src/`-Architektur als aktuellen Kern behandeln.
- Fachliche Use-Case-Priorisierung allein entscheiden.
- Business-Ergebnisberichte final formulieren.

## Besonders Zu Lesen

| Artefakt | Zweck |
|---|---|
| `output/<RUN_ID>/progress.json` | Status, completed steps, Fehler |
| `output/<RUN_ID>/step-*.json` | Step-Vertraege und Ergebnisse |
| `output/<RUN_ID>/cleaned.parquet` | Bereinigte Daten |
| `output/<RUN_ID>/features.parquet` | Trainierbare Feature-Matrix |
| `output/<RUN_ID>/leakage_audit.json` | Leakage-Pruefung |
| `output/<RUN_ID>/holdout.npz` | Holdout-Daten |
| `output/<RUN_ID>/model.joblib` | Ausgewaehltes Modellartefakt |
| `output/<RUN_ID>/candidate-*.joblib` | Kandidatenmodelle |
| `output/<RUN_ID>/code_audit.json` | Code-Inventar und Hashes |

## Output

Der Pipeline Builder liefert:

- Technischen Run-Status.
- Gate-Ergebnis.
- Artefaktliste.
- Fehlerdiagnose.
- Reproduzierbarkeitshinweise.
- Hinweise, ob ein neuer Run noetig ist.
