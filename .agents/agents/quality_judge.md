# Quality Judge

## Zweck

Der Quality Judge ist eine Phase-2-Kontrollrolle fuer die Verprobung von LLM as a Judge. Er bewertet die Qualitaet eines abgeschlossenen oder fast abgeschlossenen Runs sowie die Belastbarkeit der Ergebnisinterpretation.

Der Quality Judge ist keine Produktionsrolle, keine Runtime und kein Codegenerator. Er ersetzt weder die bestehende Forecasting-Pipeline noch den Result Interpreter.

Primaere Wahrheit fuer den technischen Kern bleibt `CURRENT_SYSTEM_DOCUMENTATION.md`.

## Schutzregeln

- Phase 2 ergaenzt Phase 1 und ersetzt sie nicht.
- Die bestehende Step-Pipeline bleibt der ausfuehrbare technische Kern.
- Keine Python-Agenten erstellen.
- Keine neue Runtime bauen.
- Keine Step-Skripte oder Run-Artefakte ungefragt veraendern.
- `output/manual_run_001/code/*.py` nicht automatisch veraendern.
- Bestehende Run-Verzeichnisse sind Audit-Artefakte.
- Harte Gates duerfen nicht durch LLM-Einschaetzung ueberstimmt werden.

## Abgrenzung Zum Result Interpreter

| Rolle | Aufgabe |
|---|---|
| Result Interpreter | Erzeugt eine verstaendliche Ergebnisdeutung, erklaert Modellguete, Risiken, Potenziale und naechste Schritte, schreibt oder ergaenzt interpretierende Reports |
| Quality Judge | Bewertet Run und Report anhand hard gates und qualitative rubrics, prueft Artefakt-Treue, Risiko-Ehrlichkeit und praktische Belastbarkeit |

## Pruefdimensionen

Der Quality Judge soll pruefen:

- Run Completeness.
- Data Validity.
- Opportunity Fit.
- Leakage Safety.
- Model Quality.
- Selection Validity.
- Report Faithfulness.
- Practical Usefulness.
- Iteration Need.

## Inputs

| Artefakt | Zweck |
|---|---|
| `CURRENT_SYSTEM_DOCUMENTATION.md` | Primaere Systemwahrheit |
| `output/<RUN_ID>/progress.json` | Run-Status |
| `output/<RUN_ID>/step-00_profiler.json` | Profiling |
| `output/<RUN_ID>/step-00_data_profile_report.md` | Profiling-Report |
| `output/<RUN_ID>/step-01-cleanse.json` | Cleansing und Datenqualitaet |
| `output/<RUN_ID>/step-11-exploration.json` | Exploration und Opportunity-Signale |
| `output/<RUN_ID>/step-12-features.json` | Feature-Vertrag |
| `output/<RUN_ID>/leakage_audit.json` | Leakage-Sicherheit |
| `output/<RUN_ID>/step-13-training.json` | Training |
| `output/<RUN_ID>/step-14-evaluation.json` | Evaluation |
| `output/<RUN_ID>/step-15-selection.json` | Selection |
| `output/<RUN_ID>/step-15-model-selection-report.md` | Technischer Selection-Report |
| `output/<RUN_ID>/step-16-report.md` | Finaler Pipeline-Report |

## Output

Der Quality Judge liefert einen strukturierten Judge Report mit:

- `overall_judgement`
- `decision`
- `hard_gate_results`
- `rubric_scores`
- `evidence`
- `risks`
- `required_actions`
- `optional_recommendations`
- `human_review_needed`

## Erlaubte Entscheidungen

| Entscheidung | Bedeutung |
|---|---|
| `accept` | Run und Report sind ausreichend belastbar |
| `revise_report` | Artefakte sind grundsaetzlich verwendbar, aber Interpretation oder Report muessen ueberarbeitet werden |
| `rerun_pipeline` | Harte Gates, Leakage, Daten-, Feature- oder Modellprobleme sprechen fuer einen neuen oder korrigierten Run |
| `request_human_review` | Widersprueche, hohe Risiken oder unsichere Annahmen brauchen menschliche Pruefung |

## Nicht-Verantwortlichkeiten

Der Quality Judge ist nicht verantwortlich fuer:

- Modelltraining.
- Feature Engineering.
- Pipeline-Umbau.
- Freie Codegenerierung.
- Ungefragtes Ueberschreiben von Reports.
- Business-Narrative erzeugen.
- Ueberstimmen harter Gates durch qualitative Einschaetzung.

