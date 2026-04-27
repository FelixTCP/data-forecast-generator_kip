# Skill: Run-Qualitaet Bewerten

## Zweck

Dieser Skill beschreibt, wie der Quality Judge die Qualitaet eines Pipeline-Runs bewertet. Er ist Teil der Phase-2-Spezifikationsschicht fuer LLM as a Judge und erzeugt keine Runtime.

Primaere Wahrheit bleibt `CURRENT_SYSTEM_DOCUMENTATION.md`.

## Schutzregeln

- Phase 2 ergaenzt Phase 1.
- Die bestehende Step-Pipeline wird nicht ersetzt.
- Keine Python-Agenten erstellen.
- Keine Step-Skripte, Daten, Modelle oder Run-Artefakte ungefragt veraendern.
- Bestehende Run-Verzeichnisse bleiben Audit-Artefakte.
- Hard gates haben Vorrang vor qualitative rubrics.

## A. Hard Gates

Hard gates sind regelbasiert und nicht durch LLM-Meinung ersetzbar.

| Gate | Erwartung |
|---|---|
| Run-Status | `progress.json.status = completed` |
| Completed Steps | `completed_steps` enthaelt alle erwarteten Steps |
| Artefakte | Zentrale Artefakte sind vorhanden |
| JSON | JSON-Dateien sind parsebar |
| Leakage | `leakage_audit.status = pass` |
| Modell | `model.joblib` vorhanden, wenn ein Modell gewaehlt wurde |
| Holdout | `holdout.npz` vorhanden |
| Metriken | Step 14 enthaelt endliche Metriken |
| Selection | Keine Modellwahl bei negativem R2 oder `no_viable_candidate` |
| Report | Report mit Pflichtabschnitten vorhanden |

Erwartete Steps:

1. `00-pre-exploration`
2. `01-csv-read-cleansing`
3. `11-data-exploration`
4. `12-feature-extraction`
5. `13-model-training`
6. `14-model-evaluation`
7. `15-model-selection`
8. `16-result-presentation`

## B. Qualitative Rubrics

Diese Punkte werden als LLM-as-a-Judge-Aufgabe bewertet, aber nur nach bestandenen oder dokumentiert eingeordneten hard gates:

- Ist die Modellauswahl durch die Metriken gedeckt?
- Ist die Qualitaetsbewertung angemessen?
- Wird die Modellguete nicht ueberverkauft?
- Sind Risiken und Caveats vollstaendig genug?
- Sind Empfehlungen praktisch sinnvoll?
- Gibt es Hinweise auf Daten-, Target- oder Feature-Probleme?
- Ist eine Wiederholung des Runs sinnvoll?

## Bewertungsformat

Jede Rubric wird bewertet mit:

| Feld | Inhalt |
|---|---|
| `score` | 1 bis 5 |
| `rationale` | Kurze Begruendung |
| `evidence` | Konkrete Artefakte und Felder |
| `uncertainty` | Unsicherheit oder fehlende Evidenz |

Score-Orientierung:

| Score | Bedeutung |
|---:|---|
| 1 | Stark problematisch oder nicht belegt |
| 2 | Schwach, relevante Luecken |
| 3 | Ausreichend, aber mit klaren Einschraenkungen |
| 4 | Gut belegt, kleinere Restfragen |
| 5 | Sehr gut belegt und konsistent |

## Output

Der Skill liefert:

- Hard-gate-Ergebnis je Gate.
- Rubric-Scores mit Evidenz.
- Risiken.
- Unsicherheiten.
- Entscheidungsempfehlung: `accept`, `revise_report`, `rerun_pipeline` oder `request_human_review`.

