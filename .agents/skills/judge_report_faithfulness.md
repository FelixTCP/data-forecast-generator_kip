# Skill: Report Faithfulness Bewerten

## Zweck

Dieser Skill beschreibt, wie der Quality Judge prueft, ob ein Report artefaktgetreu ist. Er bewertet Aussagen gegen vorhandene Pipeline-Artefakte und ersetzt den Report nicht automatisch.

Primaere Wahrheit bleibt `CURRENT_SYSTEM_DOCUMENTATION.md`.

## Schutzregeln

- Keine Reports ungefragt ueberschreiben.
- Keine Artefakte veraendern.
- Keine Pipeline ersetzen.
- Faithfulness-Bewertung darf hard gates nicht ueberstimmen.
- Bestehende Runs bleiben Audit-Artefakte.

## Zu Pruefen

| Prueffrage | Evidenz |
|---|---|
| Stimmen behauptete Metriken? | `step-14-evaluation.json`, `step-15-selection.json` |
| Wird das ausgewaehlte Modell korrekt benannt? | `step-15-selection.json` |
| Werden Qualitaetsflags korrekt wiedergegeben? | `step-14-evaluation.json`, `step-15-selection.json` |
| Werden Leakage-Ergebnisse korrekt dargestellt? | `leakage_audit.json`, `step-14-evaluation.json.leakage_probe` |
| Werden Datenprobleme nicht verschwiegen? | `step-00_profiler.json`, `step-01-cleanse.json`, `step-11-exploration.json` |
| Werden Empfehlungen durch Artefakte gestuetzt? | Step 11 bis Step 15 |
| Werden keine nicht belegten Versprechen gemacht? | Report gegen alle relevanten Artefakte |
| Wird Unsicherheit transparent benannt? | Report, Quality Flags, Risiken |

## Bewertungsformat

| Feld | Inhalt |
|---|---|
| Faithfulness Score | 1 bis 5 |
| Belegte Aussagen | Report-Aussagen mit Artefakt-Evidenz |
| Fragliche oder unbelegte Aussagen | Aussagen ohne ausreichende Evidenz |
| Erforderliche Revisionen | Konkrete Korrekturen |
| Entscheidungsempfehlung | `accept`, `revise_report` oder `request_human_review` |

## Score-Orientierung

| Score | Bedeutung |
|---:|---|
| 1 | Report widerspricht Artefakten oder verschweigt kritische Risiken |
| 2 | Mehrere wichtige Aussagen sind unbelegt oder ueberzogen |
| 3 | Im Kern korrekt, aber mit relevanten Praezisierungsluecken |
| 4 | Artefaktgetreu mit kleineren Verbesserungen |
| 5 | Vollstaendig, konsistent und transparent belegt |

## Entscheidungsempfehlungen

- `accept`: Report ist artefaktgetreu genug.
- `revise_report`: Report braucht konkrete Korrekturen oder Ergaenzungen.
- `request_human_review`: Aussagen, Risiken oder Business-Annahmen sind nicht sicher maschinell beurteilbar.

