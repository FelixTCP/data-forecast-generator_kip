# Workflow: Phase 2 Judged Forecast Run

## Zweck

Phase 2 ergaenzt Phase 1 um den Quality Judge fuer LLM as a Judge. Die bestehende Forecasting-Pipeline wird nicht ersetzt.

Der Judge ist keine Runtime, kein Codegenerator und keine Produktionsrolle. LLM-Judgement dient qualitativer Bewertung, nicht zum Ueberstimmen technischer Fehler.

Primaere Wahrheit bleibt `CURRENT_SYSTEM_DOCUMENTATION.md`.

## Schutzregeln

- Die Step-Pipeline bleibt der ausfuehrbare technische Kern.
- Keine Python-Agenten erstellen.
- Keine neue Runtime bauen.
- Keine Step-Skripte, Daten, Modelle oder Run-Artefakte ungefragt veraendern.
- Bestehende Run-Verzeichnisse sind Audit-Artefakte.
- Hard gates haben Vorrang vor qualitative rubrics.
- Der Result Interpreter ist nicht der Quality Judge.

## Ablauf

| Schritt | Rolle | Aufgabe | Output |
|---:|---|---|---|
| 1 | Conductor | Kontext sammeln | Run-Kontext, Eingaben, offene Fragen |
| 2 | Opportunity Analyst | Use Case und Target pruefen oder empfehlen | Opportunity Report |
| 3 | Conductor | Verwertbarkeit pruefen | Handoff an Pipeline Builder |
| 4 | Pipeline Builder | Bestehende Pipeline nutzen oder vorhandenen Run pruefen | Technischer Run-Status, Gate-Ergebnis |
| 5 | Result Interpreter | Ergebnisdeutung erzeugen | Interpretation oder Report-Ergaenzung |
| 6 | Quality Judge | Run und Report bewerten | Judge Report |
| 7 | Conductor | Auf Basis des Judge Reports entscheiden | `accept`, `revise_report`, `rerun_pipeline`, `request_human_review` |
| 8 | Rollen nach Auftrag | Revision, Rerun oder erneute Use-Case-Pruefung | Aktualisierte Ergebnisse nur nach Auftrag |
| 9 | Conductor | Finalen Status konsolidieren | Abschlussstatus und naechste Schritte |

## Schritt 1: Conductor Sammelt Kontext

Der Conductor klaert:

- CSV-Pfad.
- Target-Status.
- Run-ID oder Bedarf fuer neuen Run.
- Vorhandene Artefakte.
- Ob Phase-2-Judging gewuenscht ist.

## Schritt 2: Opportunity Analyst Prueft Oder Empfiehlt

Der Opportunity Analyst bewertet Use Case und Target. Er trainiert keine Modelle und aendert keine Pipeline.

## Schritt 3: Conductor Prueft Verwertbarkeit

Der Conductor prueft, ob die Opportunity-Empfehlung mit der bestehenden Pipeline ausfuehrbar oder pruefbar ist.

## Schritt 4: Pipeline Builder Nutzt Bestehende Pipeline

Der Pipeline Builder nutzt die bestehende Step-Pipeline oder prueft einen vorhandenen Run. Er baut keine neue Pipeline.

## Schritt 5: Result Interpreter Erzeugt Ergebnisdeutung

Der Result Interpreter erstellt eine verstaendliche Interpretation. Er ist nicht der Judge und entscheidet nicht ueber Akzeptanz.

## Schritt 6: Quality Judge Bewertet Run Und Report

Der Quality Judge prueft:

- Hard gates.
- Qualitative rubrics.
- Report Faithfulness.
- Risiko-Ehrlichkeit.
- Praktische Belastbarkeit.

Hard gates haben Vorrang. LLM-Judgement darf technische Fehler nicht ueberstimmen.

## Schritt 7: Conductor Entscheidet

Moegliche Entscheidungen:

| Entscheidung | Bedeutung |
|---|---|
| `accept` | Run und Report sind ausreichend belastbar |
| `revise_report` | Interpretation oder Report brauchen Ueberarbeitung |
| `rerun_pipeline` | Neuer oder korrigierter Run ist noetig |
| `request_human_review` | Menschliche Pruefung ist erforderlich |

## Schritt 8: Revision Nur Nach Auftrag

Falls Revision noetig ist:

- Result Interpreter ueberarbeitet Interpretation nur nach Auftrag.
- Pipeline Builder prueft oder wiederholt Run nur nach Auftrag.
- Opportunity Analyst prueft Target oder Use Case erneut nur nach Auftrag.
- Bestehende Runs bleiben Audit-Artefakte.

## Schritt 9: Conductor Konsolidiert Finalen Status

Der Conductor dokumentiert:

- Judge-Entscheidung.
- Hard-gate-Ergebnisse.
- Qualitative Risiken.
- Erforderliche Aktionen.
- Offene Human-Review-Punkte.
- Naechste Schritte.

## Handoff-Vertrag

| Handoff | Mindestinhalt |
|---|---|
| Conductor -> Quality Judge | Run-Pfad, Interpretation/Report, Gate-Zusammenfassung, offene Risiken |
| Quality Judge -> Conductor | Judge Report mit Entscheidung und Evidenz |
| Conductor -> Result Interpreter | Konkrete Report-Revisionen bei `revise_report` |
| Conductor -> Pipeline Builder | Konkreter Rerun- oder Pruefauftrag bei `rerun_pipeline` |
| Conductor -> Opportunity Analyst | Konkrete Target-/Use-Case-Fragen bei Bedarf |

