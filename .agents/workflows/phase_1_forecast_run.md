# Workflow: Phase 1 Forecast Run

## Zweck

Dieser Workflow beschreibt den Phase-1-Hauptablauf fuer Rollen, Skills und Handoffs. Die `.agents/`-Schicht ist ein Spezifikationsvertrag und erzeugt keine neue Runtime.

Die bestehende Forecasting-Pipeline wird nicht ersetzt. Die vorhandenen Step-Skripte unter `output/<RUN_ID>/code/*.py` bleiben der ausfuehrbare technische Kern.

Primaere Wahrheit ist `CURRENT_SYSTEM_DOCUMENTATION.md`.

## Schutzregeln

- Phase 1 erzeugt keine neuen Python-Agenten.
- Keine neue Pipeline bauen.
- Keine Step-Skripte ungefragt veraendern.
- `output/manual_run_001/code/*.py` nicht automatisch veraendern.
- Bestehende Run-Verzeichnisse wie `output/singleagent_20260424T073352Z/` sind Audit-Artefakte und duerfen nicht automatisch veraendert werden.
- Nicht von historischen Step-10- oder `src/`-Annahmen ausgehen.

## Gueltige Step-Reihenfolge

| Reihenfolge | Step |
|---:|---|
| 1 | `00-pre-exploration` |
| 2 | `01-csv-read-cleansing` |
| 3 | `11-data-exploration` |
| 4 | `12-feature-extraction` |
| 5 | `13-model-training` |
| 6 | `14-model-evaluation` |
| 7 | `15-model-selection` |
| 8 | `16-result-presentation` |

## Ablauf

| Schritt | Rolle | Aufgabe | Output |
|---:|---|---|---|
| 1 | Conductor | Kontext sammeln | Run-Kontext, Eingaben, offene Fragen |
| 2 | Opportunity Analyst | Use Case und Target pruefen oder empfehlen | Opportunity Report |
| 3 | Conductor | Empfehlung auf Verwertbarkeit pruefen | Handoff an Pipeline Builder |
| 4 | Pipeline Builder | Bestehende Pipeline nutzen oder vorhandenen Run pruefen | Technischer Run-Status, Gate-Ergebnis |
| 5 | Conductor | Artefakte und Gate-Ergebnisse sammeln | Konsolidierte Gate-Zusammenfassung |
| 6 | Result Interpreter | Ergebnisdeutung, Potenzialanalyse und Empfehlungen erstellen | Ergebnisinterpretation |
| 7 | Conductor | Status, Risiken und naechste Schritte konsolidieren | Finaler Gesamtstatus |

## Schritt 1: Conductor Sammelt Kontext

Der Conductor klaert:

- CSV-Pfad.
- Target-Status: gegeben oder offen.
- Run-ID oder Bedarf fuer neuen Run.
- Split-Modus.
- Vorhandene Artefakte.
- Offene fachliche oder technische Fragen.

## Schritt 2: Opportunity Analyst Prueft Oder Empfiehlt

Der Opportunity Analyst arbeitet in einem von zwei Modi:

- Target gegeben: Plausibilitaet, Problemtyp und Risiken pruefen.
- Target offen: Zielkandidaten identifizieren und priorisieren.

Output ist eine Empfehlung an den Pipeline Builder, keine technische Pipeline-Aenderung.

## Schritt 3: Conductor Prueft Empfehlung

Der Conductor prueft, ob die Empfehlung:

- Ein klares Target nennt.
- Einen Problemtyp benennt.
- Annahmen und Risiken dokumentiert.
- Fuer die bestehende Pipeline verwertbar ist.

## Schritt 4: Pipeline Builder Nutzt Bestehende Pipeline

Der Pipeline Builder:

- Nutzt die vorhandene Step-Pipeline oder prueft einen vorhandenen Run.
- Validiert Artefakte.
- Beachtet Leakage-Audit.
- Prueft Modellartefakte, Holdout und Metriken.
- Dokumentiert Reproduzierbarkeit.

Er baut keine neue Pipeline.

## Schritt 5: Conductor Sammelt Gates

Der Conductor sammelt:

- Vollstaendigkeit der Artefakte.
- Ergebnis je Validierungsgate.
- Leakage-Status.
- `progress.json.status`.
- Fehler und offene Punkte.

## Schritt 6: Result Interpreter Deutet Ergebnisse

Der Result Interpreter:

- Liest Step 11 bis Step 16.
- Ordnet Modellguete verstaendlich ein.
- Erklaert Risiken und Caveats.
- Formuliert Potenzial und naechste Schritte.
- Ueberschreibt keine Reports ohne expliziten Auftrag.

## Schritt 7: Conductor Konsolidiert

Der Conductor liefert:

- Finalen Gesamtstatus.
- Zusammenfassung von Use Case, Run und Modell.
- Gate-Ergebnis.
- Risiken.
- Offene Fragen.
- Empfohlene naechste Schritte.

## Handoff-Vertrag

| Handoff | Mindestinhalt |
|---|---|
| Conductor -> Opportunity Analyst | CSV-Kontext, Target-Status, vorhandene Profiling-Artefakte |
| Opportunity Analyst -> Conductor | Use Case, Target, Problemtyp, Risiken, Empfehlung |
| Conductor -> Pipeline Builder | Run-Auftrag oder Run-Pfad, Target, Split-Modus, Gates |
| Pipeline Builder -> Conductor | Artefaktliste, Gate-Ergebnis, Fehlerdiagnose |
| Conductor -> Result Interpreter | Validierte Artefakte, Qualitaetsflag, Leakage-Status |
| Result Interpreter -> Conductor | Ergebnisdeutung, Potenzial, Grenzen, naechste Schritte |

## Ausblick Auf Phase 2

Phase 1 endet ohne Quality Judge. Phase 2 kann darauf aufbauen und einen Quality Judge zur Bewertung von Run und Report ergaenzen.

Der Quality Judge ersetzt weder die Pipeline noch den Result Interpreter. Er prueft hard gates, qualitative rubrics und Report Faithfulness als zusaetzliche Kontrollschicht.
