# Skill: Forecast-Opportunity Bewerten

## Zweck

Dieser Skill beschreibt, wie eine sinnvolle Forecast- oder Regression-Opportunity anhand vorhandener Artefakte bewertet wird. Er erzeugt keine Pipeline und fuehrt kein Training aus.

Primaere Wahrheit fuer den technischen Kontext ist `CURRENT_SYSTEM_DOCUMENTATION.md`.

## Schutzregeln

- Diese Skill-Datei beschreibt Verhalten fuer Phase 1.
- Phase 1 erzeugt keine neue Runtime und keine Python-Agenten.
- Die bestehende Step-Pipeline wird nicht ersetzt.
- Bestehende Run-Verzeichnisse sind Audit-Artefakte.
- Aenderungen an Pipeline-Code oder Run-Artefakten erfolgen nur nach explizitem Auftrag.
- Keine Step-Skripte veraendern.
- Keine neue Pipeline vorschlagen, wenn die bestehende Pipeline ausreicht.
- Keine Modell- oder Feature-Implementierung vornehmen.
- Nicht von historischen Step-10- oder `src/`-Annahmen ausgehen.

## Target Gegeben

Wenn ein Target vorgegeben ist:

- Pruefen, ob die Zielspalte im bereinigten Datensatz existiert.
- Pruefen, ob die Zielspalte numerisch oder sinnvoll numerisch ableitbar ist.
- Zeitbezug pruefen, wenn eine Zeitspalte erkannt wurde.
- Einschaetzen, ob Regression oder Forecasting passender ist.
- Risiken, Grenzen und moegliche Alternativen dokumentieren.

## Target Offen

Wenn kein Target vorgegeben ist:

- Numerische Zielkandidaten aus Profiling, Cleansing und Exploration identifizieren.
- Kandidaten nach Praxisnutzen, Datenqualitaet und Prognostizierbarkeit priorisieren.
- Offensichtliche IDs, konstante Spalten, Leckage-Spalten oder technische Hilfsspalten ausschliessen.
- Eine Zielvariable begruendet empfehlen.

## Zielvariablen-Kriterien

| Kriterium | Frage |
|---|---|
| Fachliche Bedeutung | Ist das Target praktisch relevant? |
| Messbarkeit | Ist das Target sauber beobachtet? |
| Varianz | Hat das Target genug Streuung? |
| Datenqualitaet | Gibt es Nulls, Ausreisser oder Casting-Probleme? |
| Zeitbezug | Gibt es eine sinnvolle zeitliche Ordnung? |
| Steuerbarkeit | Koennen Empfehlungen daraus abgeleitet werden? |
| Leakage-Risiko | Gibt es direkte Kopien oder Ziel-Proxies? |

## Forecasting Vs. Regression

| Problemtyp | Geeignet Wenn |
|---|---|
| Forecasting | Zeitspalte vorhanden, zeitliche Reihenfolge relevant, Zukunftswerte sollen prognostiziert werden |
| Regression | Zielwert aus aktuellen oder nicht strikt zeitlichen Features erklaert oder vorhergesagt werden soll |

## Business- Oder Praxisnutzen

Bewerte:

- Potenzielle Einsparungen.
- Bessere Planung.
- Fruehwarnung oder Risikoerkennung.
- Operative Steuerbarkeit.
- Verstaendlichkeit fuer Anwender_innen.
- Aufwand fuer Datenbereitstellung.

## Datenqualitaet

Beruecksichtige:

- Zeilenanzahl.
- Null-Raten.
- Datentypen.
- Duplikate.
- Zeitspaltenqualitaet.
- High-cardinality oder low-variance Features.

## Leakage-Risiko

Warnsignale:

- Feature ist direkte Kopie oder Transformation des Targets.
- Feature enthaelt Zielwert zum selben Zeitpunkt.
- Unplausibel hohe Korrelation oder Mutual Information.
- Feature entsteht erst nach dem vorherzusagenden Ereignis.

## Ausschlussgruende

Ein Use Case oder Target kann ausgeschlossen werden bei:

- Nicht numerischem oder ungeeignetem Target.
- Zu wenig Zeilen.
- Zu wenig Varianz.
- Starkem Leakage-Verdacht.
- Fehlendem praktischen Nutzen.
- Unklarer fachlicher Interpretation.

## Output-Schema

```markdown
## Opportunity Report

- Empfohlener Use Case:
- Empfohlene oder gepruefte Zielvariable:
- Problemtyp: Regression | Forecasting
- Begruendung:
- Annahmen:
- Ausgeschlossene Kandidaten:
- Risiken:
- Empfehlung an Pipeline Builder:
- Offene Fragen:
```
