# Skill: Business-Report Schreiben

## Zweck

Dieser Skill beschreibt, wie aus vorhandenen Pipeline-Artefakten eine verstaendliche Ergebnisinterpretation entsteht. Er ueberschreibt keine Report-Artefakte, ausser dies wurde explizit beauftragt.

Primaere Wahrheit ist `CURRENT_SYSTEM_DOCUMENTATION.md`.

## Schutzregeln

- Diese Skill-Datei beschreibt Verhalten fuer Phase 1.
- Phase 1 erzeugt keine neue Runtime und keine Python-Agenten.
- Die bestehende Step-Pipeline wird nicht ersetzt.
- Bestehende Run-Verzeichnisse sind Audit-Artefakte.
- Aenderungen an Pipeline-Code oder Run-Artefakten erfolgen nur nach explizitem Auftrag.
- Keine Pipeline ersetzen.
- Keine Step-Skripte aendern.
- Keine bestehenden Reports ungefragt ueberschreiben.
- Keine ueberzogenen Versprechen machen.
- Nicht von historischen Step-10- oder `src/`-Annahmen ausgehen.

## Schreibprinzipien

- Nicht rein technisch schreiben.
- Use Case klar erklaeren.
- Modellguete verstaendlich einordnen.
- Nutzen vorsichtig und konkret formulieren.
- Grenzen transparent machen.
- Naechste Schritte empfehlen.
- Zwischen gesicherten Befunden, Einschaetzungen und offenen Fragen unterscheiden.

## Eingaben

| Artefakt | Verwendung |
|---|---|
| `step-11-exploration.json` | Daten- und Feature-Kontext |
| `step-12-features.json` | Welche Features verwendet wurden |
| `step-13-training.json` | Welche Modelle trainiert wurden |
| `step-14-evaluation.json` | Modellguete und Qualitaetsflag |
| `step-15-selection.json` | Auswahlentscheidung |
| `step-16-report.md` | Bestehender finaler Pipeline-Report |
| `leakage_audit.json` | Gueltigkeit und Risiko der Bewertung |

## Empfohlene Struktur

```markdown
## Ergebnisinterpretation

### Use Case
Was wurde vorhergesagt und warum ist das praktisch relevant?

### Modellguete
Wie gut ist das Ergebnis, in Alltagssprache und mit wenigen Kennzahlen?

### Nutzen
Welche Entscheidungen koennte das Ergebnis unterstuetzen?

### Grenzen
Welche Daten-, Modell- oder Leakage-Risiken bestehen?

### Naechste Schritte
Welche fachlichen und technischen Pruefungen sind sinnvoll?
```

## Umgang Mit Unsicherheit

- Keine Produktionsreife behaupten, wenn `quality_flag` nicht `acceptable` ist.
- Leakage-Warnungen prominent nennen.
- Negative R2-Werte klar als schlechter als Mean-Baseline erklaeren.
- RMSE und MAE immer im Kontext der Zielwert-Skala interpretieren.
- Business-Potenzial als Hypothese formulieren, wenn keine wirtschaftlichen Zahlen vorliegen.

## Output

Der Output ist eine ergaenzende Interpretation im Chat oder in einer explizit beauftragten Markdown-Datei. Ohne expliziten Auftrag werden `step-16-report.md` und andere Artefakte nicht veraendert.
