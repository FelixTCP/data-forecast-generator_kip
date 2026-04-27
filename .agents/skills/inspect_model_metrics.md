# Skill: Modellmetriken Interpretieren

## Zweck

Dieser Skill beschreibt, wie Modellmetriken aus vorhandenen Pipeline-Artefakten interpretiert werden. Er fuehrt kein Training aus und ersetzt keine Evaluation.

Primaere Wahrheit ist `CURRENT_SYSTEM_DOCUMENTATION.md`.

## Schutzregeln

- Diese Skill-Datei beschreibt Verhalten fuer Phase 1.
- Phase 1 erzeugt keine neue Runtime und keine Python-Agenten.
- Die bestehende Step-Pipeline wird nicht ersetzt.
- Bestehende Run-Verzeichnisse sind Audit-Artefakte.
- Aenderungen an Pipeline-Code oder Run-Artefakten erfolgen nur nach explizitem Auftrag.
- Keine Step-Skripte aendern.
- Keine Modellartefakte ueberschreiben.
- Keine Pipeline-Reports ungefragt ersetzen.
- Nicht von historischen Step-10- oder `src/`-Annahmen ausgehen.

## Wichtige Artefakte

- `output/<RUN_ID>/step-13-training.json`
- `output/<RUN_ID>/step-14-evaluation.json`
- `output/<RUN_ID>/step-15-selection.json`
- `output/<RUN_ID>/leakage_audit.json`
- `output/<RUN_ID>/holdout.npz`

## Metriken

| Metrik | Interpretation |
|---|---|
| `r2` | Anteil erklaerter Varianz; `0` entspricht Mean-Baseline, negativ ist schlechter als Mean-Baseline |
| `rmse` | Root Mean Squared Error; bestraft grosse Fehler stark |
| `mae` | Mean Absolute Error; robuster typischer absoluter Fehler |
| `residual_mean` | Mittlerer Bias der Vorhersagen |
| `residual_max_abs` | Groesster absoluter Fehler im Holdout |
| `cv_mean_r2` | Durchschnittliche Cross-Validation-Guete |
| `cv_std_r2` | Stabilitaet der CV-Ergebnisse |
| naive Baseline | Vergleich gegen einfache Vorhersage, oft `y_hat_t = y_(t-1)` |
| `mape` | Prozentualer Fehler, nur sinnvoll wenn Zielwerte nicht null oder nahe null sind |

## Qualitaetsflags

| Flag | Bedeutung |
|---|---|
| `acceptable` | Ergebnis ist fuer weitere fachliche Pruefung brauchbar |
| `marginal` | Nutzbar mit Vorsicht, Follow-up empfohlen |
| `subpar` | Unzureichend; Expansion oder Feature-Review noetig |
| `subpar_after_expansion` | Auch nach Erweiterung schwach; Ergebnis klar begrenzen |
| `no_viable_candidate` | Kein Kandidat ist belastbar auswaehlbar |

## No Viable Candidate

`no_viable_candidate` bedeutet, dass kein Kandidat belastbar auswaehlbar ist. In diesem Fall darf keine positive Modellnarrative erzeugt werden. Empfehlungen muessen auf Datenverbesserung, Feature-Ueberarbeitung, Zielvariablenpruefung oder erneute Modellierung zielen.

## Negativer R2

Ein negativer R2 bedeutet, dass das Modell schlechter ist als eine Mean-Baseline. Solche Kandidaten sollen:

- Nicht als Gewinner ausgewaehlt werden.
- In Reports klar als nicht geeignet markiert werden.
- Als Hinweis auf schwache Features, falschen Split, Overfitting oder ungeeigneten Use Case betrachtet werden.

## Leakage-Probe

Immer pruefen:

- `leakage_audit.status`.
- `step-14-evaluation.json.leakage_probe`.
- Ob sehr hohe R2-Werte plausibel sind.
- Ob target-derived Features die Bewertung dominieren.

Bei Leakage-Verdacht duerfen Metriken nicht als produktionsreif interpretiert werden.

## Stabilitaet Und Praktische Belastbarkeit

Beruecksichtige:

- Abstand zwischen CV- und Holdout-Leistung.
- Hoehe von `cv_std_r2`.
- Vergleich mit naive Baseline.
- Fehlergroesse relativ zu `target_stats.mean`, `std`, `min`, `max`.
- Ob Fehlerspitzen fuer den Use Case kritisch sind.

## Output

Eine Metrikinterpretation soll enthalten:

- Bestes Modell und Ranking.
- Qualitaetsflag.
- Einordnung von R2, RMSE und MAE.
- Baseline-Vergleich.
- Stabilitaetseinschaetzung.
- Leakage-Einschaetzung.
- Praktische Belastbarkeit.
- Naechste technische oder fachliche Pruefschritte.
