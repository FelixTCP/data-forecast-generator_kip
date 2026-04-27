# Skill: Feedback Loop Entwerfen

## Zweck

Dieser Skill beschreibt den Feedback Loop fuer iterative Verbesserung nach einem Judge Report. Er spricht Empfehlungen aus und aendert nicht automatisch Code, Reports oder Artefakte.

Primaere Wahrheit bleibt `CURRENT_SYSTEM_DOCUMENTATION.md`.

## Schutzregeln

- Keine automatische Codeaenderung.
- Keine automatische Report-Ueberschreibung.
- Keine neue Pipeline und keine neue Runtime.
- Bestehende Runs bleiben Audit-Artefakte.
- Aenderungen erfolgen nur nach explizitem Auftrag.
- Hard gates haben Vorrang vor LLM-Judgement.

## Entscheidungen

### 1. `accept`

Run und Report sind ausreichend belastbar.

Zustaendig:

- Conductor konsolidiert den finalen Status.
- Result Interpreter kann die akzeptierte Interpretation bereitstellen.

Betroffene Artefakte:

- Keine Aenderungen erforderlich.

### 2. `revise_report`

Technische Artefakte sind okay, aber Interpretation oder Report sind unvollstaendig, ueberzogen oder nicht artefaktgetreu.

Zustaendig:

- Conductor dokumentiert die noetige Revision.
- Result Interpreter ueberarbeitet die Interpretation nur nach Auftrag und anhand konkreter Judge-Hinweise.
- Quality Judge kann nach der Revision erneut pruefen.

Betroffene Artefakte:

- Interpretierende Reports oder Chat-Zusammenfassungen, nur nach explizitem Auftrag.

### 3. `rerun_pipeline`

Ein neuer oder korrigierter Run ist sinnvoll bei:

- Hard gates nicht erfuellt.
- Leakage-Verdacht.
- Daten- oder Feature-Problemen.
- Unzureichender Modellqualitaet.
- Neu zu pruefendem Target.

Zustaendig:

- Conductor dokumentiert Grund und benoetigte Eingaben.
- Pipeline Builder prueft oder wiederholt den Run nur nach Auftrag.
- Opportunity Analyst prueft Target oder Use Case erneut nur nach Auftrag.

Betroffene Artefakte:

- Bestehende Runs bleiben Audit-Artefakte.
- Ein neuer Run erhaelt ein eigenes `output/<RUN_ID>/`.

### 4. `request_human_review`

Menschliche Pruefung ist sinnvoll bei:

- Widerspruechlichen Artefakten.
- Unklaren Business-Annahmen.
- Hohem Risiko.
- Unsicherheit des Judges.

Zustaendig:

- Conductor sammelt offene Fragen und Risiken.
- Fachliche oder technische Entscheider_innen pruefen die markierten Punkte.

Betroffene Artefakte:

- Keine automatische Aenderung.

## Output

Der Feedback Loop liefert:

- Entscheidung.
- Begruendung.
- Zustaendige Rolle.
- Betroffene Artefakte.
- Erforderliche naechste Schritte.
- Ob explizite Beauftragung oder Human Review noetig ist.

