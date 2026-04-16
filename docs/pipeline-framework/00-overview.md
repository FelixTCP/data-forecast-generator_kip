# Single-Agent Pipeline Framework

Dieses Verzeichnis dokumentiert die **technische Spezifikation** der Data Forecast Generator Pipeline — als Referenz für Implementierung, Tests und LLM-gestützte Code-Generierung.

---

## Pipeline-Übersicht

```
Data Forecast Generator Pipeline
├─ [10] csv_read_cleansing.py    → Einlesen + Cleansing + bereinigtes DataFrame
├─ [11] data_exploration.py      → Datenexploration + Metadata + Zeitreihen-Analyse
├─ [12] feature_extraction.py    → Features + Modell-Empfehlung
├─ [13] model_training.py        → Trainiertes Modell + CV-Ergebnisse
├─ [14] model_evaluation.py      → Evaluationsmetriken + Residualanalyse
├─ [15] model_selection.py       → Modellvergleich + Auswahl
└─ [16] result_presentation.py   → Report + Artefakte
```

Alle Module liegen unter `src/data_forecast_generator/pipeline/`.  
Der Orchestrator `src/data_forecast_generator/pipeline/orchestrator.py` ruft sie sequenziell auf und gibt deren Output jeweils als Input an den nächsten Schritt weiter.

---

## Modul-Dokumentation

| Schritt | Modul | Spezifikation |
|---------|-------|---------------|
| 10 | `csv_read_cleansing.py` | [10-csv-read-cleansing.md](10-csv-read-cleansing.md) |
| 11 | `data_exploration.py` | [11-data-exploration.md](11-data-exploration.md) |
| 12 | `feature_extraction.py` | [12-feature-extraction.md](12-feature-extraction.md) |
| 13 | `model_training.py` | [13-model-training.md](13-model-training.md) |
| 14 | `model_evaluation.py` | [14-model-evaluation.md](14-model-evaluation.md) |
| 15 | `model_selection.py` | [15-model-selection.md](15-model-selection.md) |
| 16 | `result_presentation.py` | [16-result-presentation.md](16-result-presentation.md) |

---

## Allgemeine Konventionen

### Modul-Interface
Jedes Modul exportiert genau eine Einstiegsfunktion:

```python
def run_analysis(previous_output: dict, config: dict | None = None) -> dict:
    ...
```

### Output-Dictionary-Pflichtfelder
Jedes Modul-Output-Dictionary enthält:

```python
{
    "metadata": {
        "execution_id": str,          # uuid4()[:8]
        "module_name": str,
        "timestamp_created": str,     # ISO 8601 UTC
        "runtime_seconds": float,
    },
    "errors": list[str],
    "warnings": list[str],
    # ... modulspezifische Felder
}
```

### Logging
- `logging.info()` für wichtige Schritte
- `logging.debug()` für interne Details
- Kein `print()`

### Fehlerbehandlung
- `ValueError` bei ungültigem Input
- `RuntimeError` bei Verarbeitungsfehlern
- Alle Exceptions werden im Orchestrator gefangen und weitergemeldet
