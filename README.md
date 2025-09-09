# Empirische Evaluation von In-Context Learning Strategien für IT-Support-Ticket-Klassifikation

**DHBW Projektarbeit - 4. Semester Wirtschaftsinformatik**

## Forschungsgegenstand

Diese experimentelle Studie evaluiert systematisch die Effektivität von In-Context Learning Strategien bei der automatisierten Klassifikation von IT-Support-Tickets unter Verwendung lokaler Large Language Models. Das Forschungsdesign folgt dem CRISP-DM Vorgehensmodell und implementiert ein faktorielles 2×4×2 Experimentaldesign zur quantitativen Analyse verschiedener Few-Shot Learning Konfigurationen.

### Wissenschaftliche Fragestellungen

Die Untersuchung adressiert drei zentrale Forschungsfragen:

1. **Quantifizierung des Few-Shot Learning Effekts**: Systematische Evaluation der Klassifikationsgenauigkeit in Abhängigkeit der Anzahl bereitgestellter Beispiele (0-shot bis 5-shot)
2. **Modell-spezifische Performance-Analyse**: Vergleichende Bewertung der lokalen LLMs Llama 3.1 8B und Mistral 7B hinsichtlich ihrer Few-Shot Lernfähigkeiten
3. **Prompt Engineering Optimierung**: Empirische Untersuchung strukturierter versus unstrukturierter Prompt-Formate auf die Klassifikationsleistung

### Kategorien

- **Hardware**: Geräteprobleme (Laptop, Drucker, Monitor)
- **Software**: Programmprobleme (Excel, VPN, Email)
- **Network**: Netzwerkprobleme (Internet, WLAN, Server)
- **Security**: Sicherheitsprobleme (Phishing, Passwort, Malware)

## Installation und Setup

### Voraussetzungen
```bash
# Python 3.8+
# Ollama Server (für LLM-Inferenz)
```

### Installation
```bash
# Repository klonen
git clone <repository-url>
cd few-shot-classification-experiment

# Virtual Environment erstellen
python -m venv venv
source venv/bin/activate  # macOS/Linux
# oder
venv\Scripts\activate     # Windows

# Dependencies installieren
pip install -r requirements.txt
```

### Ollama Setup
```bash
# Ollama installieren (https://ollama.ai/)
# Modelle herunterladen
ollama pull llama3.1:8b
ollama pull mistral:7b

# Ollama Server starten
ollama serve
```

## Experiment ausführen

### Schnellstart
```bash
python src/few_shot_experiment.py
```

### Wissenschaftlicher Workflow
Das Experiment durchläuft folgende Phasen:

1. **Experimentelle Validierung**
   - Überprüfung der Konfiguration
   - Validierung des Experimental Design
   - Power-Analyse

2. **Power-Analyse**
   - Berechnung erforderlicher Stichprobengrößen
   - Schätzung der Experimentdauer
   - Validierung der statistischen Power

3. **Experimentelle Parameter**
   - Anzeige aller Faktoren und Level
   - Bestätigung durch Benutzer
   - Final setup validation

4. **Wissenschaftliche Analyse**
   - Deskriptive Statistiken mit Konfidenzintervallen
   - Mehrfaktorielle ANOVA
   - Post-hoc Tests (Tukey HSD)
   - Effektgrößenberechnung (Cohen's f)
   - Assumption Testing

5. **Visualisierungen**
   - Few-Shot Progression Plots
   - Model Comparison Charts
   - Statistical Heatmaps
   - Effect Size Forest Plots
   - Confusion Matrices

6. **Wissenschaftlicher Bericht**
   - Markdown Report mit allen Ergebnissen
   - Reproduzierbarkeits-Informationen

## Projektstruktur

```
few-shot-classification-experiment/
├── src/
│   └── few_shot_experiment.py    # Hauptexperimentcode
├── data/
│   ├── tickets_extended_187.xlsx # Hauptdatensatz (186 Tickets)
│   └── tickets_examples.xlsx     # Few-Shot Beispiele
├── results/                      # Experimentelle Ergebnisse
│   └── test_YYYYMMDD_HHMMSS/    # Eindeutige Test-IDs
│       ├── results_*.csv         # Rohdaten
│       ├── comprehensive_analysis_*.json # Statistische Analyse
│       ├── report_*.md           # Wissenschaftlicher Bericht
│       ├── *.png                 # Visualisierungen
│       └── experiment_*.log      # Vollständiges Log
├── config.yaml                  # Experimentelle Konfiguration
├── requirements.txt             # Python Dependencies
└── README.md                   # Diese Datei
```

## Konfiguration

Die Datei `config.yaml` enthält alle experimentellen Parameter:

```yaml
experiment:
  random_seed: 42                    # Reproduzierbarkeit
  total_experimental_tickets: 200    # Gesamtanzahl Tickets
  min_per_condition: 25             # Minimum pro Bedingung

quality_control:
  min_ticket_length: 20             # Mindestlänge pro Ticket
  max_retry_attempts: 3             # LLM Retry-Versuche
  timeout_seconds: 45               # Timeout pro Request

statistics:
  alpha_level: 0.05                 # Signifikanzniveau
  power_target: 0.80               # Statistische Power
  bonferroni_correction: true      # Multiple Comparisons
```

## Beispiel-Ergebnisse

### Statistische Ausgabe
```
EXPERIMENT ZUSAMMENFASSUNG
Overall Accuracy: 0.847
F1-Score (Weighted): 0.845
Matthews Correlation: 0.798
Balanced Accuracy: 0.849

ANOVA Ergebnisse:
   model: p = 0.0123 *
   few_shot_count: p = 0.0001 ***
   prompt_type: p = 0.2341 ns

Effektgrößen (Cohen's f):
   model: 0.142 (small)
   few_shot_count: 0.287 (medium)
   prompt_type: 0.089 (negligible)
```

### Generierte Dateien
- **Rohdaten**: `results_TIMESTAMP.csv`
- **Statistische Analyse**: `comprehensive_analysis_TIMESTAMP.json`
- **Visualisierungen**: Multiple PNG-Dateien
- **Wissenschaftlicher Bericht**: `report_TIMESTAMP.md`
- **Experiment-Log**: `experiment_TIMESTAMP.log`

## Wissenschaftliche Validität

### Statistische Rigorosität
- **Faktorielles Design**: Vollständige 2×4×2 Faktorenstruktur
- **Power-Analyse**: Minimum n=25 pro Bedingung für 80% Power
- **Multiple Comparisons**: Bonferroni-Korrektur
- **Effect Sizes**: Cohen's f für praktische Signifikanz
- **Assumption Testing**: Normalität und Homoskedastizität

### Reproduzierbarkeit
- **Seeding**: Alle Zufallsprozesse deterministisch
- **Logging**: Vollständige Dokumentation aller Schritte
- **Versionierung**: Alle Parameter in config.yaml
- **Metadata**: Experimentelle Bedingungen gespeichert

### Qualitätskontrolle
- **Data Validation**: Automatische Datenqualitätsprüfung
- **Error Handling**: Robuste Fehlerbehandlung
- **Progress Monitoring**: Real-time Fortschrittsverfolgung
- **Intermediate Saves**: Datensicherheit durch Zwischenspeicherung

## Erweiterte Nutzung

### Custom Analysis
```python
from src.few_shot_experiment import FewShotExperiment

# Eigenes Experiment
experiment = FewShotExperiment('custom_config.yaml')
results = experiment.run_experiment()
analysis = experiment.analyze_results(results)
```

### Parameter Tuning
```yaml
# config.yaml anpassen für verschiedene Experimente
models:
  - "gpt-4"          # Wenn verfügbar
  - "claude-3"       # Wenn verfügbar
  
few_shot_counts: [0, 2, 4, 6, 8]  # Andere Progression
```

## Wissenschaftliche Integrität

Diese Arbeit entspricht den Standards guter wissenschaftlicher Praxis der DHBW. Alle verwendeten Quellen sind ordnungsgemäß zitiert. Die experimentellen Daten und Analysemethoden sind vollständig dokumentiert und reproduzierbar.

### Limitationen

Die Studie unterliegt folgenden methodischen Einschränkungen:

- **Synthetische Daten**: Verwendung algorithmisch generierter anstelle authentischer Support-Tickets
- **Beschränkte Modellauswahl**: Fokus auf lokale 7B-8B Parameter Modelle
- **Domänen-Spezifität**: Beschränkung auf IT-Support-Kontext
- **Temporäre Validität**: Snapshot-Evaluation ohne longitudinale Komponente
