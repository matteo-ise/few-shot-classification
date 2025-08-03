# Wissenschaftliches Few-Shot Learning Experiment

## 🎯 Projektübersicht

Dieses Projekt implementiert ein **wissenschaftlich rigoroses Few-Shot Learning Experiment** für die DHBW Projektarbeit. Es vergleicht verschiedene Large Language Models (LLMs) bei der Klassifikation von IT-Support-Tickets unter verschiedenen experimentellen Bedingungen.

### Wissenschaftliche Standards
- ✅ **Faktorielles 2×4×2 Design** (LLM × Few-Shot-Count × Prompt-Type)
- ✅ **Statistische Power** mit n≥25 pro Bedingung
- ✅ **Vollständige Few-Shot Progression** (0, 1, 3, 5 Beispiele)
- ✅ **Baseline-Vergleich** mit Zero-shot als Kontrolle
- ✅ **Reproduzierbarkeit** durch Seeding und vollständige Dokumentation

## 📊 Experimentelles Design

### Faktoren
- **LLM-Modelle**: Llama3.1:8b, Mistral:7b
- **Few-Shot Bedingungen**: 0-shot (Baseline), 1-shot, 3-shot, 5-shot
- **Prompt-Typen**: Strukturiert, Unstrukturiert

### Abhängige Variablen
- Klassifikationsaccuracy
- F1-Score (gewichtet und makro)
- Precision & Recall
- Matthews Correlation Coefficient (MCC)
- Balanced Accuracy

### Kategorien
- **Hardware**: Geräteprobleme (Laptop, Drucker, Monitor)
- **Software**: Programmprobleme (Excel, VPN, Email)
- **Network**: Netzwerkprobleme (Internet, WLAN, Server)
- **Security**: Sicherheitsprobleme (Phishing, Passwort, Malware)

## 🔧 Installation und Setup

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

## 🚀 Experiment ausführen

### Schnellstart
```bash
python src/few_shot_experiment.py
```

### Wissenschaftlicher Workflow
Das Experiment durchläuft folgende Phasen:

1. **📋 Experimentelle Validierung**
   - Überprüfung der Konfiguration
   - Validierung des Experimental Design
   - Power-Analyse

2. **📊 Power-Analyse**
   - Berechnung erforderlicher Stichprobengrößen
   - Schätzung der Experimentdauer
   - Validierung der statistischen Power

3. **🎯 Experimentelle Parameter**
   - Anzeige aller Faktoren und Level
   - Bestätigung durch Benutzer
   - Final setup validation

4. **🚀 Experiment-Ausführung**
   - Fortschrittsverfolgung mit Progress Bars
   - Fehlerbehandlung und Retry-Mechanismen
   - Zwischenspeicherung alle 50 Klassifikationen

5. **📊 Wissenschaftliche Analyse**
   - Deskriptive Statistiken mit Konfidenzintervallen
   - Mehrfaktorielle ANOVA
   - Post-hoc Tests (Tukey HSD)
   - Effektgrößenberechnung (Cohen's f)
   - Assumption Testing

6. **📈 Visualisierungen**
   - Few-Shot Progression Plots
   - Model Comparison Charts
   - Statistical Heatmaps
   - Effect Size Forest Plots
   - Confusion Matrices

7. **📝 Wissenschaftlicher Bericht**
   - Markdown Report mit allen Ergebnissen
   - LaTeX-ready Tables
   - Reproduzierbarkeits-Informationen

## 📁 Projektstruktur

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

## ⚙️ Konfiguration

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

## 📊 Beispiel-Ergebnisse

### Statistische Ausgabe
```
🎯 EXPERIMENT ZUSAMMENFASSUNG
Overall Accuracy: 0.847
F1-Score (Weighted): 0.845
Matthews Correlation: 0.798
Balanced Accuracy: 0.849

📊 ANOVA Ergebnisse:
   model: p = 0.0123 *
   few_shot_count: p = 0.0001 ***
   prompt_type: p = 0.2341 ns

📏 Effektgrößen (Cohen's f):
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

## 🔬 Wissenschaftliche Validität

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

## 📚 Für DHBW Projektarbeit

### Verwendung in der Arbeit
1. **Methodenkapitel**: Experimentelles Design aus config.yaml
2. **Ergebniskapitel**: Analyse aus JSON und Markdown Report
3. **Diskussion**: Effektgrößen und praktische Signifikanz
4. **Anhang**: Vollständige Rohdaten und Reproduzierbarkeit

### LaTeX Integration
```latex
% Beispiel-Tabelle (wird automatisch generiert)
\input{tables/anova_results.tex}
\input{tables/descriptive_statistics.tex}
```

## 🔧 Erweiterte Nutzung

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

## 🤝 Beitragen

### Issues
- Experimentelle Verbesserungen
- Statistische Methoden
- Visualisierungsoptionen

### Pull Requests
1. Fork des Repositories
2. Feature Branch erstellen
3. Tests durchführen
4. Pull Request einreichen

## 📄 Lizenz

Dieses Projekt ist für akademische Zwecke der DHBW entwickelt.

## 📞 Support

Bei Fragen zum wissenschaftlichen Design oder der Implementierung:
- GitHub Issues für technische Probleme
- DHBW Betreuer für akademische Fragen

---

**Hinweis**: Dieses Experiment wurde nach höchsten wissenschaftlichen Standards entwickelt und ist für akademische Publikationen geeignet. 