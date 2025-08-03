# LaTeX Report Integration Guide

## Schnelle Verwendung

### 1. Debug-Test ausführen (empfohlen)
```bash
python debug_reports.py
```
- Generiert alle Reports mit synthetischen Daten
- Keine Ressourcenverschwendung
- Testet alle LaTeX-Funktionen

### 2. Vollständiges Experiment (nur wenn nötig)
```bash
python run_experiment.py
```
- Dauert ~30-45 Minuten
- Hohe CPU-Last auf MacBook M4
- Reale Experiment-Daten

## Generierte Files

### LaTeX-Substitutionen
**Datei:** `results/test_[ID]/latex_substitutions_[ID].tex`

**Verwendung in chapter4.tex:**
```latex
% In der Präambel einfügen:
\input{results/test_[ID]/latex_substitutions_[ID].tex}

% Dann verwenden:
Die Gesamtaccuracy beträgt \OVERALL_ACCURACY\%.
Das Experiment umfasste \ANZAHL_TOTAL_TICKETS{} Tickets.
```

### Verfügbare LaTeX-Variablen

#### Grunddaten
- `\ANZAHL_TOTAL_TICKETS` - Gesamtanzahl Tickets
- `\ANZAHL_HARDWARE`, `\ANZAHL_SOFTWARE`, etc. - Pro Kategorie
- `\TICKETS_PRO_KATEGORIE` - Tickets pro Kategorie
- `\N_PRO_BEDINGUNG` - Stichprobengröße pro Bedingung

#### Performance-Metriken
- `\OVERALL_ACCURACY` - Gesamtaccuracy (%)
- `\OVERALL_F1` - F1-Score (%)
- `\OVERALL_MCC` - Matthews Correlation Coefficient

#### Zero-Shot Performance
- `\LLAMA_STRUCT_0SHOT` - Llama structured zero-shot (%)
- `\LLAMA_UNSTRUCT_0SHOT` - Llama unstructured zero-shot (%)
- `\MISTRAL_STRUCT_0SHOT` - Mistral structured zero-shot (%)
- `\MISTRAL_UNSTRUCT_0SHOT` - Mistral unstructured zero-shot (%)

#### Few-Shot Progression
- `\L_S_0`, `\L_S_1`, `\L_S_3`, `\L_S_5` - Llama structured 0-5 shot
- `\L_U_0`, `\L_U_1`, `\L_U_3`, `\L_U_5` - Llama unstructured 0-5 shot
- `\M_S_0`, `\M_S_1`, `\M_S_3`, `\M_S_5` - Mistral structured 0-5 shot
- `\M_U_0`, `\M_U_1`, `\M_U_3`, `\M_U_5` - Mistral unstructured 0-5 shot

#### Delta-Werte (Verbesserung 0-Shot → 5-Shot)
- `\L_S_DELTA`, `\L_U_DELTA`, `\M_S_DELTA`, `\M_U_DELTA`

#### Statistische Ergebnisse
- `\F_STAT_FEWSHOTCOUNT`, `\F_STAT_MODEL`, `\F_STAT_PROMPTTYPE` - F-Statistiken
- `\P_VALUE_FEWSHOTCOUNT`, `\P_VALUE_MODEL`, `\P_VALUE_PROMPTTYPE` - p-Werte
- `\EFFECT_SIZE_FEWSHOTCOUNT`, etc. - η² Effektgrößen
- `\COHENS_F_FEW_SHOT_COUNT`, etc. - Cohen's f Effektgrößen

### TikZ-Koordinaten
**Datei:** `results/test_[ID]/tikz_data_[ID].tex`

**Verwendung für Plots:**
```latex
\begin{tikzpicture}
\input{results/test_[ID]/tikz_data_[ID].tex}

% Plot Few-Shot Progression
\draw plot coordinates {
    (llama318bstructured0)
    (llama318bstructured1)
    (llama318bstructured3)
    (llama318bstructured5)
};
\end{tikzpicture}
```

### Beispiel chapter4.tex Integration

```latex
\section{Experimentelle Ergebnisse}

Das Experiment umfasste insgesamt \ANZAHL_TOTAL_TICKETS{} IT-Support-Tickets, 
gleichmäßig verteilt auf \TICKETS_PRO_KATEGORIE{} Tickets pro Kategorie 
(Hardware: \ANZAHL_HARDWARE{}, Software: \ANZAHL_SOFTWARE{}, 
Network: \ANZAHL_NETWORK{}, Security: \ANZAHL_SECURITY{}).

\subsection{Overall Performance}
Die Gesamtaccuracy über alle experimentellen Bedingungen betrug 
\OVERALL_ACCURACY\% mit einem gewichteten F1-Score von \OVERALL_F1\%.

\subsection{Zero-Shot Baseline}
\begin{table}[h!]
\centering
\caption{Zero-Shot Performance der LLM-Modelle}
\begin{tabular}{lcc}
\toprule
\textbf{Modell} & \textbf{Structured} & \textbf{Unstructured} \\
\midrule
Llama 3.1 8B & \LLAMA_STRUCT_0SHOT\% & \LLAMA_UNSTRUCT_0SHOT\% \\
Mistral 7B & \MISTRAL_STRUCT_0SHOT\% & \MISTRAL_UNSTRUCT_0SHOT\% \\
\bottomrule
\end{tabular}
\end{table}

\subsection{Few-Shot Learning Effekt}
Der stärkste Few-Shot-Effekt zeigte sich bei Mistral mit strukturierten Prompts:
Verbesserung von \M_S_0\% (0-shot) auf \M_S_5\% (5-shot), 
entsprechend einem Delta von \M_S_DELTA{} Prozentpunkten.

\subsection{Statistische Signifikanz}
ANOVA-Analyse ergab statistisch signifikante Haupteffekte:
\begin{itemize}
\item Few-Shot Count: $F = \F_STAT_FEWSHOTCOUNT$, $p = \P_VALUE_FEWSHOTCOUNT$ 
\item Modell: $F = \F_STAT_MODEL$, $p = \P_VALUE_MODEL$
\item Prompt-Typ: $F = \F_STAT_PROMPTTYPE$, $p = \P_VALUE_PROMPTTYPE$
\end{itemize}
```

## Empfohlenes Workflow

1. **Entwicklung:** `python debug_reports.py` für schnelle Tests
2. **Finale Thesis:** `python run_experiment.py` für reale Daten (einmal)
3. **LaTeX-Integration:** Kopiere gewünschte Variables aus `.tex` Files
4. **Kompilierung:** LaTeX sollte alle Variablen automatisch ersetzen

## Troubleshooting

### "Undefined control sequence"
- Stelle sicher, dass `\input{latex_substitutions_[ID].tex}` in der Präambel steht
- Prüfe, ob der Pfad zur .tex-Datei korrekt ist

### "Variable not found"
- Prüfe Schreibweise (case-sensitive)
- Alle verfügbaren Variablen sind in der generierten .tex-Datei dokumentiert

### Performance-Probleme
- Nutze `debug_reports.py` für Entwicklung
- Vollständiges Experiment nur einmal am Ende ausführen
