"""
Few-Shot Learning Experiment-System für IT-Support-Tickets

Dieses Programm vergleicht verschiedene KI-Modelle (z.B. Llama3.1, Mistral) bei der Klassifikation von IT-Support-Tickets.
Es unterstützt verschiedene Few-Shot-Strategien, Prompt-Typen und ist für wissenschaftliche Auswertung optimiert.

Autor: DHBW Student (4. Semester)
"""

import pandas as pd
import requests
import yaml
import json
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, recall_score, matthews_corrcoef, balanced_accuracy_score
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
warnings.filterwarnings('ignore')

class FewShotExperiment:
    """
    Hauptklasse für das Few-Shot Learning Experiment.
    Führt die Klassifikation, Auswertung und Visualisierung durch.
    """
    def __init__(self, config_path: str = "config.yaml"):
        # Konfiguration laden (z.B. Modelle, Kategorien, Datenpfade)
        self.config = self._load_config(config_path)
        # Test-ID für eindeutige Ergebnis-Ordner
        self.test_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Zufallszahlengenerator für Reproduzierbarkeit
        np.random.seed(self.config.get('experiment', {}).get('random_seed', 42))

    def _load_config(self, config_path: str) -> Dict:
        """Lädt die Konfiguration aus der YAML-Datei."""
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f)
        except Exception as e:
            print(f"❌ Fehler beim Laden der Konfiguration: {e}")
            return {}

    def _load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Lädt die Ticketdaten und Few-Shot-Beispiele aus Excel-Dateien.
        Die Pfade werden aus der config.yaml gelesen.
        """
        print("📂 Lade Daten...")
        data_dir = self.config.get('paths', {}).get('data_dir', 'data/')
        ticket_file = self.config.get('paths', {}).get('ticket_file', 'tickets_simple_10.xlsx')
        fewshot_file = self.config.get('paths', {}).get('fewshot_file', 'few_shot_examples.xlsx')
        # Tickets laden
        tickets_path = os.path.join(data_dir, ticket_file)
        tickets_df = pd.read_excel(tickets_path)
        print(f"✅ {len(tickets_df)} Tickets geladen aus {tickets_path}")
        # Few-Shot-Beispiele laden
        fewshot_path = os.path.join(data_dir, fewshot_file)
        few_shot_df = pd.read_excel(fewshot_path)
        print(f"✅ {len(few_shot_df)} Few-Shot-Beispiele geladen aus {fewshot_path}")
        return tickets_df, few_shot_df

    def _create_prompt(self, ticket_text: str, few_shot_examples: List[Dict], prompt_type: str, categories: List[str]) -> str:
        """
        Erstellt einen Prompt für das LLM.
        - ticket_text: Der Text des zu klassifizierenden Tickets
        - few_shot_examples: Liste von Beispielen (für Few-Shot-Learning)
        - prompt_type: "structured" oder "unstructured"
        - categories: Liste der möglichen Kategorien
        """
        if prompt_type == "structured":
            # Wissenschaftlich: Klare Instruktion, Definitionen, Format
            prompt = "Klassifiziere das folgende IT-Support-Ticket in eine der Kategorien.\n\n"
            for cat in categories:
                prompt += f"- {cat}\n"
            prompt += "\n"
            for example in few_shot_examples:
                prompt += f"Beispiel:\nTicket: {example['ticket_text']}\nKategorie: {example['kategorie']}\n\n"
            prompt += f"Ticket: {ticket_text}\nKategorie:"
        else:
            # Unstrukturierter Prompt, aber mit klaren Kategorien
            prompt = f"Was für ein IT-Problem ist das? Mögliche Kategorien: {', '.join(categories)}\n\n"
            for example in few_shot_examples:
                prompt += f"Problem: {example['ticket_text']}\nAntwort: {example['kategorie']}\n\n"
            prompt += f"Problem: {ticket_text}\nAntwort:"
        return prompt

    def _query_llm(self, prompt: str, model: str) -> Optional[str]:
        """
        Sendet den Prompt an das gewünschte Modell über Ollama und gibt die Antwort zurück.
        Fehler werden abgefangen und gemeldet.
        """
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": self.config.get("llm_params", {}).get("temperature", 0.1),
                "top_p": self.config.get("llm_params", {}).get("top_p", 0.9),
                "num_predict": self.config.get("llm_params", {}).get("max_tokens", 50)
            }
        }
        try:
            response = requests.post(
                f"{self.config['ollama']['base_url']}/api/generate",
                json=payload,
                timeout=self.config['ollama']['timeout']
            )
            if response.status_code == 200:
                return response.json().get("response", "").strip()
            else:
                print(f"❌ LLM-Fehler: Status {response.status_code}")
                return None
        except Exception as e:
            print(f"❌ LLM-Anfrage-Fehler: {e}")
            return None

    def _extract_category(self, response: str, categories: List[str]) -> str:
        """
        Extrahiert die Kategorie aus der LLM-Antwort.
        Gibt die erste Kategorie zurück, falls keine erkannt wird.
        """
        response_lower = response.lower()
        for category in categories:
            if category.lower() in response_lower:
                return category
        return categories[0] if categories else "Unknown"

    def _get_few_shot_examples(self, examples_df: pd.DataFrame, count: int) -> List[Dict]:
        """
        Wählt zufällig Few-Shot-Beispiele aus. Anzahl = count.
        """
        if count == 0:
            return []
        selected = examples_df.sample(min(count, len(examples_df)), random_state=42)
        return selected.to_dict('records')

    def _extract_ground_truth(self, ticket: Dict) -> str:
        """
        Bestimmt die wahre Kategorie (Ground Truth) aus den Ticket-Tags.
        """
        tags = [str(ticket.get(f'tag_{i}', '')).strip() for i in range(1, 9)]
        categories = self.config.get('categories', ['Hardware', 'Software', 'Network', 'Security'])
        for cat in categories:
            for tag in tags:
                if cat.lower() in tag.lower():
                    return cat
        return categories[0]

    def _classify_ticket(self, ticket: Dict, model: str, few_shot_count: int, prompt_type: str, examples_df: pd.DataFrame, categories: List[str]) -> Dict:
        """
        Klassifiziert ein einzelnes Ticket mit dem angegebenen Modell und Few-Shot-Strategie.
        Gibt ein Ergebnis-Dictionary zurück.
        """
        ticket_text = f"{ticket.get('subject', '')}\n\n{ticket.get('body', '')}"
        few_shot_examples = self._get_few_shot_examples(examples_df, few_shot_count)
        prompt = self._create_prompt(ticket_text, few_shot_examples, prompt_type, categories)
        response = self._query_llm(prompt, model)
        if response is None:
            return {
                'ticket_id': ticket.get('subject', 'Unknown'),
                'ground_truth': self._extract_ground_truth(ticket),
                'prediction': categories[0],
                'model': model,
                'few_shot_count': few_shot_count,
                'prompt_type': prompt_type,
                'response': 'ERROR',
                'correct': False
            }
        predicted_category = self._extract_category(response, categories)
        ground_truth = self._extract_ground_truth(ticket)
        return {
            'ticket_id': ticket.get('subject', 'Unknown'),
            'ground_truth': ground_truth,
            'prediction': predicted_category,
            'model': model,
            'few_shot_count': few_shot_count,
            'prompt_type': prompt_type,
            'response': response,
            'correct': predicted_category == ground_truth
        }

    def run_experiment(self) -> pd.DataFrame:
        """
        Führt das vollständige Experiment durch:
        - Alle Modelle, Few-Shot-Counts, Prompt-Typen und Tickets werden getestet.
        - Ergebnisse werden in einem eigenen Ordner gespeichert.
        """
        print("🚀 Starte Few-Shot Experiment für mehrere Modelle...")
        tickets_df, examples_df = self._load_data()
        if len(tickets_df) == 0:
            print("❌ Keine Tickets gefunden!")
            return pd.DataFrame()
        models = self.config.get('models', ['llama3.1:8b', 'mistral:7b'])
        few_shot_counts = self.config.get('few_shot_counts', [0, 1, 3, 5])
        prompt_types = self.config.get('prompt_types', ['structured', 'unstructured'])
        categories = self.config.get('categories', ['Hardware', 'Software', 'Network', 'Security'])
        print(f"📊 Experiment-Design:")
        print(f"   Modelle: {models}")
        print(f"   Few-Shot-Counts: {few_shot_counts}")
        print(f"   Prompt-Types: {prompt_types}")
        print(f"   Kategorien: {categories}")
        results = []
        total_experiments = len(models) * len(few_shot_counts) * len(prompt_types) * len(tickets_df)
        current_experiment = 0
        for model in models:
            for few_shot_count in few_shot_counts:
                for prompt_type in prompt_types:
                    for _, ticket in tickets_df.iterrows():
                        current_experiment += 1
                        print(f"🔬 Experiment {current_experiment}/{total_experiments}: "
                              f"{model}, {few_shot_count} shots, {prompt_type}")
                        result = self._classify_ticket(
                            ticket.to_dict(), model, few_shot_count, 
                            prompt_type, examples_df, categories
                        )
                        results.append(result)
        # Ergebnisse in einem eigenen Ordner nach Test-ID speichern
        results_dir = os.path.join(self.config.get('paths', {}).get('results_dir', 'results/'), f"test_{self.test_id}")
        os.makedirs(results_dir, exist_ok=True)
        results_df = pd.DataFrame(results)
        results_df.to_csv(os.path.join(results_dir, f"results_{self.test_id}.csv"), index=False)
        print(f"✅ Experiment abgeschlossen! {len(results)} Ergebnisse gespeichert in {results_dir}")
        return results_df

    def analyze_results(self, results_df: pd.DataFrame) -> Dict:
        """
        Analysiert die Ergebnisse wissenschaftlich:
        - Berechnet verschiedene Metriken (Accuracy, F1, Precision, Recall, MCC, Balanced Accuracy)
        - Führt ANOVA durch
        - Speichert Analyse als JSON
        """
        print("📊 Analysiere Ergebnisse...")
        analysis = {}
        # 1. Gesamt-Accuracy
        analysis['overall_accuracy'] = results_df['correct'].mean()
        # 2. Metriken pro Bedingung
        condition_metrics = {}
        for condition in results_df.groupby(['model', 'few_shot_count', 'prompt_type']):
            condition_name = f"{condition[0][0]}_{condition[0][1]}shots_{condition[0][2]}"
            condition_df = condition[1]
            y_true = condition_df['ground_truth']
            y_pred = condition_df['prediction']
            accuracy = (y_true == y_pred).mean()
            f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
            precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
            mcc = matthews_corrcoef(y_true, y_pred) if len(set(y_true)) > 1 else 0.0
            bal_acc = balanced_accuracy_score(y_true, y_pred)
            condition_metrics[condition_name] = {
                'accuracy': accuracy,
                'f1_score': f1,
                'precision': precision,
                'recall': recall,
                'mcc': mcc,
                'balanced_accuracy': bal_acc,
                'n_samples': len(condition_df)
            }
        analysis['condition_metrics'] = condition_metrics
        # 3. ANOVA-Test
        accuracy_data = []
        for condition_name, metrics in condition_metrics.items():
            condition_df = results_df[
                (results_df['model'] == condition_name.split('_')[0]) &
                (results_df['few_shot_count'] == int(condition_name.split('_')[1].replace('shots', ''))) &
                (results_df['prompt_type'] == condition_name.split('_')[2])
            ]
            accuracy_data.append(condition_df['correct'].values)
        if len(accuracy_data) > 1:
            f_stat, p_value = stats.f_oneway(*accuracy_data)
            analysis['anova'] = {
                'f_statistic': f_stat,
                'p_value': p_value,
                'significant': p_value < 0.05
            }
        # 4. Speichere Analyse
        results_dir = os.path.join(self.config.get('paths', {}).get('results_dir', 'results/'), f"test_{self.test_id}")
        with open(os.path.join(results_dir, f"analysis_{self.test_id}.json"), "w") as f:
            json.dump(analysis, f, indent=2, default=str)
        print("✅ Analyse abgeschlossen!")
        return analysis

    def create_visualizations(self, results_df: pd.DataFrame, analysis: Dict):
        """
        Erstellt wissenschaftliche Visualisierungen:
        - Speichert alle Plots im Ergebnis-Ordner
        - Verbesserte Achsen, Titel, Farben
        """
        print("📈 Erstelle Visualisierungen...")
        results_dir = os.path.join(self.config.get('paths', {}).get('results_dir', 'results/'), f"test_{self.test_id}")
        # 1. Accuracy-Vergleich
        plt.figure(figsize=(15, 8))
        conditions = list(analysis['condition_metrics'].keys())
        accuracies = [analysis['condition_metrics'][c]['accuracy'] for c in conditions]
        colors = ['skyblue' if 'structured' in c else 'lightcoral' for c in conditions]
        bars = plt.bar(range(len(conditions)), accuracies, color=colors)
        plt.title('Accuracy pro Experiment-Bedingung', fontweight='bold', fontsize=14)
        plt.xlabel('Bedingung (Modell, Few-Shot, Prompt)')
        plt.ylabel('Accuracy')
        plt.xticks(range(len(conditions)), conditions, rotation=45, ha='right')
        plt.ylim(0, 1)
        for i, v in enumerate(accuracies):
            plt.text(i, v + 0.01, f'{v:.2%}', ha='center', va='bottom')
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor='skyblue', label='Structured'), Patch(facecolor='lightcoral', label='Unstructured')]
        plt.legend(handles=legend_elements)
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, f"accuracy_{self.test_id}.png"), dpi=300, bbox_inches='tight')
        plt.close()
        # 2. F1-Score Heatmap
        pivot_data = []
        for condition, metrics in analysis['condition_metrics'].items():
            parts = condition.split('_')
            model = parts[0]
            few_shot = int(parts[1].replace('shots', ''))
            prompt_type = parts[2]
            pivot_data.append({
                'model': model,
                'few_shot_count': few_shot,
                'prompt_type': prompt_type,
                'f1_score': metrics['f1_score']
            })
        pivot_df = pd.DataFrame(pivot_data)
        heatmap_data = pivot_df.pivot_table(values='f1_score', index='model', columns='few_shot_count', aggfunc='mean')
        plt.figure(figsize=(10, 6))
        sns.heatmap(heatmap_data, annot=True, fmt='.2%', cmap='Blues', cbar_kws={'label': 'F1-Score'})
        plt.title('F1-Score: Modell vs. Few-Shot-Count', fontweight='bold')
        plt.xlabel('Few-Shot-Count')
        plt.ylabel('Modell')
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, f"f1_heatmap_{self.test_id}.png"), dpi=300, bbox_inches='tight')
        plt.close()
        print("✅ Visualisierungen erstellt!")

    def generate_report(self, results_df: pd.DataFrame, analysis: Dict):
        """
        Erstellt einen wissenschaftlichen Bericht (Markdown) mit allen Ergebnissen und Erklärungen.
        """
        print("📝 Generiere wissenschaftlichen Bericht...")
        results_dir = os.path.join(self.config.get('paths', {}).get('results_dir', 'results/'), f"test_{self.test_id}")
        report = f"""
# Few-Shot Learning Experiment Bericht

**Test-ID:** {self.test_id}  
**Datum:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}  
**Gesamt-Accuracy:** {analysis['overall_accuracy']:.2%}  
**Anzahl Experimente:** {len(results_df)}

## Experiment-Design

- **Modelle:** {', '.join(self.config.get('models', []))}
- **Few-Shot-Counts:** {self.config.get('few_shot_counts', [])}
- **Prompt-Types:** {self.config.get('prompt_types', [])}
- **Kategorien:** {', '.join(self.config.get('categories', []))}

## Ergebnisse pro Bedingung
"""
        for condition, metrics in analysis['condition_metrics'].items():
            report += f"""
### {condition}
- **Accuracy:** {metrics['accuracy']:.2%}
- **F1-Score:** {metrics['f1_score']:.2%}
- **Precision:** {metrics['precision']:.2%}
- **Recall:** {metrics['recall']:.2%}
- **MCC:** {metrics['mcc']:.2%}
- **Balanced Accuracy:** {metrics['balanced_accuracy']:.2%}
- **Stichprobengröße:** {metrics['n_samples']}
"""
        if 'anova' in analysis:
            report += f"""
## Statistische Analyse

**ANOVA-Ergebnisse:**
- **F-Statistik:** {analysis['anova']['f_statistic']:.4f}
- **p-Wert:** {analysis['anova']['p_value']:.4f}
- **Statistisch signifikant:** {analysis['anova']['significant']}
"""
        report += f"""
## Schlussfolgerungen

1. **Beste Bedingung:** {max(analysis['condition_metrics'].items(), key=lambda x: x[1]['accuracy'])[0]} mit {max(analysis['condition_metrics'].items(), key=lambda x: x[1]['accuracy'])[1]['accuracy']:.2%} Accuracy
2. **Few-Shot-Learning Effekt:** {'Ja' if any('1shots' in k or '3shots' in k or '5shots' in k for k in analysis['condition_metrics'].keys()) else 'Nein'}
3. **Modell-Unterschiede:** {'Signifikant' if analysis.get('anova', {}).get('significant', False) else 'Nicht signifikant'}

## Wissenschaftliche Hinweise
- **Accuracy**: Anteil korrekt klassifizierter Tickets
- **F1-Score**: Harm. Mittel von Precision und Recall (gut bei unbalancierten Daten)
- **Precision**: Wie viele Vorhersagen waren korrekt?
- **Recall**: Wie viele der echten Fälle wurden gefunden?
- **MCC**: Matthews Correlation Coefficient (robust bei unbalancierten Klassen)
- **Balanced Accuracy**: Durchschnittliche Recall über alle Klassen

## Reproduzierbarkeit
- Random Seed: {self.config.get('experiment', {}).get('random_seed', 42)}
- Alle Parameter in config.yaml dokumentiert
- Rohdaten, Analyse und Grafiken im Ergebnis-Ordner gespeichert
"""
        with open(os.path.join(results_dir, f"report_{self.test_id}.md"), "w", encoding="utf-8") as f:
            f.write(report)
        print("✅ Bericht generiert!")

def main():
    """
    Hauptfunktion: Führt das Experiment aus, analysiert und visualisiert die Ergebnisse.
    """
    print("🔬 Few-Shot Learning Experiment (Llama3.1 vs. Mistral)")
    print("=" * 50)
    experiment = FewShotExperiment()
    results_df = experiment.run_experiment()
    if len(results_df) > 0:
        analysis = experiment.analyze_results(results_df)
        experiment.create_visualizations(results_df, analysis)
        experiment.generate_report(results_df, analysis)
        print("\n📋 Experiment-Zusammenfassung:")
        print(f"   Gesamt-Accuracy: {analysis['overall_accuracy']:.2%}")
        print(f"   Anzahl Experimente: {len(results_df)}")
        print(f"   Test-ID: {experiment.test_id}")
        if 'anova' in analysis:
            print(f"   ANOVA p-Wert: {analysis['anova']['p_value']:.4f}")
            print(f"   Statistisch signifikant: {analysis['anova']['significant']}")
        best_condition = max(analysis['condition_metrics'].items(), key=lambda x: x[1]['accuracy'])
        print(f"   Beste Bedingung: {best_condition[0]} ({best_condition[1]['accuracy']:.2%} Accuracy)")
    print("✅ Experiment abgeschlossen!")

if __name__ == "__main__":
    main() 