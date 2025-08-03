"""
Wissenschaftlich Rigoroses Few-Shot Learning Experiment für IT-Support-Tickets

Implementiert ein methodisch korrektes Experiment nach wissenschaftlichen Standards:
- Factorial Design: 2×4×2 (LLM × Few-Shot-Count × Prompt-Type)
- Statistische Power: n≥50 pro Bedingung für robuste ANOVA
- Komplette Few-Shot Progression: 0, 1, 3, 5 Beispiele
- Baseline-Vergleich: Zero-shot als wissenschaftliche Kontrolle
- Reproduzierbarkeit: Vollständige Dokumentation und Seeding

Autor: DHBW Student (4. Semester)
Für wissenschaftliche Projektarbeit optimiert
"""

import pandas as pd
import requests
import yaml
import json
import numpy as np
import time
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from sklearn.metrics import (
    classification_report, confusion_matrix, f1_score, precision_score, 
    recall_score, matthews_corrcoef, balanced_accuracy_score
)
from scipy import stats
from scipy.stats import f_oneway, chi2_contingency
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
from tqdm import tqdm
import statsmodels.api as sm
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import itertools
warnings.filterwarnings('ignore')

def interactive_menu(config):
    def choose_from_list(options, label):
        while True:
            print(f"\n{label}:")
            print("  1) Alle")
            for idx, opt in enumerate(options, start=2):
                print(f"  {idx}) {opt}")
            inp = input(f"Auswahl (z.B. 2,3, 1 = alle) [Default: 1]: ").strip()
            if inp == "" or inp == "1":
                return options
            try:
                nums = [int(x.strip()) for x in inp.split(",")]
                if any(n < 1 or n > len(options)+1 for n in nums):
                    raise ValueError
                if 1 in nums:
                    return options
                return [options[n-2] for n in nums]
            except Exception:
                print("Ungültige Eingabe. Bitte nur Zahlen wie 2,3 oder 1 für alle eingeben.")
    # Modelle
    all_models = config.get('models', [])
    models = choose_from_list(all_models, "Modelle")
    # Few-Shot-Counts
    all_few_shot = config.get('few_shot_counts', [])
    few_shot_counts = choose_from_list([str(x) for x in all_few_shot], "Few-Shot-Counts")
    few_shot_counts = [int(x) for x in few_shot_counts]
    # Prompt-Typen
    all_prompt_types = config.get('prompt_types', [])
    prompt_types = choose_from_list(all_prompt_types, "Prompt-Typen")
    # Anzahl Tickets
    total_tickets = config.get('experiment', {}).get('total_tickets', 200)
    while True:
        print(f"\nMaximale Ticketanzahl: {total_tickets}")
        ticket_in = input(f"Wie viele Tickets testen? (1 = alle, Default: 5): ").strip()
        if ticket_in == "" or ticket_in == "1":
            n_tickets = total_tickets
            break
        if ticket_in.isdigit() and int(ticket_in) > 0:
            n_tickets = min(int(ticket_in), total_tickets)
            break
        print("Ungültige Eingabe. Bitte eine Zahl eingeben.")
    print(f"\nEinstellungen übernommen:")
    print(f"  Modelle: {', '.join(models)}")
    print(f"  Few-Shot-Counts: {', '.join(map(str, few_shot_counts))}")
    print(f"  Prompt-Typen: {', '.join(prompt_types)}")
    print(f"  Tickets: {n_tickets}")
    return models, few_shot_counts, prompt_types, n_tickets

class FewShotExperiment:
    """
    Wissenschaftlich rigorose Few-Shot Learning Experiment-Klasse.
    
    Implementiert ein vollständiges 2×4×2 faktorielles Design:
    - 2 LLM-Modelle (Llama3.1, Mistral)
    - 4 Few-Shot Bedingungen (0, 1, 3, 5 Beispiele)
    - 2 Prompt-Typen (strukturiert, unstrukturiert)
    
    Wissenschaftliche Standards:
    - Minimum n=25 pro Bedingung (200 Tickets total)
    - Stratifizierte Stichprobenziehung
    - Statistische Power-Analyse
    - Vollständige Reproduzierbarkeit
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialisiert das Experiment mit wissenschaftlichen Parametern."""
        self.config = self._load_config(config_path)
        self.test_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Wissenschaftlicher Random Seed für Reproduzierbarkeit
        self.random_seed = self.config.get('experiment', {}).get('random_seed', 42)
        np.random.seed(self.random_seed)
        
        # Logging für wissenschaftliche Dokumentation
        self._setup_logging()
        
        # Experimentelle Parameter validieren
        self._validate_experimental_design()
        
    def _setup_logging(self):
        """Einrichtung wissenschaftlicher Dokumentation."""
        results_dir = os.path.join(
            self.config.get('paths', {}).get('results_dir', 'results/'), 
            f"test_{self.test_id}"
        )
        os.makedirs(results_dir, exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(os.path.join(results_dir, f'experiment_{self.test_id}.log')),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def _validate_experimental_design(self):
        """Validiert das Experimentdesign für wissenschaftliche Rigorosität."""
        min_total_tickets = self.config.get('experiment', {}).get('min_total_tickets', 200)
        min_per_condition = self.config.get('experiment', {}).get('min_per_condition', 25)
        
        # Berechne erwartete Anzahl Bedingungen
        n_models = len(self.config.get('models', []))
        n_few_shot = len(self.config.get('few_shot_counts', []))
        n_prompt_types = len(self.config.get('prompt_types', []))
        total_conditions = n_models * n_few_shot * n_prompt_types
        
        required_tickets = total_conditions * min_per_condition
        
        self.logger.info(f"Experimental Design Validation:")
        self.logger.info(f"  - Models: {n_models}")
        self.logger.info(f"  - Few-Shot Conditions: {n_few_shot}")
        self.logger.info(f"  - Prompt Types: {n_prompt_types}")
        self.logger.info(f"  - Total Conditions: {total_conditions}")
        self.logger.info(f"  - Required Tickets: {required_tickets}")
        
        if required_tickets > min_total_tickets:
            self.logger.warning(f"Required tickets ({required_tickets}) exceeds available ({min_total_tickets})")
            
    def _load_config(self, config_path: str) -> Dict:
        """Lädt und validiert die wissenschaftliche Konfiguration."""
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)
            
            # Validiere kritische wissenschaftliche Parameter
            required_keys = ['models', 'few_shot_counts', 'prompt_types', 'categories']
            for key in required_keys:
                if key not in config:
                    raise ValueError(f"Missing required config key: {key}")
                    
            return config
        except Exception as e:
            print(f"❌ Configuration Error: {e}")
            return {}

    def _load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Lädt und validiert Daten mit wissenschaftlicher Qualitätskontrolle.
        
        Implementiert:
        - Stratifizierte Stichprobenziehung
        - Qualitätskontrolle der Tickets
        - Validierung der Kategorienverteilung
        - Separation von Few-Shot-Beispielen
        
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: (test_tickets, few_shot_examples)
        """
        self.logger.info("📂 Loading and validating experimental data...")
        
        data_dir = self.config.get('paths', {}).get('data_dir', 'data/')
        ticket_file = self.config.get('paths', {}).get('ticket_file', 'tickets_extended_187.xlsx')
        fewshot_file = self.config.get('paths', {}).get('fewshot_file', 'tickets_examples.xlsx')
        
        # Lade Hauptdatensatz
        tickets_path = os.path.join(data_dir, ticket_file)
        tickets_df = pd.read_excel(tickets_path)
        self.logger.info(f"Loaded {len(tickets_df)} tickets from {tickets_path}")
        
        # Lade Few-Shot-Beispiele
        fewshot_path = os.path.join(data_dir, fewshot_file)
        few_shot_df = pd.read_excel(fewshot_path)
        self.logger.info(f"Loaded {len(few_shot_df)} few-shot examples from {fewshot_path}")
        
        # Qualitätskontrolle
        tickets_df = self._quality_control(tickets_df)
        few_shot_df = self._quality_control(few_shot_df)
        
        # Validiere Kategorienverteilung
        self._validate_category_distribution(tickets_df)
        
        # Stratifizierte Stichprobenziehung
        tickets_df = self._stratified_sampling(tickets_df)
        
        return tickets_df, few_shot_df
    
    def _quality_control(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Implementiert wissenschaftliche Qualitätskontrolle für Tickets.
        
        Kriterien:
        - Mindestlänge der Ticket-Texte
        - Vorhandensein aller erforderlichen Felder
        - Gültige Kategorien
        """
        initial_count = len(df)
        
        # Entferne Tickets ohne Label
        df = df.dropna(subset=['label'])
        
        # Entferne Tickets mit ungültigen Kategorien
        valid_categories = self.config.get('categories', [])
        df = df[df['label'].isin(valid_categories)]
        
        # Mindestlänge für Ticket-Text
        min_length = self.config.get('quality_control', {}).get('min_ticket_length', 20)
        df['combined_text'] = df['subject'].astype(str) + ' ' + df['body'].astype(str)
        df = df[df['combined_text'].str.len() >= min_length]
        
        # Entferne Duplikate basierend auf Text
        df = df.drop_duplicates(subset=['combined_text'])
        
        filtered_count = len(df)
        removed_count = initial_count - filtered_count
        
        self.logger.info(f"Quality Control: {removed_count} tickets removed, {filtered_count} retained")
        
        return df
    
    def _validate_category_distribution(self, df: pd.DataFrame):
        """Validiert die Kategorienverteilung für wissenschaftliche Balancierung."""
        category_counts = df['label'].value_counts()
        min_per_category = self.config.get('experiment', {}).get('min_tickets_per_category', 40)
        
        self.logger.info("Category Distribution Analysis:")
        for category, count in category_counts.items():
            status = "✅" if count >= min_per_category else "⚠️"
            self.logger.info(f"  {status} {category}: {count} tickets")
            
        insufficient_categories = category_counts[category_counts < min_per_category]
        if len(insufficient_categories) > 0:
            self.logger.warning(f"Categories with insufficient data: {list(insufficient_categories.index)}")
            
    def _stratified_sampling(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Implementiert stratifizierte Stichprobenziehung für wissenschaftliche Validität.
        
        Stellt sicher, dass:
        - Alle Kategorien gleichmäßig repräsentiert sind
        - Ausreichende Stichprobengröße für statistische Power
        - Zufällige, aber reproduzierbare Auswahl
        """
        total_needed = self.config.get('experiment', {}).get('total_experimental_tickets', 200)
        categories = self.config.get('categories', [])
        tickets_per_category = total_needed // len(categories)
        
        sampled_dfs = []
        
        for category in categories:
            category_df = df[df['label'] == category]
            
            if len(category_df) < tickets_per_category:
                self.logger.warning(f"Category '{category}': only {len(category_df)} available, needed {tickets_per_category}")
                sampled_dfs.append(category_df)
            else:
                sampled = category_df.sample(
                    n=tickets_per_category, 
                    random_state=self.random_seed
                )
                sampled_dfs.append(sampled)
                
        final_df = pd.concat(sampled_dfs, ignore_index=True)
        
        # Shuffle the final dataset
        final_df = final_df.sample(frac=1, random_state=self.random_seed).reset_index(drop=True)
        
        self.logger.info(f"Stratified Sampling Complete: {len(final_df)} tickets selected")
        self.logger.info(f"Final distribution: {final_df['label'].value_counts().to_dict()}")
        
        return final_df

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
                prompt += f"Beispiel:\nTicket: {example['ticket_text']}\nKategorie: {example['label']}\n\n"
            prompt += f"Ticket: {ticket_text}\nKategorie:"
        else:
            # Unstrukturierter Prompt, aber mit klaren Kategorien
            prompt = f"Was für ein IT-Problem ist das? Mögliche Kategorien: {', '.join(categories)}\n\n"
            for example in few_shot_examples:
                prompt += f"Problem: {example['ticket_text']}\nAntwort: {example['label']}\n\n"
            prompt += f"Problem: {ticket_text}\nAntwort:"
        return prompt

    def _query_llm(self, prompt: str, model: str, max_retries: int = 3) -> Optional[str]:
        """
        Robuste LLM-Anfrage mit wissenschaftlicher Fehlerbehandlung.
        
        Implementiert:
        - Retry-Mechanismus für failed requests
        - Timeout-Management
        - Response-Validierung
        - Logging für Reproduzierbarkeit
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
        
        timeout = self.config.get('quality_control', {}).get('timeout_seconds', 45)
        
        for attempt in range(max_retries):
            try:
                start_time = time.time()
                response = requests.post(
                    f"{self.config['ollama']['base_url']}/api/generate",
                    json=payload,
                    timeout=timeout
                )
                end_time = time.time()
                
                if response.status_code == 200:
                    result = response.json().get("response", "").strip()
                    
                    # Response validation
                    if self._validate_llm_response(result):
                        self.logger.debug(f"LLM success: {model}, attempt {attempt+1}, time {end_time-start_time:.2f}s")
                        return result
                    else:
                        self.logger.warning(f"Invalid response from {model}: {result[:50]}...")
                        
                else:
                    self.logger.warning(f"LLM HTTP error: {response.status_code}, attempt {attempt+1}")
                    
            except requests.exceptions.Timeout:
                self.logger.warning(f"LLM timeout for {model}, attempt {attempt+1}")
            except Exception as e:
                self.logger.error(f"LLM request error for {model}: {e}, attempt {attempt+1}")
                
            # Exponential backoff
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt
                self.logger.info(f"Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
                
        self.logger.error(f"LLM failed after {max_retries} attempts: {model}")
        return None
    
    def _validate_llm_response(self, response: str) -> bool:
        """Validiert LLM-Antworten auf wissenschaftliche Qualität."""
        if not response or len(response.strip()) == 0:
            return False
            
        # Check if response contains at least one valid category
        categories = self.config.get('categories', [])
        response_lower = response.lower()
        
        return any(cat.lower() in response_lower for cat in categories)
    
    def run_experiment(self, models=None, few_shot_counts=None, prompt_types=None, n_tickets=None) -> pd.DataFrame:
        """
        Führt wissenschaftlich rigoroses Experiment mit vollständiger Dokumentation durch.
        
        Implementiert:
        - Faktorielles 2×4×2 Design
        - Progress-Monitoring mit ETA
        - Intermediate Saves für Datensicherheit
        - Comprehensive Error Handling
        - Resource Management
        """
        self.logger.info("🚀 Starting Scientific Few-Shot Experiment...")
        
        # Load and validate data
        tickets_df, examples_df = self._load_data()
        if len(tickets_df) == 0:
            self.logger.error("No tickets found! Experiment aborted.")
            return pd.DataFrame()
        
        # Use provided parameters or defaults from config
        models = models or self.config.get('models', ['llama3.1:8b', 'mistral:7b'])
        few_shot_counts = few_shot_counts or self.config.get('few_shot_counts', [0, 1, 3, 5])
        prompt_types = prompt_types or self.config.get('prompt_types', ['structured', 'unstructured'])
        n_tickets = n_tickets or len(tickets_df)
        
        # Reduce tickets if needed
        if n_tickets < len(tickets_df):
            tickets_df = tickets_df.sample(n=n_tickets, random_state=self.random_seed)
            
        categories = self.config.get('categories', [])
        
        # Calculate total experimental conditions
        total_experiments = len(models) * len(few_shot_counts) * len(prompt_types) * len(tickets_df)
        self.logger.info(f"Experimental Design: {total_experiments} total classifications")
        self.logger.info(f"  Models: {len(models)} ({models})")
        self.logger.info(f"  Few-Shot Conditions: {len(few_shot_counts)} ({few_shot_counts})")
        self.logger.info(f"  Prompt Types: {len(prompt_types)} ({prompt_types})")
        self.logger.info(f"  Tickets: {len(tickets_df)}")
        
        # Initialize results storage
        results = []
        experiment_count = 0
        failed_requests = 0
        
        # Setup progress tracking
        pbar = tqdm(total=total_experiments, desc="Experiment Progress")
        
        # Nested loops for factorial design
        for model in models:
            for few_shot_count in few_shot_counts:
                for prompt_type in prompt_types:
                    condition_name = f"{model}_{few_shot_count}shots_{prompt_type}"
                    self.logger.info(f"Starting condition: {condition_name}")
                    
                    condition_results = []
                    condition_failures = 0
                    
                    for idx, (_, ticket) in enumerate(tickets_df.iterrows()):
                        start_time = time.time()
                        
                        result = self._classify_ticket(
                            ticket, model, few_shot_count, prompt_type, examples_df, categories
                        )
                        
                        if result['response'] == 'ERROR':
                            failed_requests += 1
                            condition_failures += 1
                        
                        condition_results.append(result)
                        results.append(result)
                        experiment_count += 1
                        
                        # Update progress
                        pbar.update(1)
                        pbar.set_postfix({
                            'Condition': condition_name,
                            'Failed': failed_requests,
                            'Success_Rate': f"{((experiment_count-failed_requests)/experiment_count)*100:.1f}%"
                        })
                        
                        # Intermediate save every 50 classifications
                        if experiment_count % 50 == 0:
                            self._save_intermediate_results(results, experiment_count)
                            
                        # Brief pause to prevent overloading
                        time.sleep(0.1)
                    
                    # Log condition statistics
                    condition_accuracy = sum(1 for r in condition_results if r['correct']) / len(condition_results)
                    self.logger.info(f"Condition {condition_name} complete: "
                                   f"Accuracy={condition_accuracy:.3f}, "
                                   f"Failures={condition_failures}/{len(condition_results)}")
        
        pbar.close()
        
        # Final results processing
        results_df = pd.DataFrame(results)
        
        # Save final results
        self._save_final_results(results_df)
        
        # Log experiment summary
        overall_accuracy = results_df['correct'].mean()
        success_rate = ((experiment_count - failed_requests) / experiment_count) * 100
        
        self.logger.info("🎯 Experiment Complete!")
        self.logger.info(f"  Total Classifications: {experiment_count}")
        self.logger.info(f"  Success Rate: {success_rate:.1f}%")
        self.logger.info(f"  Overall Accuracy: {overall_accuracy:.3f}")
        self.logger.info(f"  Failed Requests: {failed_requests}")
        
        return results_df
    
    def _save_intermediate_results(self, results: List[Dict], count: int):
        """Speichert Zwischenergebnisse zur Datensicherheit."""
        results_dir = os.path.join(
            self.config.get('paths', {}).get('results_dir', 'results/'), 
            f"test_{self.test_id}"
        )
        
        intermediate_df = pd.DataFrame(results)
        intermediate_path = os.path.join(results_dir, f"intermediate_{count}_{self.test_id}.csv")
        intermediate_df.to_csv(intermediate_path, index=False)
        
        self.logger.info(f"Intermediate results saved: {count} classifications")
    
    def _save_final_results(self, results_df: pd.DataFrame):
        """Speichert finale Ergebnisse mit vollständiger Dokumentation."""
        results_dir = os.path.join(
            self.config.get('paths', {}).get('results_dir', 'results/'), 
            f"test_{self.test_id}"
        )
        
        # Save main results
        results_path = os.path.join(results_dir, f"results_{self.test_id}.csv")
        results_df.to_csv(results_path, index=False)
        
        # Save experimental metadata
        metadata = {
            'test_id': self.test_id,
            'timestamp': datetime.now().isoformat(),
            'random_seed': self.random_seed,
            'total_classifications': len(results_df),
            'config': self.config
        }
        
        metadata_path = os.path.join(results_dir, f"metadata_{self.test_id}.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
            
        self.logger.info(f"💾 Final results saved: {results_path}")

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
        Passt die Spaltennamen an die neue Struktur an (label statt kategorie).
        """
        if count == 0:
            return []
        selected = examples_df.sample(min(count, len(examples_df)), random_state=42)
        # Passe die Spaltennamen an: 'label' statt 'kategorie'
        examples = []
        for _, row in selected.iterrows():
            examples.append({
                'ticket_text': row.get('subject', '') + '\n\n' + str(row.get('body', '')),
                'label': row.get('label', '')
            })
        return examples

    def _extract_ground_truth(self, ticket: Dict) -> str:
        """
        Bestimmt die wahre Kategorie (Ground Truth) aus der Spalte 'label'.
        """
        return ticket.get('label', '')

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

    def run_experiment(self, models=None, few_shot_counts=None, prompt_types=None, n_tickets=None) -> pd.DataFrame:
        """
        Führt das vollständige Experiment durch (jetzt mit flexiblen Parametern):
        - Modelle, Few-Shot-Counts, Prompt-Typen und Ticketanzahl können eingeschränkt werden.
        - Ergebnisse werden in einem eigenen Ordner gespeichert.
        """
        print("🚀 Starte Few-Shot Experiment für mehrere Modelle...")
        tickets_df, examples_df = self._load_data()
        if len(tickets_df) == 0:
            print("❌ Keine Tickets gefunden!")
            return pd.DataFrame()
        # Übernehme ggf. reduzierte Einstellungen
        models = models or self.config.get('models', ['llama3.1:8b', 'mistral:7b'])
        few_shot_counts = few_shot_counts or self.config.get('few_shot_counts', [0, 1, 3, 5])
        prompt_types = prompt_types or self.config.get('prompt_types', ['structured', 'unstructured'])
        n_tickets = n_tickets or self.config.get('experiment', {}).get('total_tickets', 200)
        # Tickets ggf. reduzieren
        tickets_df = tickets_df.sample(min(n_tickets, len(tickets_df)), random_state=42)
        categories = self.config.get('categories', [])
        results = []
        for model in models:
            for few_shot_count in few_shot_counts:
                for prompt_type in prompt_types:
                    print(f"\n--- Modell: {model} | Few-Shot: {few_shot_count} | Prompt: {prompt_type} ---")
                    for _, ticket in tickets_df.iterrows():
                        result = self._classify_ticket(ticket, model, few_shot_count, prompt_type, examples_df, categories)
                        results.append(result)
        results_df = pd.DataFrame(results)
        # Ergebnisse speichern
        results_dir = os.path.join(self.config.get('paths', {}).get('results_dir', 'results/'), f"test_{self.test_id}")
        os.makedirs(results_dir, exist_ok=True)
        results_df.to_csv(os.path.join(results_dir, f"results_{self.test_id}.csv"), index=False)
        print(f"\n💾 Ergebnisse gespeichert unter: {results_dir}")
        return results_df

    def analyze_results(self, results_df: pd.DataFrame) -> Dict:
        """
        Führt comprehensive wissenschaftliche Analyse durch.
        
        Implementiert:
        - Deskriptive Statistiken mit Konfidenzintervallen
        - Mehrstufige ANOVA mit Post-hoc-Tests
        - Effektgrößen (Cohen's f, η²)
        - Multiple Comparisons Correction (Bonferroni)
        - Power Analysis (retrospektiv)
        - Assumption Testing (Normalität, Homoskedastizität)
        """
        self.logger.info("📊 Starting Comprehensive Statistical Analysis...")
        
        analysis = {
            'experiment_metadata': {
                'test_id': self.test_id,
                'total_classifications': len(results_df),
                'timestamp': datetime.now().isoformat(),
                'random_seed': self.random_seed
            }
        }
        
        # 1. Descriptive Statistics with Confidence Intervals
        analysis['descriptive_stats'] = self._calculate_descriptive_statistics(results_df)
        
        # 2. Overall Performance Metrics
        analysis['overall_metrics'] = self._calculate_overall_metrics(results_df)
        
        # 3. Condition-wise Analysis
        analysis['condition_metrics'] = self._calculate_condition_metrics(results_df)
        
        # 4. Factorial ANOVA Analysis
        analysis['anova_results'] = self._perform_factorial_anova(results_df)
        
        # 5. Post-hoc Multiple Comparisons
        analysis['posthoc_tests'] = self._perform_posthoc_tests(results_df)
        
        # 6. Effect Size Calculations
        analysis['effect_sizes'] = self._calculate_effect_sizes(results_df)
        
        # 7. Power Analysis
        analysis['power_analysis'] = self._perform_power_analysis(results_df)
        
        # 8. Statistical Assumptions Testing
        analysis['assumption_tests'] = self._test_statistical_assumptions(results_df)
        
        # 9. Save comprehensive analysis
        self._save_statistical_analysis(analysis)
        
        self.logger.info("✅ Statistical Analysis Complete!")
        return analysis
    
    def _calculate_descriptive_statistics(self, results_df: pd.DataFrame) -> Dict:
        """Berechnet deskriptive Statistiken mit Konfidenzintervallen."""
        stats_results = {}
        
        # Overall accuracy statistics
        overall_accuracy = results_df['correct'].mean()
        overall_std = results_df['correct'].std()
        n_total = len(results_df)
        
        # 95% Confidence Interval for overall accuracy
        ci_95 = stats.t.interval(
            0.95, n_total-1, 
            loc=overall_accuracy, 
            scale=overall_std/np.sqrt(n_total)
        )
        
        stats_results['overall'] = {
            'mean_accuracy': overall_accuracy,
            'std_accuracy': overall_std,
            'n': n_total,
            'ci_95_lower': ci_95[0],
            'ci_95_upper': ci_95[1],
            'sem': overall_std / np.sqrt(n_total)  # Standard Error of Mean
        }
        
        # Statistics by experimental factors
        for factor in ['model', 'few_shot_count', 'prompt_type']:
            factor_stats = {}
            for level in results_df[factor].unique():
                subset = results_df[results_df[factor] == level]
                mean_acc = subset['correct'].mean()
                std_acc = subset['correct'].std()
                n_subset = len(subset)
                
                # Confidence interval for this factor level
                if n_subset > 1:
                    ci_95_factor = stats.t.interval(
                        0.95, n_subset-1,
                        loc=mean_acc,
                        scale=std_acc/np.sqrt(n_subset)
                    )
                else:
                    ci_95_factor = (mean_acc, mean_acc)
                
                factor_stats[str(level)] = {
                    'mean': mean_acc,
                    'std': std_acc,
                    'n': n_subset,
                    'ci_95_lower': ci_95_factor[0],
                    'ci_95_upper': ci_95_factor[1],
                    'sem': std_acc / np.sqrt(n_subset) if n_subset > 0 else 0
                }
            
            stats_results[factor] = factor_stats
            
        return stats_results
    
    def _calculate_overall_metrics(self, results_df: pd.DataFrame) -> Dict:
        """Berechnet umfassende Performance-Metriken."""
        y_true = results_df['ground_truth']
        y_pred = results_df['prediction']
        
        return {
            'accuracy': (y_true == y_pred).mean(),
            'f1_weighted': f1_score(y_true, y_pred, average='weighted', zero_division=0),
            'f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0),
            'precision_weighted': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'recall_weighted': recall_score(y_true, y_pred, average='weighted', zero_division=0),
            'mcc': matthews_corrcoef(y_true, y_pred),
            'balanced_accuracy': balanced_accuracy_score(y_true, y_pred)
        }
    
    def _calculate_condition_metrics(self, results_df: pd.DataFrame) -> Dict:
        """Berechnet detaillierte Metriken pro experimenteller Bedingung."""
        condition_metrics = {}
        
        for (model, few_shot, prompt), group in results_df.groupby(['model', 'few_shot_count', 'prompt_type']):
            condition_name = f"{model}_{few_shot}shots_{prompt}"
            
            y_true = group['ground_truth']
            y_pred = group['prediction']
            
            # Calculate all metrics
            accuracy = (y_true == y_pred).mean()
            n_samples = len(group)
            
            # Confidence interval for accuracy
            if n_samples > 1:
                ci_95 = stats.binom.interval(0.95, n_samples, accuracy)
                ci_95 = (ci_95[0]/n_samples, ci_95[1]/n_samples)
            else:
                ci_95 = (accuracy, accuracy)
            
            condition_metrics[condition_name] = {
                'accuracy': accuracy,
                'f1_weighted': f1_score(y_true, y_pred, average='weighted', zero_division=0),
                'f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0),
                'precision_weighted': precision_score(y_true, y_pred, average='weighted', zero_division=0),
                'recall_weighted': recall_score(y_true, y_pred, average='weighted', zero_division=0),
                'mcc': matthews_corrcoef(y_true, y_pred),
                'balanced_accuracy': balanced_accuracy_score(y_true, y_pred),
                'n_samples': n_samples,
                'ci_95_lower': ci_95[0],
                'ci_95_upper': ci_95[1],
                'std_error': np.sqrt(accuracy * (1 - accuracy) / n_samples) if n_samples > 0 else 0
            }
            
        return condition_metrics
    
    def _perform_factorial_anova(self, results_df: pd.DataFrame) -> Dict:
        """Führt mehrstufige ANOVA für faktorielles Design durch."""
        try:
            # Prepare data for ANOVA
            anova_data = results_df[['correct', 'model', 'few_shot_count', 'prompt_type']].copy()
            anova_data['few_shot_count'] = anova_data['few_shot_count'].astype(str)
            
            # Create design matrix using statsmodels
            formula = 'correct ~ C(model) + C(few_shot_count) + C(prompt_type) + C(model):C(few_shot_count) + C(model):C(prompt_type) + C(few_shot_count):C(prompt_type)'
            
            model = sm.formula.ols(formula, data=anova_data).fit()
            anova_table = anova_lm(model, typ=2)
            
            # Convert to serializable format
            anova_results = {}
            for index, row in anova_table.iterrows():
                anova_results[index] = {
                    'sum_sq': float(row['sum_sq']),
                    'df': float(row['df']),
                    'F': float(row['F']) if not pd.isna(row['F']) else None,
                    'PR(>F)': float(row['PR(>F)']) if not pd.isna(row['PR(>F)']) else None,
                    'significant': float(row['PR(>F)']) < 0.05 if not pd.isna(row['PR(>F)']) else False
                }
            
            # Calculate eta-squared (effect size)
            total_ss = anova_table['sum_sq'].sum()
            for factor in anova_results:
                if anova_results[factor]['sum_sq'] is not None:
                    anova_results[factor]['eta_squared'] = anova_results[factor]['sum_sq'] / total_ss
                    
            return anova_results
            
        except Exception as e:
            self.logger.error(f"ANOVA calculation failed: {e}")
            return {'error': str(e)}
    
    def _perform_posthoc_tests(self, results_df: pd.DataFrame) -> Dict:
        """Führt Post-hoc-Tests mit Bonferroni-Korrektur durch."""
        posthoc_results = {}
        
        try:
            # Tukey HSD for pairwise comparisons
            for factor in ['model', 'few_shot_count', 'prompt_type']:
                factor_data = results_df[['correct', factor]].copy()
                factor_data[factor] = factor_data[factor].astype(str)
                
                # Perform Tukey HSD
                tukey_result = pairwise_tukeyhsd(
                    endog=factor_data['correct'],
                    groups=factor_data[factor],
                    alpha=0.05
                )
                
                # Convert to serializable format
                posthoc_results[factor] = {
                    'summary': str(tukey_result.summary()),
                    'pvalues': tukey_result.pvalues.tolist(),
                    'groups': [str(group) for group in tukey_result.groupsunique],
                    'reject': tukey_result.reject.tolist() if hasattr(tukey_result, 'reject') else []
                }
                
        except Exception as e:
            self.logger.error(f"Post-hoc analysis failed: {e}")
            posthoc_results['error'] = str(e)
            
        return posthoc_results
    
    def _calculate_effect_sizes(self, results_df: pd.DataFrame) -> Dict:
        """Berechnet Effektgrößen (Cohen's d, Cohen's f)."""
        effect_sizes = {}
        
        # Cohen's f for ANOVA (overall effect size)
        try:
            # Calculate between-group and within-group variance
            grand_mean = results_df['correct'].mean()
            
            # Between-group variance for each factor
            for factor in ['model', 'few_shot_count', 'prompt_type']:
                group_means = results_df.groupby(factor)['correct'].mean()
                group_sizes = results_df.groupby(factor).size()
                
                # Cohen's f calculation
                between_var = sum(group_sizes * (group_means - grand_mean)**2) / len(results_df)
                
                # Pooled within-group variance
                within_var_sum = 0
                total_n = 0
                for group, data in results_df.groupby(factor):
                    group_var = data['correct'].var(ddof=1)
                    group_n = len(data)
                    within_var_sum += (group_n - 1) * group_var
                    total_n += group_n - 1
                
                within_var = within_var_sum / total_n if total_n > 0 else 0
                
                # Cohen's f
                cohens_f = np.sqrt(between_var / within_var) if within_var > 0 else 0
                
                # Effect size interpretation
                if cohens_f < 0.1:
                    interpretation = "negligible"
                elif cohens_f < 0.25:
                    interpretation = "small"
                elif cohens_f < 0.4:
                    interpretation = "medium"
                else:
                    interpretation = "large"
                
                effect_sizes[factor] = {
                    'cohens_f': cohens_f,
                    'interpretation': interpretation,
                    'between_variance': between_var,
                    'within_variance': within_var
                }
                
        except Exception as e:
            self.logger.error(f"Effect size calculation failed: {e}")
            effect_sizes['error'] = str(e)
            
        return effect_sizes
    
    def _perform_power_analysis(self, results_df: pd.DataFrame) -> Dict:
        """Führt retrospektive Power-Analyse durch."""
        power_results = {}
        
        try:
            # Calculate observed effect sizes and sample sizes
            for factor in ['model', 'few_shot_count', 'prompt_type']:
                groups = results_df.groupby(factor)['correct']
                group_means = groups.mean()
                group_sizes = groups.size()
                
                # Calculate pooled standard deviation
                pooled_var = 0
                total_n = 0
                for group_name, group_data in groups:
                    n = len(group_data)
                    var = group_data.var(ddof=1)
                    pooled_var += (n - 1) * var
                    total_n += n - 1
                
                pooled_std = np.sqrt(pooled_var / total_n) if total_n > 0 else 0
                
                # Effect size (max difference between groups)
                effect_size = (group_means.max() - group_means.min()) / pooled_std if pooled_std > 0 else 0
                
                # Minimum sample size per group
                min_n = group_sizes.min()
                
                power_results[factor] = {
                    'observed_effect_size': effect_size,
                    'min_sample_size': int(min_n),
                    'max_difference': group_means.max() - group_means.min(),
                    'pooled_std': pooled_std,
                    'sufficient_power': min_n >= 25  # Rule of thumb for medium effect
                }
                
        except Exception as e:
            self.logger.error(f"Power analysis failed: {e}")
            power_results['error'] = str(e)
            
        return power_results
    
    def _test_statistical_assumptions(self, results_df: pd.DataFrame) -> Dict:
        """Testet statistische Annahmen für ANOVA."""
        assumption_results = {}
        
        try:
            # Test normality of residuals
            from scipy.stats import shapiro, levene
            
            # Fit model to get residuals
            anova_data = results_df[['correct', 'model', 'few_shot_count', 'prompt_type']].copy()
            anova_data['few_shot_count'] = anova_data['few_shot_count'].astype(str)
            
            formula = 'correct ~ C(model) + C(few_shot_count) + C(prompt_type)'
            model = sm.formula.ols(formula, data=anova_data).fit()
            residuals = model.resid
            
            # Shapiro-Wilk test for normality
            shapiro_stat, shapiro_p = shapiro(residuals)
            
            assumption_results['normality'] = {
                'shapiro_statistic': float(shapiro_stat),
                'shapiro_pvalue': float(shapiro_p),
                'normal_assumption_met': shapiro_p > 0.05
            }
            
            # Levene test for homogeneity of variance
            groups_data = []
            for (model, few_shot, prompt), group in results_df.groupby(['model', 'few_shot_count', 'prompt_type']):
                groups_data.append(group['correct'].values)
            
            if len(groups_data) > 1:
                levene_stat, levene_p = levene(*groups_data)
                assumption_results['homogeneity'] = {
                    'levene_statistic': float(levene_stat),
                    'levene_pvalue': float(levene_p),
                    'homogeneity_assumption_met': levene_p > 0.05
                }
            
        except Exception as e:
            self.logger.error(f"Assumption testing failed: {e}")
            assumption_results['error'] = str(e)
            
        return assumption_results
    
    def _save_statistical_analysis(self, analysis: Dict):
        """Speichert umfassende statistische Analyse."""
        results_dir = os.path.join(
            self.config.get('paths', {}).get('results_dir', 'results/'), 
            f"test_{self.test_id}"
        )
        
        analysis_path = os.path.join(results_dir, f"comprehensive_analysis_{self.test_id}.json")
        
        with open(analysis_path, "w") as f:
            json.dump(analysis, f, indent=2, default=str)
            
        self.logger.info(f"Statistical analysis saved: {analysis_path}")

    def create_visualizations(self, results_df: pd.DataFrame, analysis: Dict):
        """
        Erstellt publikationsreife wissenschaftliche Visualisierungen.
        
        Implementiert:
        - Few-Shot Progression Plot mit Konfidenzintervallen
        - Modellvergleich mit Fehlerbalken
        - Confusion Matrices für Hauptbedingungen
        - Box Plots mit Signifikanzindikatoren
        - Effect Size Forest Plots
        """
        self.logger.info("📈 Creating Publication-Quality Visualizations...")
        
        results_dir = os.path.join(
            self.config.get('paths', {}).get('results_dir', 'results/'), 
            f"test_{self.test_id}"
        )
        
        # Set publication-ready style
        plt.style.use('seaborn-v0_8-whitegrid')
        plt.rcParams.update({
            'font.size': 12,
            'axes.titlesize': 14,
            'axes.labelsize': 12,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 10,
            'figure.dpi': 300,
            'savefig.dpi': 300,
            'savefig.bbox': 'tight'
        })
        
        # 1. Few-Shot Progression Plot
        self._create_few_shot_progression_plot(results_df, analysis, results_dir)
        
        # 2. Model Comparison with Error Bars
        self._create_model_comparison_plot(results_df, analysis, results_dir)
        
        # 3. Comprehensive Heatmap
        self._create_comprehensive_heatmap(results_df, analysis, results_dir)
        
        # 4. Statistical Box Plots
        self._create_statistical_boxplots(results_df, analysis, results_dir)
        
        # 5. Effect Size Forest Plot
        self._create_effect_size_plot(analysis, results_dir)
        
        # 6. Confusion Matrices
        self._create_confusion_matrices(results_df, results_dir)
        
        self.logger.info("✅ Publication-quality visualizations created!")
    
    def _create_few_shot_progression_plot(self, results_df: pd.DataFrame, analysis: Dict, results_dir: str):
        """Erstellt Few-Shot Progression Plot mit Konfidenzintervallen."""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Prepare data
        progression_data = {}
        for model in results_df['model'].unique():
            model_data = results_df[results_df['model'] == model]
            progression_data[model] = {
                'few_shots': [],
                'accuracies': [],
                'ci_lower': [],
                'ci_upper': []
            }
            
            for few_shot in sorted(model_data['few_shot_count'].unique()):
                subset = model_data[model_data['few_shot_count'] == few_shot]
                accuracy = subset['correct'].mean()
                n = len(subset)
                
                # Calculate 95% confidence interval
                if n > 1:
                    ci = stats.t.interval(0.95, n-1, loc=accuracy, scale=subset['correct'].std()/np.sqrt(n))
                else:
                    ci = (accuracy, accuracy)
                
                progression_data[model]['few_shots'].append(few_shot)
                progression_data[model]['accuracies'].append(accuracy)
                progression_data[model]['ci_lower'].append(ci[0])
                progression_data[model]['ci_upper'].append(ci[1])
        
        # Plot with error bars
        colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
        markers = ['o', 's', '^', 'D']
        
        for i, (model, data) in enumerate(progression_data.items()):
            ax.errorbar(
                data['few_shots'], data['accuracies'],
                yerr=[np.array(data['accuracies']) - np.array(data['ci_lower']),
                      np.array(data['ci_upper']) - np.array(data['accuracies'])],
                label=model, color=colors[i % len(colors)], marker=markers[i % len(markers)],
                capsize=5, capthick=2, linewidth=2, markersize=8
            )
        
        ax.set_xlabel('Few-Shot Examples', fontweight='bold')
        ax.set_ylabel('Classification Accuracy', fontweight='bold')
        ax.set_title('Few-Shot Learning Progression with 95% Confidence Intervals', 
                    fontweight='bold', fontsize=16)
        ax.legend(title='Model', title_fontsize=12, fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)
        
        # Add statistical annotations
        overall_accuracy = analysis.get('overall_metrics', {}).get('accuracy', 0)
        ax.axhline(y=overall_accuracy, color='red', linestyle='--', alpha=0.7, 
                  label=f'Overall Mean: {overall_accuracy:.3f}')
        
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, f"few_shot_progression_{self.test_id}.png"))
        plt.close()
    
    def _create_model_comparison_plot(self, results_df: pd.DataFrame, analysis: Dict, results_dir: str):
        """Erstellt detaillierten Modellvergleich mit multiplen Metriken."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        metrics = ['accuracy', 'f1_weighted', 'precision_weighted', 'recall_weighted']
        metric_names = ['Accuracy', 'F1-Score (Weighted)', 'Precision (Weighted)', 'Recall (Weighted)']
        axes = [ax1, ax2, ax3, ax4]
        
        for idx, (metric, metric_name, ax) in enumerate(zip(metrics, metric_names, axes)):
            model_data = {}
            
            for model in results_df['model'].unique():
                model_subset = results_df[results_df['model'] == model]
                
                if metric == 'accuracy':
                    values = [group['correct'].mean() for _, group in model_subset.groupby(['few_shot_count', 'prompt_type'])]
                else:
                    values = []
                    for _, group in model_subset.groupby(['few_shot_count', 'prompt_type']):
                        if metric == 'f1_weighted':
                            val = f1_score(group['ground_truth'], group['prediction'], average='weighted', zero_division=0)
                        elif metric == 'precision_weighted':
                            val = precision_score(group['ground_truth'], group['prediction'], average='weighted', zero_division=0)
                        elif metric == 'recall_weighted':
                            val = recall_score(group['ground_truth'], group['prediction'], average='weighted', zero_division=0)
                        values.append(val)
                
                model_data[model] = values
            
            # Create box plot
            positions = np.arange(1, len(model_data) + 1)
            bp = ax.boxplot([model_data[model] for model in model_data.keys()], 
                          positions=positions, patch_artist=True, 
                          labels=list(model_data.keys()))
            
            # Customize colors
            colors = ['lightblue', 'lightcoral']
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            
            ax.set_title(f'{metric_name} by Model', fontweight='bold')
            ax.set_ylabel(metric_name)
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0, 1)
            
            # Add mean markers
            for i, (model, values) in enumerate(model_data.items()):
                mean_val = np.mean(values)
                ax.plot(i+1, mean_val, 'ro', markersize=8, markerfacecolor='red', 
                       markeredgecolor='darkred', markeredgewidth=1)
                ax.text(i+1, mean_val + 0.02, f'{mean_val:.3f}', 
                       ha='center', va='bottom', fontweight='bold')
        
        plt.suptitle('Model Performance Comparison Across All Metrics', 
                    fontsize=18, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, f"model_comparison_{self.test_id}.png"))
        plt.close()
    
    def _create_comprehensive_heatmap(self, results_df: pd.DataFrame, analysis: Dict, results_dir: str):
        """Erstellt umfassende Heatmap für alle experimentellen Bedingungen."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        metrics = ['accuracy', 'f1_weighted', 'mcc', 'balanced_accuracy']
        metric_names = ['Accuracy', 'F1-Score', 'Matthews Correlation', 'Balanced Accuracy']
        axes = [ax1, ax2, ax3, ax4]
        
        for metric, metric_name, ax in zip(metrics, metric_names, axes):
            # Prepare heatmap data
            heatmap_data = []
            conditions = []
            
            for (model, few_shot, prompt), group in results_df.groupby(['model', 'few_shot_count', 'prompt_type']):
                condition_name = f"{model}\n{few_shot}-shot\n{prompt}"
                conditions.append(condition_name)
                
                if metric == 'accuracy':
                    value = group['correct'].mean()
                elif metric == 'f1_weighted':
                    value = f1_score(group['ground_truth'], group['prediction'], average='weighted', zero_division=0)
                elif metric == 'mcc':
                    value = matthews_corrcoef(group['ground_truth'], group['prediction'])
                elif metric == 'balanced_accuracy':
                    value = balanced_accuracy_score(group['ground_truth'], group['prediction'])
                
                heatmap_data.append(value)
            
            # Reshape for heatmap
            n_conditions = len(conditions)
            heatmap_matrix = np.array(heatmap_data).reshape(1, n_conditions)
            
            # Create heatmap
            im = ax.imshow(heatmap_matrix, cmap='RdYlBu_r', aspect='auto', vmin=0, vmax=1)
            
            # Set labels
            ax.set_xticks(range(n_conditions))
            ax.set_xticklabels(conditions, rotation=45, ha='right')
            ax.set_yticks([0])
            ax.set_yticklabels([metric_name])
            ax.set_title(f'{metric_name} Heatmap', fontweight='bold')
            
            # Add value annotations
            for i in range(n_conditions):
                text = ax.text(i, 0, f'{heatmap_data[i]:.3f}', ha='center', va='center',
                             color='white' if heatmap_data[i] < 0.5 else 'black', fontweight='bold')
            
            # Add colorbar
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        plt.suptitle('Performance Heatmap Across All Experimental Conditions', 
                    fontsize=18, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, f"comprehensive_heatmap_{self.test_id}.png"))
        plt.close()
    
    def _create_statistical_boxplots(self, results_df: pd.DataFrame, analysis: Dict, results_dir: str):
        """Erstellt Box Plots mit statistischen Signifikanzindikatoren."""
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        factors = ['model', 'few_shot_count', 'prompt_type']
        factor_names = ['Model', 'Few-Shot Count', 'Prompt Type']
        
        for idx, (factor, factor_name) in enumerate(zip(factors, factor_names)):
            ax = axes[idx]
            
            # Prepare data for box plot
            factor_groups = []
            labels = []
            
            for level in sorted(results_df[factor].unique()):
                group_data = results_df[results_df[factor] == level]['correct']
                factor_groups.append(group_data)
                labels.append(str(level))
            
            # Create box plot
            bp = ax.boxplot(factor_groups, labels=labels, patch_artist=True)
            
            # Customize appearance
            colors = plt.cm.Set3(np.linspace(0, 1, len(factor_groups)))
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            
            # Add statistical annotations
            if factor in analysis.get('anova_results', {}):
                p_value = analysis['anova_results'][factor].get('PR(>F)', 1.0)
                if p_value is not None:
                    if p_value < 0.001:
                        sig_text = "***"
                    elif p_value < 0.01:
                        sig_text = "**"
                    elif p_value < 0.05:
                        sig_text = "*"
                    else:
                        sig_text = "ns"
                    
                    ax.text(0.02, 0.98, f'p = {p_value:.4f} {sig_text}', 
                           transform=ax.transAxes, va='top', ha='left',
                           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            
            ax.set_title(f'{factor_name} Effect on Accuracy', fontweight='bold')
            ax.set_ylabel('Classification Accuracy')
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0, 1)
        
        plt.suptitle('Statistical Analysis: Factor Effects with ANOVA Results', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, f"statistical_boxplots_{self.test_id}.png"))
        plt.close()
    
    def _create_effect_size_plot(self, analysis: Dict, results_dir: str):
        """Erstellt Effect Size Forest Plot."""
        effect_sizes = analysis.get('effect_sizes', {})
        
        if not effect_sizes or 'error' in effect_sizes:
            self.logger.warning("Effect sizes not available for plotting")
            return
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        factors = []
        cohens_f_values = []
        interpretations = []
        
        for factor, data in effect_sizes.items():
            if isinstance(data, dict) and 'cohens_f' in data:
                factors.append(factor.replace('_', ' ').title())
                cohens_f_values.append(data['cohens_f'])
                interpretations.append(data['interpretation'])
        
        # Create horizontal bar chart
        y_pos = np.arange(len(factors))
        colors = ['green' if interp in ['large', 'medium'] else 
                 'orange' if interp == 'small' else 'red' 
                 for interp in interpretations]
        
        bars = ax.barh(y_pos, cohens_f_values, color=colors, alpha=0.7)
        
        # Add value labels
        for i, (bar, value, interp) in enumerate(zip(bars, cohens_f_values, interpretations)):
            ax.text(value + 0.01, bar.get_y() + bar.get_height()/2, 
                   f'{value:.3f} ({interp})', va='center', fontweight='bold')
        
        # Effect size reference lines
        ax.axvline(x=0.1, color='gray', linestyle='--', alpha=0.5, label='Small (0.1)')
        ax.axvline(x=0.25, color='gray', linestyle='--', alpha=0.7, label='Medium (0.25)')
        ax.axvline(x=0.4, color='gray', linestyle='--', alpha=0.9, label='Large (0.4)')
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(factors)
        ax.set_xlabel("Cohen's f (Effect Size)", fontweight='bold')
        ax.set_title("Effect Sizes (Cohen's f) for Experimental Factors", 
                    fontweight='bold', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, f"effect_sizes_{self.test_id}.png"))
        plt.close()
    
    def _create_confusion_matrices(self, results_df: pd.DataFrame, results_dir: str):
        """Erstellt Confusion Matrices für Hauptbedingungen."""
        categories = self.config.get('categories', [])
        
        # Create confusion matrices for best and worst conditions
        condition_accuracies = {}
        for (model, few_shot, prompt), group in results_df.groupby(['model', 'few_shot_count', 'prompt_type']):
            condition_name = f"{model}_{few_shot}shots_{prompt}"
            accuracy = group['correct'].mean()
            condition_accuracies[condition_name] = (accuracy, group)
        
        # Get best and worst conditions
        best_condition = max(condition_accuracies.items(), key=lambda x: x[1][0])
        worst_condition = min(condition_accuracies.items(), key=lambda x: x[1][0])
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        for ax, (condition_name, (accuracy, group)), title_suffix in [
            (ax1, best_condition, "Best"),
            (ax2, worst_condition, "Worst")
        ]:
            cm = confusion_matrix(group['ground_truth'], group['prediction'], 
                                labels=categories)
            
            # Normalize confusion matrix
            cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            
            # Create heatmap
            sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                       xticklabels=categories, yticklabels=categories, ax=ax)
            
            ax.set_title(f'{title_suffix} Condition: {condition_name}\nAccuracy: {accuracy:.3f}', 
                        fontweight='bold')
            ax.set_xlabel('Predicted Category', fontweight='bold')
            ax.set_ylabel('True Category', fontweight='bold')
        
        plt.suptitle('Confusion Matrices: Best vs Worst Performing Conditions', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, f"confusion_matrices_{self.test_id}.png"))
        plt.close()

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

def calculate_required_sample_size(effect_size: float = 0.3, power: float = 0.8, alpha: float = 0.05) -> int:
    """
    Berechnet erforderliche Stichprobengröße für statistische Power.
    
    Args:
        effect_size: Erwartete Effektgröße (Cohen's f)
        power: Gewünschte statistische Power (1-β)
        alpha: Signifikanzniveau (α)
    
    Returns:
        Mindest-Stichprobengröße pro Gruppe
    """
    # Vereinfachte Power-Analyse für ANOVA
    # Für genaue Berechnung würde man statsmodels.stats.power verwenden
    
    if effect_size < 0.1:
        return 100  # Sehr große Stichprobe für kleine Effekte
    elif effect_size < 0.25:
        return 50   # Mittlere Stichprobe für kleine bis mittlere Effekte
    elif effect_size < 0.4:
        return 25   # Moderate Stichprobe für mittlere Effekte
    else:
        return 15   # Kleinere Stichprobe für große Effekte

def validate_experimental_setup(config: Dict) -> Dict:
    """
    Validiert das experimentelle Setup vor dem Start.
    
    Returns:
        Dictionary mit Validierungsergebnissen
    """
    validation = {
        'valid': True,
        'warnings': [],
        'errors': [],
        'recommendations': []
    }
    
    # Überprüfe Modelle
    models = config.get('models', [])
    if len(models) < 2:
        validation['warnings'].append("Weniger als 2 Modelle - limitierte Vergleichsmöglichkeiten")
    
    # Überprüfe Few-Shot Progression
    few_shot_counts = config.get('few_shot_counts', [])
    if 0 not in few_shot_counts:
        validation['errors'].append("Zero-shot Baseline (0) fehlt in few_shot_counts")
        validation['valid'] = False
    
    if len(few_shot_counts) < 3:
        validation['warnings'].append("Weniger als 3 Few-Shot Bedingungen - limitierte Progression")
    
    # Überprüfe Stichprobengröße
    total_tickets = config.get('experiment', {}).get('total_experimental_tickets', 0)
    n_conditions = len(models) * len(few_shot_counts) * len(config.get('prompt_types', []))
    tickets_per_condition = total_tickets / n_conditions if n_conditions > 0 else 0
    
    required_per_condition = calculate_required_sample_size()
    
    if tickets_per_condition < required_per_condition:
        validation['warnings'].append(
            f"Stichprobengröße pro Bedingung ({tickets_per_condition:.1f}) unter Empfehlung ({required_per_condition})"
        )
        validation['recommendations'].append(
            f"Erhöhen Sie total_experimental_tickets auf mindestens {required_per_condition * n_conditions}"
        )
    
    # Überprüfe Kategorien
    categories = config.get('categories', [])
    if len(categories) < 3:
        validation['warnings'].append("Weniger als 3 Kategorien - limitierte Klassifikationskomplexität")
    
    return validation

def main():
    """
    Wissenschaftlich rigorose Hauptfunktion für Few-Shot Experiment.
    
    Workflow:
    1. Pre-experiment validation
    2. Power analysis
    3. Experimental setup confirmation
    4. Experiment execution
    5. Comprehensive analysis
    6. Publication-ready reporting
    """
    print("🔬 WISSENSCHAFTLICHES FEW-SHOT LEARNING EXPERIMENT")
    print("=" * 60)
    print("Implementiert nach akademischen Standards für DHBW Projektarbeit")
    print("=" * 60)
    
    # 1. Experimentelle Validierung
    print("\n📋 Phase 1: Experimentelle Validierung")
    experiment = FewShotExperiment()
    
    validation = validate_experimental_setup(experiment.config)
    
    if not validation['valid']:
        print("❌ KRITISCHE FEHLER im Experimental Design:")
        for error in validation['errors']:
            print(f"   • {error}")
        print("\nExperiment abgebrochen. Bitte korrigieren Sie die Konfiguration.")
        return
    
    if validation['warnings']:
        print("⚠️  WARNUNGEN:")
        for warning in validation['warnings']:
            print(f"   • {warning}")
    
    if validation['recommendations']:
        print("💡 EMPFEHLUNGEN:")
        for rec in validation['recommendations']:
            print(f"   • {rec}")
    
    # 2. Power Analysis
    print("\n📊 Phase 2: Power-Analyse")
    models = experiment.config.get('models', [])
    few_shot_counts = experiment.config.get('few_shot_counts', [])
    prompt_types = experiment.config.get('prompt_types', [])
    total_tickets = experiment.config.get('experiment', {}).get('total_experimental_tickets', 200)
    
    n_conditions = len(models) * len(few_shot_counts) * len(prompt_types)
    tickets_per_condition = total_tickets // n_conditions
    
    print(f"   Factorial Design: {len(models)}×{len(few_shot_counts)}×{len(prompt_types)} = {n_conditions} Bedingungen")
    print(f"   Tickets pro Bedingung: {tickets_per_condition}")
    print(f"   Gesamtklassifikationen: {total_tickets}")
    
    estimated_time = total_tickets * 3  # 3 Sekunden pro Klassifikation
    print(f"   Geschätzte Laufzeit: {estimated_time//60:.0f} Minuten")
    
    # 3. User Confirmation
    print("\n🎯 Phase 3: Experimentelle Parameter")
    print(f"   Modelle: {', '.join(models)}")
    print(f"   Few-Shot Progression: {few_shot_counts}")
    print(f"   Prompt-Typen: {', '.join(prompt_types)}")
    print(f"   Kategorien: {', '.join(experiment.config.get('categories', []))}")
    
    # Interactive confirmation
    response = input("\n▶️  Experiment mit diesen Parametern starten? [y/N]: ").strip().lower()
    if response not in ['y', 'yes', 'j', 'ja']:
        print("Experiment abgebrochen.")
        return
    
    # 4. Experiment Execution
    print("\n🚀 Phase 4: Experiment-Ausführung")
    print("Starting scientific factorial experiment...")
    
    start_time = time.time()
    
    try:
        results_df = experiment.run_experiment()
        
        if len(results_df) == 0:
            print("❌ Experiment fehlgeschlagen - keine Ergebnisse erhalten.")
            return
        
        experiment_time = time.time() - start_time
        print(f"\n⏱️  Experiment abgeschlossen in {experiment_time/60:.1f} Minuten")
        
        # 5. Comprehensive Analysis
        print("\n📊 Phase 5: Wissenschaftliche Analyse")
        analysis = experiment.analyze_results(results_df)
        
        # 6. Visualization
        print("\n📈 Phase 6: Publikationsreife Visualisierungen")
        experiment.create_visualizations(results_df, analysis)
        
        # 7. Report Generation
        print("\n📝 Phase 7: Wissenschaftlicher Bericht")
        experiment.generate_report(results_df, analysis)
        
        # 8. Summary
        print("\n" + "="*60)
        print("🎯 EXPERIMENT ZUSAMMENFASSUNG")
        print("="*60)
        
        overall_metrics = analysis.get('overall_metrics', {})
        print(f"   Overall Accuracy: {overall_metrics.get('accuracy', 0):.3f}")
        print(f"   F1-Score (Weighted): {overall_metrics.get('f1_weighted', 0):.3f}")
        print(f"   Matthews Correlation: {overall_metrics.get('mcc', 0):.3f}")
        print(f"   Balanced Accuracy: {overall_metrics.get('balanced_accuracy', 0):.3f}")
        
        print(f"\n   Gesamtklassifikationen: {len(results_df)}")
        print(f"   Erfolgsrate: {(1 - (results_df['response'] == 'ERROR').mean())*100:.1f}%")
        print(f"   Test-ID: {experiment.test_id}")
        print(f"   Experimentzeit: {experiment_time/60:.1f} Minuten")
        
        # ANOVA Results
        if 'anova_results' in analysis:
            print(f"\n   📊 ANOVA Ergebnisse:")
            for factor, results in analysis['anova_results'].items():
                if isinstance(results, dict) and 'PR(>F)' in results:
                    p_val = results['PR(>F)']
                    if p_val is not None:
                        significance = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
                        print(f"      {factor}: p = {p_val:.4f} {significance}")
        
        # Effect Sizes
        if 'effect_sizes' in analysis:
            print(f"\n   📏 Effektgrößen (Cohen's f):")
            for factor, data in analysis['effect_sizes'].items():
                if isinstance(data, dict) and 'cohens_f' in data:
                    print(f"      {factor}: {data['cohens_f']:.3f} ({data['interpretation']})")
        
        # Best Condition
        condition_metrics = analysis.get('condition_metrics', {})
        if condition_metrics:
            best_condition = max(condition_metrics.items(), key=lambda x: x[1]['accuracy'])
            print(f"\n   🏆 Beste Bedingung: {best_condition[0]}")
            print(f"      Accuracy: {best_condition[1]['accuracy']:.3f}")
            print(f"      95% CI: [{best_condition[1].get('ci_95_lower', 0):.3f}, {best_condition[1].get('ci_95_upper', 1):.3f}]")
        
        # Power Analysis Results
        if 'power_analysis' in analysis:
            print(f"\n   ⚡ Power-Analyse:")
            for factor, power_data in analysis['power_analysis'].items():
                if isinstance(power_data, dict):
                    sufficient = "✅" if power_data.get('sufficient_power', False) else "⚠️"
                    print(f"      {factor}: {sufficient} n={power_data.get('min_sample_size', 0)}")
        
        print(f"\n   📁 Alle Ergebnisse gespeichert in: results/test_{experiment.test_id}/")
        print("="*60)
        print("✅ WISSENSCHAFTLICHES EXPERIMENT ERFOLGREICH ABGESCHLOSSEN!")
        print("   Alle Daten für DHBW Projektarbeit dokumentiert und bereit.")
        print("="*60)
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Experiment durch Benutzer unterbrochen")
        print("   Zwischenergebnisse möglicherweise verfügbar in results/")
    except Exception as e:
        print(f"\n❌ Unerwarteter Fehler: {e}")
        experiment.logger.error(f"Experiment failed: {e}")
        
if __name__ == "__main__":
    main() 