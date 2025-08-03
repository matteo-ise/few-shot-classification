#!/usr/bin/env python3
"""
Debug-Script für Report-Generierung ohne vollständiges Experiment.

Generiert alle LaTeX-Reports mit synthetischen Daten um:
1. Report-Funktionen zu testen
2. LaTeX-Output zu validieren
3. Keine Ressourcen zu verschwenden
"""

import pandas as pd
import numpy as np
import os
import sys
import json
from datetime import datetime
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from few_shot_experiment import FewShotExperiment

def create_synthetic_results() -> pd.DataFrame:
    """
    Erstellt synthetische Experiment-Ergebnisse für Tests.
    Simuliert realistische Daten ohne Experiment-Overhead.
    """
    print("🔧 Generiere synthetische Experimentdaten...")
    
    # Experimentparameter
    models = ['llama3.1:8b', 'mistral:7b']
    few_shot_counts = [0, 1, 3, 5]
    prompt_types = ['structured', 'unstructured']
    categories = ['Hardware', 'Software', 'Network', 'Security']
    
    # Realistische Performance-Trends
    base_accuracy = {
        'llama3.1:8b': {'structured': 0.75, 'unstructured': 0.68},
        'mistral:7b': {'structured': 0.72, 'unstructured': 0.65}
    }
    
    # Few-Shot Verbesserung (realistisch)
    few_shot_boost = {0: 0.0, 1: 0.05, 3: 0.12, 5: 0.18}
    
    results = []
    ticket_id = 1
    
    # Generiere für jede Bedingung
    for model in models:
        for few_shot in few_shot_counts:
            for prompt in prompt_types:
                for category in categories:
                    # 20 Tickets pro Bedingung/Kategorie (realistisch)
                    n_tickets = 20
                    
                    # Berechne erwartete Accuracy für diese Bedingung
                    base_acc = base_accuracy[model][prompt]
                    boost = few_shot_boost[few_shot]
                    expected_acc = min(0.95, base_acc + boost)  # Cap bei 95%
                    
                    # Kategorien-spezifische Anpassungen (realistisch)
                    category_modifier = {
                        'Hardware': 0.05,    # Einfacher
                        'Software': -0.02,   # Schwieriger
                        'Network': -0.03,    # Schwieriger
                        'Security': 0.02     # Mittel
                    }
                    expected_acc += category_modifier.get(category, 0)
                    expected_acc = max(0.2, min(0.95, expected_acc))  # Bounds
                    
                    # Generiere Tickets mit realistischer Varianz
                    for _ in range(n_tickets):
                        # Binomial sampling für realistische Verteilung
                        correct = np.random.random() < expected_acc
                        
                        # Simuliere realistische Prediction-Fehler
                        if correct:
                            prediction = category
                        else:
                            # Fehler meist zu ähnlichen Kategorien
                            other_categories = [c for c in categories if c != category]
                            prediction = np.random.choice(other_categories)
                        
                        results.append({
                            'ticket_id': f'ticket_{ticket_id:04d}',
                            'ground_truth': category,
                            'prediction': prediction,
                            'model': model,
                            'few_shot_count': few_shot,
                            'prompt_type': prompt,
                            'response': f'Klassifikation: {prediction}',
                            'correct': correct
                        })
                        ticket_id += 1
    
    df = pd.DataFrame(results)
    print(f"✅ {len(df)} synthetische Datenpunkte generiert")
    print(f"   Bedingungen: {len(df.groupby(['model', 'few_shot_count', 'prompt_type']))}")
    print(f"   Overall Accuracy: {df['correct'].mean():.3f}")
    
    return df

def create_synthetic_analysis(results_df: pd.DataFrame) -> dict:
    """
    Erstellt synthetische Analyse-Ergebnisse.
    Simuliert realistische statistische Resultate.
    """
    print("📊 Generiere synthetische Analyse...")
    
    analysis = {
        'experiment_metadata': {
            'test_id': f'debug_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
            'total_classifications': len(results_df),
            'timestamp': datetime.now().isoformat(),
            'random_seed': 42
        },
        'overall_metrics': {
            'accuracy': results_df['correct'].mean(),
            'f1_weighted': results_df['correct'].mean() * 0.95,  # Realistisch
            'f1_macro': results_df['correct'].mean() * 0.93,
            'precision_weighted': results_df['correct'].mean() * 0.96,
            'recall_weighted': results_df['correct'].mean() * 0.94,
            'mcc': (results_df['correct'].mean() - 0.25) * 2,  # Realistic MCC
            'balanced_accuracy': results_df['correct'].mean() * 0.98
        },
        'condition_metrics': {},
        'anova_results': {
            'few_shot_count': {
                'F': 15.47,
                'PR(>F)': 0.0001,
                'df': 3,
                'eta_squared': 0.234
            },
            'model': {
                'F': 8.23,
                'PR(>F)': 0.0045,
                'df': 1,
                'eta_squared': 0.089
            },
            'prompt_type': {
                'F': 12.91,
                'PR(>F)': 0.0003,
                'df': 1,
                'eta_squared': 0.156
            }
        },
        'effect_sizes': {
            'few_shot_count': {
                'cohens_f': 0.354,
                'interpretation': 'medium'
            },
            'model': {
                'cohens_f': 0.189,
                'interpretation': 'small'
            },
            'prompt_type': {
                'cohens_f': 0.298,
                'interpretation': 'medium'
            }
        }
    }
    
    # Condition metrics berechnen
    for (model, few_shot, prompt), group in results_df.groupby(['model', 'few_shot_count', 'prompt_type']):
        condition_name = f"{model}_{few_shot}shots_{prompt}"
        accuracy = group['correct'].mean()
        n = len(group)
        
        # Realistic confidence intervals
        std_err = np.sqrt(accuracy * (1 - accuracy) / n)
        ci_95_lower = max(0, accuracy - 1.96 * std_err)
        ci_95_upper = min(1, accuracy + 1.96 * std_err)
        
        analysis['condition_metrics'][condition_name] = {
            'accuracy': accuracy,
            'f1_score': accuracy * 0.95,
            'precision': accuracy * 0.96,
            'recall': accuracy * 0.94,
            'mcc': (accuracy - 0.25) * 2,
            'balanced_accuracy': accuracy * 0.98,
            'n_samples': n,
            'ci_95_lower': ci_95_lower,
            'ci_95_upper': ci_95_upper
        }
    
    print("✅ Synthetische Analyse generiert")
    return analysis

def main():
    """Debug-Hauptfunktion für Report-Tests."""
    print("🔬 DEBUG: LaTeX Report Generation Test")
    print("=" * 50)
    print("Testet alle Report-Funktionen mit synthetischen Daten")
    print("=" * 50)
    
    try:
        # 1. Initialisiere Experiment-Klasse
        print("\n📋 Phase 1: Experiment Setup")
        experiment = FewShotExperiment()
        
        # Override test_id für Debug
        experiment.test_id = f'debug_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
        print(f"   Debug Test-ID: {experiment.test_id}")
        
        # 2. Generiere synthetische Daten
        print("\n🔧 Phase 2: Synthetische Daten")
        results_df = create_synthetic_results()
        
        # 3. Generiere synthetische Analyse
        print("\n📊 Phase 3: Synthetische Analyse")
        analysis = create_synthetic_analysis(results_df)
        
        # 3.5. Erstelle Ergebnis-Verzeichnis
        results_dir = os.path.join('results', f'test_{experiment.test_id}')
        os.makedirs(results_dir, exist_ok=True)
        print(f"   Ergebnis-Verzeichnis erstellt: {results_dir}")
        
        # 4. Überspringe Visualisierungen (Problembereich)
        print("\n📈 Phase 4: Visualisierungen übersprungen (Debug-Fokus auf Reports)")
        
        # 5. Teste Report-Generierung
        print("\n📝 Phase 5: Test LaTeX Reports")
        experiment.generate_report(results_df, analysis)
        
        # 6. Validiere Output
        print("\n✅ Phase 6: Output-Validierung")
        results_dir = os.path.join('results', f'test_{experiment.test_id}')
        
        expected_files = [
            f'latex_substitutions_{experiment.test_id}.tex',
            f'tikz_data_{experiment.test_id}.tex',
            f'category_analysis_{experiment.test_id}.json',
            f'report_{experiment.test_id}.md'
        ]
        
        print(f"   Ergebnis-Ordner: {results_dir}")
        for filename in expected_files:
            filepath = os.path.join(results_dir, filename)
            if os.path.exists(filepath):
                size = os.path.getsize(filepath)
                print(f"   ✅ {filename} ({size} bytes)")
            else:
                print(f"   ❌ {filename} FEHLT")
        
        # 7. Zeige LaTeX-Substitutionen Preview
        subs_path = os.path.join(results_dir, f'latex_substitutions_{experiment.test_id}.tex')
        if os.path.exists(subs_path):
            print(f"\n📄 LaTeX-Substitutionen Preview:")
            with open(subs_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()[:10]  # Erste 10 Zeilen
                for line in lines:
                    print(f"   {line.strip()}")
                if len(f.readlines()) > 10:
                    print("   ...")
        
        print("\n" + "=" * 50)
        print("🎯 DEBUG ERFOLGREICH ABGESCHLOSSEN!")
        print(f"   Alle Reports generiert in: {results_dir}")
        print("   Keine Ressourcen verschwendet ✅")
        print("   LaTeX-Integration bereit für chapter4.tex ✅")
        print("=" * 50)
        
    except Exception as e:
        print(f"\n❌ DEBUG FEHLER: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
