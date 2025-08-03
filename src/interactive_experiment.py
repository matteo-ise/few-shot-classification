#!/usr/bin/env python3
"""
Vereinfachtes Experiment mit interaktivem Menü
Für schnelle Tests ohne volle wissenschaftliche Rigorosität
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from few_shot_experiment import FewShotExperiment, interactive_menu

def main_interactive():
    """Hauptfunktion mit dem alten interaktiven Menü"""
    print("🔬 Few-Shot Learning Experiment (Interaktive Version)")
    print("=" * 60)
    
    experiment = FewShotExperiment()
    
    # Das alte interaktive Menü
    models, few_shot_counts, prompt_types, n_tickets = interactive_menu(experiment.config)
    
    print(f"\n🚀 Starte Experiment mit:")
    print(f"   Modelle: {models}")
    print(f"   Few-Shot: {few_shot_counts}")
    print(f"   Prompts: {prompt_types}")
    print(f"   Tickets: {n_tickets}")
    
    # Vereinfachtes Experiment
    results_df = experiment.run_experiment(
        models=models, 
        few_shot_counts=few_shot_counts, 
        prompt_types=prompt_types, 
        n_tickets=n_tickets
    )
    
    if len(results_df) > 0:
        analysis = experiment.analyze_results(results_df)
        experiment.create_visualizations(results_df, analysis)
        experiment.generate_report(results_df, analysis)
        
        print(f"\n✅ Experiment abgeschlossen!")
        print(f"   Test-ID: {experiment.test_id}")
        print(f"   Ergebnisse: results/test_{experiment.test_id}/")

if __name__ == "__main__":
    main_interactive()
