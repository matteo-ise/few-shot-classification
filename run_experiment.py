#!/usr/bin/env python3
"""
🚀 EINFACHER STARTUP SCRIPT FÜR FEW-SHOT EXPERIMENT
=================================================

Einfach dieses Skript ausführen um das komplette Experiment zu starten.
Alle Ergebnisse werden automatisch gespeichert.

VERWENDUNG:
-----------
1. F5 drücken in VS Code
2. Oder: python run_experiment.py

EXPERIMENT ÜBERSICHT:
--------------------
- 2×4×2 faktorielles Design (32 Bedingungen)
- ~3200 Klassifikationen total
- Laufzeit: 2-3 Stunden
- Automatische Analyse & Visualisierungen
"""

import sys
import os
from datetime import datetime

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def main():
    print("🚀 Starting Few-Shot Learning Experiment...")
    print("=" * 60)
    print(f"⏰ Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("📊 Experimental Design: 2×4×2 Factorial")
    print("🤖 Models: Llama3.1:8b, Mistral:7b")
    print("🎯 Few-Shot: 0, 1, 3, 5 examples")
    print("💬 Prompts: Structured, Unstructured")
    print("⏱️  Expected Duration: 2-3 hours")
    print("=" * 60)
    
    try:
        # Import and run experiment
        from few_shot_experiment import FewShotExperiment
        
        # Create experiment instance
        experiment = FewShotExperiment()
        
        # Run complete experiment
        print("\n🔬 Running Scientific Few-Shot Experiment...")
        results_df = experiment.run_experiment()
        
        if len(results_df) > 0:
            print("\n📊 Performing Statistical Analysis...")
            analysis = experiment.analyze_results(results_df)
            
            print("\n📈 Creating Visualizations...")
            experiment.create_visualizations(results_df, analysis)
            
            print("\n📝 Generating Report...")
            experiment.generate_report(results_df, analysis)
            
            print("\n" + "=" * 60)
            print("✅ EXPERIMENT COMPLETE!")
            print(f"⏰ End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"📁 Results saved in: results/test_{experiment.test_id}/")
            print("📊 Check the generated report and visualizations!")
            print("=" * 60)
        else:
            print("❌ Experiment failed - no results generated")
            
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        print("💡 Mögliche Lösungen:")
        print("   - Überprüfen Sie dass Ollama läuft")
        print("   - Überprüfen Sie config.yaml")
        print("   - Überprüfen Sie data/tickets_examples.xlsx")
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
