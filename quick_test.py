#!/usr/bin/env python3
"""
Quick Test Tool für isolierte Few-Shot Learning Tests
"""

import pandas as pd
import numpy as np
import requests
import json
import time
import logging
from sklearn.metrics import accuracy_score, f1_score
from typing import List, Dict, Optional
import os

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)

class QuickTest:
    def __init__(self):
        self.base_url = "http://localhost:11434/api/generate"
        self.categories = ["Hardware", "Software", "Network", "Security"]
        
        # Load data
        self.tickets_df = pd.read_excel("data/tickets_200.xlsx")
        self.examples_df = pd.read_excel("data/tickets_examples.xlsx")
        
        # Combine subject and body for ticket text
        self.tickets_df['ticket_text'] = self.tickets_df['subject'] + " " + self.tickets_df['body']
        self.examples_df['ticket_text'] = self.examples_df['subject'] + " " + self.examples_df['body']
    
    def _create_prompt(self, ticket_text: str, few_shot_examples: List[Dict], prompt_type: str) -> str:
        """Erstellt Prompt basierend auf Typ und Few-Shot Beispielen."""
        categories_str = ", ".join(self.categories)
        
        if prompt_type == "structured":
            prompt = f"""Klassifizieren Sie das folgende IT-Support-Ticket in eine der Kategorien: {categories_str}

Beispiele:
"""
            for example in few_shot_examples:
                prompt += f"Ticket: {example['ticket_text']}\nKategorie: {example['category']}\n\n"
            
            prompt += f"""Ticket: {ticket_text}
Kategorie: 

WICHTIG: Geben Sie NUR das Wort der Kategorie an (Hardware, Software, Network oder Security). Keine Sätze, keine Erklärungen, keine Satzzeichen."""
        else:
            prompt = f"""Hier sind einige Beispiele für IT-Support-Ticket-Klassifizierung:

"""
            for example in few_shot_examples:
                prompt += f"Ticket: {example['ticket_text']} → {example['category']}\n"
            
            prompt += f"""
Klassifiziere dieses Ticket: {ticket_text}

WICHTIG: Antworte nur mit einem Wort (Hardware, Software, Network oder Security). Keine Sätze!"""
        
        return prompt
    
    def _query_llm(self, prompt: str, model: str) -> Optional[str]:
        """Sendet Anfrage an Ollama API."""
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.1,
                "top_p": 0.9,
                "num_predict": 10,
                "timeout": 30
            }
        }
        
        try:
            response = requests.post(self.base_url, json=payload, timeout=30)
            response.raise_for_status()
            return response.json()['response'].strip()
        except Exception as e:
            logger.error(f"API error: {e}")
            return None
    
    def _extract_category(self, response: str) -> str:
        """Extrahiert Kategorie aus LLM-Antwort."""
        response_lower = response.lower().strip()
        
        for category in self.categories:
            if category.lower() in response_lower:
                return category
        
        # Fallback: return first category if no match
        return self.categories[0]
    
    def _get_few_shot_examples(self, count: int) -> List[Dict]:
        """Holt Few-Shot Beispiele aus dem Examples Dataset - WISSENSCHAFTLICH KORREKT."""
        if count == 0:
            return []
        
        examples = []
        categories = self.categories
        
        # Wähle pro Kategorie die Beispiele aus (wie im Hauptprogramm)
        for category in categories:
            category_examples = self.examples_df[self.examples_df['label'] == category]
            
            if len(category_examples) >= count:
                # Wähle die ersten 'count' Beispiele dieser Kategorie
                selected = category_examples.head(count)
            else:
                # Falls weniger verfügbar, nimm alle
                selected = category_examples
                logger.warning(f"Nur {len(category_examples)} Beispiele für {category} verfügbar, benötigt: {count}")
            
            for _, row in selected.iterrows():
                examples.append({
                    'ticket_text': row['ticket_text'],
                    'category': row['label']
                })
        
        logger.info(f"Few-Shot: {count} Beispiele pro Kategorie = {len(examples)} total")
        return examples
    
    def run_single_test(self, model: str, few_shot_count: int, prompt_type: str, n_tickets: int = 10) -> Dict:
        """Führt einen einzelnen Test durch."""
        logger.info(f"🚀 Quick Test: {model}, {few_shot_count}-shot, {prompt_type}, {n_tickets} tickets")
        
        results = []
        correct_count = 0
        
        # Get few-shot examples
        few_shot_examples = self._get_few_shot_examples(few_shot_count)
        
        # Debug: Show few-shot examples
        if few_shot_count > 0:
            logger.info(f"Few-shot examples ({few_shot_count}):")
            for i, example in enumerate(few_shot_examples):
                logger.info(f"  {i+1}. {example['ticket_text'][:50]}... → {example['category']}")
        
        # Process tickets
        for i, (_, ticket) in enumerate(self.tickets_df.head(n_tickets).iterrows()):
            logger.info(f"Processing ticket {i+1}/{n_tickets}")
            
            # Create prompt
            prompt = self._create_prompt(ticket['ticket_text'], few_shot_examples, prompt_type)
            
            # Debug: Show prompt for first ticket
            if i == 0:
                logger.info(f"Sample prompt for {model}, {few_shot_count}-shot, {prompt_type}:")
                logger.info(f"Prompt length: {len(prompt)} characters")
                logger.info(f"Prompt preview: {prompt[:200]}...")
            
            # Query LLM
            response = self._query_llm(prompt, model)
            if response is None:
                continue
            
            # Extract prediction
            prediction = self._extract_category(response)
            ground_truth = ticket['label']
            
            # Check if correct
            correct = prediction == ground_truth
            if correct:
                correct_count += 1
            
            # Debug: Show response for first few tickets
            if i < 3:
                logger.info(f"  Ticket: {ticket['ticket_text'][:50]}...")
                logger.info(f"  Ground truth: {ground_truth}")
                logger.info(f"  Response: '{response}'")
                logger.info(f"  Prediction: {prediction}")
                logger.info(f"  Correct: {correct}")
            
            results.append({
                'ground_truth': ground_truth,
                'prediction': prediction,
                'response': response,
                'correct': correct
            })
        
        # Calculate metrics
        accuracy = correct_count / len(results) if results else 0
        
        # Calculate F1 score
        y_true = [r['ground_truth'] for r in results]
        y_pred = [r['prediction'] for r in results]
        f1_weighted = f1_score(y_true, y_pred, average='weighted') if len(set(y_true)) > 1 else accuracy
        
        return {
            'accuracy': accuracy,
            'f1_weighted': f1_weighted,
            'n_tickets': len(results),
            'results': results
        }
    
    def run_comprehensive_test(self) -> Dict:
        """Führt einen umfassenden Test durch, der die Fehler reproduziert."""
        logger.info("🔬 Running comprehensive test to validate fixes")
        
        # Run all combinations
        test_conditions = [
            ("llama3.1:8b", 0, "structured"),
            ("llama3.1:8b", 0, "unstructured"),
            ("llama3.1:8b", 5, "structured"),
            ("mistral:7b", 0, "structured"),
            ("mistral:7b", 0, "unstructured"),
            ("mistral:7b", 5, "structured"),
        ]
        
        all_results = []
        
        for model, few_shot_count, prompt_type in test_conditions:
            result = self.run_single_test(model, few_shot_count, prompt_type, 5)  # Smaller sample for speed
            
            # Add metadata
            for r in result['results']:
                r.update({
                    'model': model,
                    'few_shot_count': few_shot_count,
                    'prompt_type': prompt_type
                })
                all_results.append(r)
        
        # Create DataFrame for analysis
        results_df = pd.DataFrame(all_results)
        
        # Test ANOVA functionality
        logger.info("Testing ANOVA calculation...")
        try:
            # Convert boolean to int for ANOVA
            anova_data = results_df[['correct', 'model', 'few_shot_count', 'prompt_type']].copy()
            anova_data['correct'] = anova_data['correct'].astype(int)
            anova_data['few_shot_count'] = anova_data['few_shot_count'].astype(str)
            
            logger.info("ANOVA data prepared successfully")
            logger.info(f"Data types: {anova_data.dtypes}")
            logger.info(f"Sample data:\n{anova_data.head()}")
            
        except Exception as e:
            logger.error(f"ANOVA test failed: {e}")
        
        # Test visualization functionality
        logger.info("Testing visualization preparation...")
        try:
            # Test boxplot data preparation
            factor = 'model'
            factor_groups = []
            labels = []
            
            for level in sorted(results_df[factor].unique()):
                group_data = results_df[results_df[factor] == level]['correct'].astype(int)
                factor_groups.append(group_data)
                labels.append(str(level))
            
            logger.info("Boxplot data prepared successfully")
            logger.info(f"Groups: {len(factor_groups)}")
            logger.info(f"Labels: {labels}")
            
        except Exception as e:
            logger.error(f"Visualization test failed: {e}")
        
        return {
            'results_df': results_df,
            'total_tests': len(all_results),
            'success': True
        }

def main():
    quick_test = QuickTest()
    
    print("🚀 Quick Test Tool - Extended Analysis")
    print("=" * 50)
    
    # Test 1: Zero-shot structured
    print("\n1. Testing: llama3.1:8b, 0-shot, structured")
    result1 = quick_test.run_single_test("llama3.1:8b", 0, "structured", 10)
    print(f"Accuracy: {result1['accuracy']:.3f}")
    
    # Test 2: Zero-shot unstructured
    print("\n2. Testing: llama3.1:8b, 0-shot, unstructured")
    result2 = quick_test.run_single_test("llama3.1:8b", 0, "unstructured", 10)
    print(f"Accuracy: {result2['accuracy']:.3f}")
    
    # Test 3: 5-shot structured
    print("\n3. Testing: llama3.1:8b, 5-shot, structured")
    result3 = quick_test.run_single_test("llama3.1:8b", 5, "structured", 10)
    print(f"Accuracy: {result3['accuracy']:.3f}")
    
    # Test 4: Mistral comparison
    print("\n4. Testing: mistral:7b, 0-shot, structured")
    result4 = quick_test.run_single_test("mistral:7b", 0, "structured", 10)
    print(f"Accuracy: {result4['accuracy']:.3f}")
    
    # Test 5: Mistral 5-shot
    print("\n5. Testing: mistral:7b, 5-shot, structured")
    result5 = quick_test.run_single_test("mistral:7b", 5, "structured", 10)
    print(f"Accuracy: {result5['accuracy']:.3f}")
    
    # Test 6: Mistral unstructured
    print("\n6. Testing: mistral:7b, 0-shot, unstructured")
    result6 = quick_test.run_single_test("mistral:7b", 0, "unstructured", 10)
    print(f"Accuracy: {result6['accuracy']:.3f}")
    
    print("\n" + "=" * 50)
    print("COMPREHENSIVE SUMMARY:")
    print(f"Llama3.1 0-shot structured: {result1['accuracy']:.3f}")
    print(f"Llama3.1 0-shot unstructured: {result2['accuracy']:.3f}")
    print(f"Llama3.1 5-shot structured: {result3['accuracy']:.3f}")
    print(f"Mistral 0-shot structured: {result4['accuracy']:.3f}")
    print(f"Mistral 5-shot structured: {result5['accuracy']:.3f}")
    print(f"Mistral 0-shot unstructured: {result6['accuracy']:.3f}")
    
    # Analyze patterns
    print("\n" + "=" * 50)
    print("PATTERN ANALYSIS:")
    
    # Few-shot effect
    llama_few_shot_effect = result3['accuracy'] - result1['accuracy']
    mistral_few_shot_effect = result5['accuracy'] - result4['accuracy']
    
    print(f"Llama3.1 Few-Shot Effect: {llama_few_shot_effect:+.3f}")
    print(f"Mistral Few-Shot Effect: {mistral_few_shot_effect:+.3f}")
    
    # Prompt type effect
    llama_prompt_effect = result1['accuracy'] - result2['accuracy']
    mistral_prompt_effect = result4['accuracy'] - result6['accuracy']
    
    print(f"Llama3.1 Prompt Type Effect: {llama_prompt_effect:+.3f}")
    print(f"Mistral Prompt Type Effect: {mistral_prompt_effect:+.3f}")
    
    # Model comparison
    zero_shot_comparison = result4['accuracy'] - result1['accuracy']
    five_shot_comparison = result5['accuracy'] - result3['accuracy']
    
    print(f"Model Comparison (0-shot): {zero_shot_comparison:+.3f}")
    print(f"Model Comparison (5-shot): {five_shot_comparison:+.3f}")
    
    # Comprehensive test for error validation
    print("\n" + "=" * 50)
    print("COMPREHENSIVE ERROR VALIDATION TEST:")
    comprehensive_result = quick_test.run_comprehensive_test()
    print(f"Comprehensive test completed: {comprehensive_result['success']}")
    print(f"Total tests run: {comprehensive_result['total_tests']}")
    
    # Detailed analysis of results
    print("\n" + "=" * 50)
    print("DETAILED ANALYSIS:")
    
    # Analyze individual results for patterns
    for i, (name, result) in enumerate([
        ("Llama3.1 0-shot structured", result1),
        ("Llama3.1 0-shot unstructured", result2),
        ("Llama3.1 5-shot structured", result3),
        ("Mistral 0-shot structured", result4),
        ("Mistral 5-shot structured", result5),
        ("Mistral 0-shot unstructured", result6)
    ]):
        print(f"\n{name}:")
        print(f"  Accuracy: {result['accuracy']:.3f}")
        print(f"  F1-Score: {result['f1_weighted']:.3f}")
        
        # Show some example predictions
        correct_count = sum(1 for r in result['results'] if r['correct'])
        print(f"  Correct: {correct_count}/{result['n_tickets']}")
        
        # Show some example responses
        print("  Sample responses:")
        for j, r in enumerate(result['results'][:3]):
            status = "✅" if r['correct'] else "❌"
            print(f"    {status} {r['ground_truth']} -> {r['prediction']} ('{r['response'][:50]}...')")

if __name__ == "__main__":
    main() 