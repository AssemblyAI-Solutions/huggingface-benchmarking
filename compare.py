# Process CSV files in 'table_wers' directory to calculate average WER for each model-language pair.
# Output the average WER for each model across all languages in a table format using tabulate.

import os
import pandas as pd
from tabulate import tabulate

def process_csv_files(directory):
    results = {}
    for filename in os.listdir(directory):
        if filename.endswith('.csv'):
            # Parse filename like "assemblyai_it_it_nano.csv"
            parts = filename.replace('.csv', '').split('_')
            provider = parts[0]
            lang_code = parts[1]
            model_variant = parts[-1]  # Get the model variant (e.g., 'nano')
            
            # Create a unique model identifier combining provider and variant
            model_id = f"{provider}_{model_variant}"
            
            df = pd.read_csv(os.path.join(directory, filename))
            avg_wer = df['wer'].mean()
            
            if model_id not in results:
                results[model_id] = {}
            results[model_id][lang_code] = avg_wer
    
    return results

def calculate_overall_averages(results):
    overall_averages = {}
    for model, lang_data in results.items():
        overall_averages[model] = sum(lang_data.values()) / len(lang_data)
    return overall_averages

def main():
    directory = 'table_wers'
    results = process_csv_files(directory)
    
    # Prepare data for tabulation
    table_data = []
    languages = sorted(set(lang for model_data in results.values() for lang in model_data.keys()))
    
    # Get unique models from results
    models = sorted(results.keys())
    
    for lang in languages:
        row = [lang]
        for model in models:
            value = results.get(model, {}).get(lang, 'N/A')
            if value != 'N/A':
                row.append(f"{value:.4f}")
            else:
                row.append(value)
        table_data.append(row)
    
    # Print table of WERs for each model and language
    headers = ['Language'] + models
    print(tabulate(table_data, headers=headers, tablefmt='grid'))
    
    # Calculate and print overall averages
    overall_averages = calculate_overall_averages(results)
    print("\nOverall Average WER for each model:")
    for model in models:
        if model in overall_averages:
            print(f"{model}: {overall_averages[model]:.4f}")
        else:
            print(f"{model}: N/A")

if __name__ == "__main__":
    main()