# files in table_wers have csv format with columns: audio_path,target,prediction,wer,insertions,deletions,substitutions,duration
# file format is {model}_{language_code}.csv
# we want to average the wers for each model language pairing
# then we want to print out the average wer for each model across all languages
# we can output this in a table format with tabulate

import os
import pandas as pd
from tabulate import tabulate

def process_csv_files(directory):
    results = {}
    for filename in os.listdir(directory):
        if filename.endswith('.csv'):
            model, lang_code = filename.split('_', 1)
            lang_code = lang_code.rsplit('.', 1)[0]  # Remove file extension
            
            df = pd.read_csv(os.path.join(directory, filename))
            avg_wer = df['wer'].mean()
            
            if model not in results:
                results[model] = {}
            results[model][lang_code] = avg_wer
    
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
    models = ['assemblyai', 'speechmatics']  # Specify the order of models
    
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
    headers = ['Language', 'assemblyai', 'speechmatics']
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