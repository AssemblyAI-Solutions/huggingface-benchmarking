from datasets import load_dataset
import pandas as pd
import os

def load_commonvoice():
    dataset = load_dataset("mozilla-foundation/common_voice_5_1", "en", split="test", trust_remote_code=True)
    cv_df = pd.DataFrame(
        {
            "path": dataset['path'],
            "target": dataset['sentence']
        }
    )
    return cv_df

def load_librispeech_test_clean():
    dataset = load_dataset("librispeech_asr", 'clean', split="test", trust_remote_code=True)
    print(dataset['audio'][0])
    cv_df = pd.DataFrame(
        {
            "path": dataset['file'],
            "target": dataset['text']
        }
    )
    return cv_df

def load_fleurs(language_code='en_us', file_limit=100):
    cache_dir = "/cache/huggingface/datasets" if os.path.exists("/cache") else None
    print(f"Loading FLEURS dataset for language: {language_code}")
    
    # Load dataset with streaming to avoid downloading all files at once
    try:
        dataset = load_dataset(
            "google/fleurs",
            language_code,
            split=f"test[0:{file_limit}]",
            trust_remote_code=True,
            cache_dir=cache_dir,
        )
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None
    
    # Debug: Print first item to see its structure
    try:
        print('Length of dataset:', len(dataset))
        first_item = dataset[0]
        print("First item structure:", first_item.keys())
        print("First item path:", first_item.get('path'))
        print("First item transcription:", first_item.get('transcription'))
    except IndexError:
        print("Error: Dataset list is empty or index out of range")
    
    # Create DataFrame with audio data and transcriptions
    df = pd.DataFrame(
        {
            "path": dataset['path'],
            "target": dataset['transcription']
        }
    )
    return df