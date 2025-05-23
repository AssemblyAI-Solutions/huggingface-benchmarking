import os

from assemblyai_transcribe_hf import transcribe_all_files_assembly
from openai_whisper_transcribe_hf import transcribe_all_files_whisper
from speechmatics_transcribe_hf import transcribe_all_files_speechmatics
from deepgram_transcribe_hf import transcribe_all_files_deepgram
from amazon_transcribe_hf import transcribe_all_files_amazon
from google_transcribe_hf import transcribe_all_files_google

from hf_datasets import load_fleurs

providers = {
    "assemblyai": transcribe_all_files_assembly,
    "whisper": transcribe_all_files_whisper,
    "speechmatics": transcribe_all_files_speechmatics,
    "deepgram": transcribe_all_files_deepgram,
    "aws": transcribe_all_files_amazon,
    "google": transcribe_all_files_google
}

# SET MODEL FOR EACH PROVIDER
models_per_provider = {
    "deepgram": "nova-2", # nova-3, nova-2, enhanced, general
    "speechmatics": "default",
    "whisper": "whisper-1",
    "assemblyai": "universal", # universal, nano, slam-1
    "aws": "default",
    "google": "latest_long" # latest_long, latest_short
}

# COMMENT OUT PROVIDERS YOU DON'T WANT TO USE
providers_to_use = [
    "assemblyai",
    # "whisper",
    # "speechmatics",
    # "aws",
    # "google",
    # "deepgram",
]

# COMMENT OUT LANGUAGES YOU DON'T WANT TO USE
languages = [
    "fr_fr",
    "it_it",
    "pt_br",
    "nl_nl",
    "hi_in",
    "ja_jp",
    "cmn_hans_cn",
    "fi_fi",
    "ko_kr",
    "pl_pl",
    "ru_ru",
    "tr_tr",
    "uk_ua",
    "vi_vn",
    "en_us",
    "es_419",
    "de_de"
]

file_limit = 100 # set this to 0 to run all files

for language_code in languages:
    dataset = load_fleurs(language_code, file_limit)

    # create table_csvs, table_wers
    os.makedirs('table_csvs', exist_ok=True)
    os.makedirs('table_wers', exist_ok=True)

    if file_limit > 0:
        path_list = dataset['path'][:file_limit]
        labels_list = dataset['target'][:file_limit]
    else:
        path_list = dataset['path']
        labels_list = dataset['target']

    # Important: the audio_files list and labels list need to have the same files in the same order.
    def run(providers_list, file_list=[], labels_list=[]):
        for provider in providers_list:        
            script = providers[provider]
            default_model = models_per_provider[provider]

            # check if provider language file model csv exists
            if os.path.exists(f"table_wers/{provider}_{language_code}_{default_model}.csv"):
                print(f"Skipping {provider}_{language_code}_{default_model}.csv because it already exists")
                continue
            
            try:
                script(file_list, labels_list, f"{provider}_{language_code}_{default_model}.csv", language_code, speech_model=default_model)
            except Exception as e:
                print(f"Error running {provider}_{language_code}_{default_model}.csv: {e}")

    run(providers_to_use, path_list, labels_list)