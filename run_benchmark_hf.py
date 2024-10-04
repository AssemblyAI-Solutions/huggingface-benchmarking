from hf_datasets import load_fleurs
from assemblyai_transcribe_hf import transcribe_all_files_assembly
# from openai_whisper_transcribe_hf import transcribe_all_files_whisper
from speechmatics_transcribe_hf import transcribe_all_files_speechmatics
import os

providers = {
    "assemblyai": transcribe_all_files_assembly,
    # "openai_whisper": transcribe_all_files_whisper,
    "speechmatics": transcribe_all_files_speechmatics
}

providers_to_use = [
    "assemblyai",
    # "openai_whisper"
    "speechmatics"
]

languages = [
    # "fr_fr",
    # "it_it",
    # "pt_br",
    # "nl_nl",
    # "hi_in",
    # "ja_jp",
    "cmn_hans_cn",
    # "fi_fi",
    # "ko_kr",
    # "pl_pl",
    # "ru_ru",
    # "tr_tr",
    # "uk_ua",
    # "vi_vn",
]

file_limit = 50 # default is 50

for language_code in languages:
    dataset = load_fleurs(language_code)

    # create table_csvs, table_wers
    os.makedirs('table_csvs', exist_ok=True)
    os.makedirs('table_wers', exist_ok=True)

    # run for 50 files
    path_list = dataset['path'][:file_limit]
    labels_list = dataset['target'][:file_limit]

    #important: the audio_files list and labels list need to have the same files in the same order.
    def run(providers_list, file_list=[], labels_list=[]):
        for provider in providers_list:
            script = providers[provider]  
            if provider == "assemblyai":
                script(file_list, labels_list, f"{provider}_{language_code}.csv", language_code, speech_model='best') # best or nano
            else:
                script(file_list, labels_list, f"{provider}_{language_code}.csv", language_code)

    run(providers_to_use, path_list, labels_list)