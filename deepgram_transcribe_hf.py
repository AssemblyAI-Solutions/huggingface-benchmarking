import os
from dotenv import load_dotenv
import pandas as pd
from calculate_wer import calculate_wer
from utils import load_files
import concurrent.futures
from deepgram import (
    DeepgramClient,
    PrerecordedOptions,
    FileSource,
)

load_dotenv()
deepgram_api_key = os.getenv('DEEPGRAM_API_KEY')
deepgram = DeepgramClient(deepgram_api_key)

def transcribe_all_files_deepgram(audio_files, labels_list, output_csv_path, language_code, speech_model='nova-2'):
    file_mappings = load_files(audio_files, labels_list)
    audio_paths = []
    truth_text = []
    transcript_outputs = []

    def process_file(file, language_code):
        if language_code == "cmn_hans_cn":
            language_code = "zh"
            
        with open(file['audio'], 'rb') as audio:
            buffer_data = audio.read()

        payload: FileSource = {
            "buffer": buffer_data,
        }

        options = PrerecordedOptions(
            model=speech_model,
            smart_format=True,
            language=language_code.split("_")[0]
        )

        response = deepgram.listen.rest.v("1").transcribe_file(payload, options)
        transcript = response.results.channels[0].alternatives[0].transcript
        print(transcript)

        truth_str = file['truth']

        return {
            'audio': file['audio'],
            'truth': truth_str,
            'transcript': transcript
        }
        
    with concurrent.futures.ThreadPoolExecutor(max_workers=50) as executor:
        futures = [executor.submit(process_file, file, language_code) for file in file_mappings]
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            print(result['transcript'])
            
            truth_text.append(result['truth'])
            audio_paths.append(result['audio'])
            transcript_outputs.append(result['transcript'])

    df = pd.DataFrame({
        "audio_path": audio_paths,
        "target": truth_text,
        "prediction": transcript_outputs
    })

    df.to_csv(f"table_csvs/{output_csv_path}", index=False)
    print(df)
    calculate_wer(f"table_csvs/{output_csv_path}", f"table_wers/{output_csv_path}", language_code)
    return df