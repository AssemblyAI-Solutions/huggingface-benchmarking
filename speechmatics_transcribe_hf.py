from dotenv import load_dotenv
import pandas as pd
from calculate_wer import calculate_wer
from utils import load_files
import os
import concurrent.futures
import time

load_dotenv()

from speechmatics.models import BatchTranscriptionConfig, FetchData
from speechmatics.batch_client import BatchClient

API_KEY = os.getenv('SPEECHMATICS_API_KEY')

headers = {
    'Authorization': 'Bearer ' + API_KEY,
}

def transcribe(path, language):
    language = language.split("_")[0]
    if language == 'zh':
        language = 'cmn'

    with BatchClient(API_KEY) as client:
        try:
            config = BatchTranscriptionConfig(
                language=language, 
                operating_point='enhanced',
            )

            job_id = client.submit_job(path, config)
            # return job_id
            
            # Wait for the job to complete
            transcript = client.wait_for_completion(job_id, transcription_format='txt')
            if not transcript:
                return None
            
            return transcript
        except Exception as e:
            print(e)

def transcribe_all_files_speechmatics(audio_files, labels_list, output_csv_path, language_code, speech_model=''):

    file_mappings = load_files(audio_files, labels_list)
    audio_paths = []
    truth_text = []
    transcript_outputs = []

    def process_file(file, language_code):
        transcript = transcribe(file['audio'], language_code)
        print(transcript)

        truth_str = file['truth']

        return {
            'audio': file['audio'],
            'truth': truth_str,
            'transcript': transcript
        }
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=200) as executor:
        futures = []
        for file in file_mappings:
            futures.append(executor.submit(process_file, file, language_code))
            if len(futures) % 10 == 0:
                time.sleep(5)
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