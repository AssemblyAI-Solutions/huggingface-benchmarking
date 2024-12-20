import os 
import pandas as pd
from calculate_wer import calculate_wer
from utils import load_files, convert_flac_to_mp3
from dotenv import load_dotenv
from openai import OpenAI
import concurrent.futures

load_dotenv()
openai_key = os.getenv('OPENAI_API_KEY')

client = OpenAI(api_key=openai_key)

def process_file(file, language_code, speech_model):
    if 'cmn' in language_code:
        language_code = 'zh'

    audio_path = file['audio']
    try:
        with open(audio_path, "rb") as audio_file:
            transcript = client.audio.transcriptions.create(
                file=audio_file,
                model=speech_model,
                response_format="text",
                language=language_code.split("_")[0]
            )
    except Exception as e:
        print(f"Error transcribing {audio_path}: {e}")
        transcript = "transcription failed"

    try:
        truth_str = file['truth']
    except Exception as e:
        print(f"Error reading truth file {file['truth']}: {e}")
        truth_str = "truth file read failed"

    return {
        'audio': audio_path,
        'truth': truth_str,
        'transcript': transcript
    }

def transcribe_all_files_whisper(audio_files, labels_list, output_csv_path, language_code, speech_model='whisper-1'):

    file_mappings = load_files(audio_files, labels_list)

    audio_paths = []
    truth_text = []
    transcript_outputs = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=100) as executor:
        futures = [executor.submit(process_file, file, language_code, speech_model) for file in file_mappings]
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            print(result['transcript'])

            audio_paths.append(result['audio'])
            truth_text.append(result['truth'])
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