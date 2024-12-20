import os 
import time
import pandas as pd
from calculate_wer import calculate_wer
from utils import load_files, convert_flac_to_mp3
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
openai_key = os.getenv('OPENAI_API_KEY')

client = OpenAI(api_key=openai_key)

def transcribe_all_files_whisper(audio_files, labels_list, output_csv_path, language_code, speech_model='whisper-1'):

    file_mappings = load_files(audio_files, labels_list)

    audio_paths = []
    truth_text = []
    transcript_outputs = []

    for file in file_mappings:
        try:
            # Note: Whisper does not support flac
            # mp3 = convert_flac_to_mp3(file['audio'])
            audio_path = file['audio']
            with open(audio_path, "rb") as audio_file:
                transcript = client.audio.transcriptions.create(
                    file=audio_file,
                    model=speech_model,
                    # prompt="Sample whisper context here.",
                    response_format="text",
                    language=language_code.split("_")[0]
                )
        except Exception as e:
            print(f"Error transcribing {audio_path}: {e}")
            transcript = "transcription failed"

        audio_paths.append(audio_path)

        try:
            truth_str = file['truth']
        except Exception as e:
            print(f"Error reading truth file {file['truth']}: {e}")
            truth_str = "truth file read failed"

        truth_text.append(truth_str)
        print(transcript)
        transcript_outputs.append(transcript)
        time.sleep(3)

    df = pd.DataFrame({
        "audio_path": audio_paths,
        "target": truth_text,
        "prediction": transcript_outputs
    })

    df.to_csv(f"table_csvs/{output_csv_path}", index=False)
    print(df)
    calculate_wer(f"table_csvs/{output_csv_path}", f"table_wers/{output_csv_path}")
    
    return df