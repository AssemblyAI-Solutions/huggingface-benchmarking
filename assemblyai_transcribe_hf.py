import assemblyai as aai 
import os
from dotenv import load_dotenv
import pandas as pd
from calculate_wer import calculate_wer
from utils import load_files
import concurrent.futures

load_dotenv()
aai.settings.api_key = os.getenv('ASSEMBLYAI_API_KEY')

config = aai.TranscriptionConfig(language_code="en")
transcriber = aai.Transcriber(config=config)

def transcribe_all_files_assembly(audio_files, labels_list, output_csv_path, language_code, speech_model='best'):

    file_mappings = load_files(audio_files, labels_list)
    audio_paths = []
    truth_text = []
    transcript_outputs = []

    def process_file(file, language_code):
        if language_code == "cmn_hans_cn":
            language_code = "zh"
        
        model = aai.SpeechModel.best
        if speech_model != "best":
            model = aai.SpeechModel.nano
        
        transcript = transcriber.transcribe(file['audio'], config=aai.TranscriptionConfig(language_code=language_code.split("_")[0], speech_model=model))
        print(transcript.text)

        truth_str = file['truth']

        return {
            'audio': file['audio'],
            'truth': truth_str,
            'transcript': transcript.text
        }
        
    with concurrent.futures.ThreadPoolExecutor(max_workers=200) as executor:
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