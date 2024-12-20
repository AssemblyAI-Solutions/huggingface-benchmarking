from google.cloud import speech
from dotenv import load_dotenv
import pandas as pd
from calculate_wer import calculate_wer
from utils import load_files, convert_32bit_to_16bit
import concurrent.futures

load_dotenv()

client = speech.SpeechClient()

def transcribe_audio(file_uri, language_code, speech_model):
    # Google requires 16bit wav files
    new_file_path = convert_32bit_to_16bit(file_uri)

    with open(new_file_path, "rb") as f:
        audio_content = f.read()

    audio = speech.RecognitionAudio(content=audio_content)

    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=16000,
        language_code=language_code.split("_")[0] + "-" + language_code.split("_")[1].upper(),
        model=speech_model,
    )

    transcript = ''
    try:
        response = client.recognize(config=config, audio=audio)
        transcript = ''
        for result in response.results:
            if result.alternatives:
                transcript += result.alternatives[0].transcript
    except Exception as e:
        print(f"Error during transcription: {e}")

    return transcript

def transcribe_all_files_google(audio_files, labels_list, output_csv_path, language_code, speech_model='default'):
    file_mappings = load_files(audio_files, labels_list)
    audio_paths = []
    truth_text = []
    transcript_outputs = []

    def process_file(file, language_code):
        transcript = transcribe_audio(file['audio'], language_code, speech_model)
        
        if not transcript:
            print(f"Transcription failed for {file['audio']}. Setting transcript to empty string.")
            transcript = ' '
        
        truth_str = file['truth']

        return {
            'audio': file['audio'],
            'truth': truth_str,
            'transcript': transcript
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
    calculate_wer(f"table_csvs/{output_csv_path}", f"table_wers/{output_csv_path}")
    return df