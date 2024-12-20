import boto3
import os
from dotenv import load_dotenv
import pandas as pd
from calculate_wer import calculate_wer
from utils import load_files
import concurrent.futures
import time
import uuid
import requests
load_dotenv()

# Initialize the Amazon Transcribe client
transcribe_client = boto3.client('transcribe',
                                 region_name=os.getenv('AWS_REGION'),
                                 aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
                                 aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'))

def transcribe_audio(file_uri, job_name, language_code):
    new_language_code = language_code.split("_")[0] + "-" + language_code.split("_")[1].upper()
    response = transcribe_client.start_transcription_job(
        TranscriptionJobName=job_name,
        Media={'MediaFileUri': file_uri},
        MediaFormat='mp3',
        LanguageCode=new_language_code
    )
    return response

def upload_audio_to_s3(file_path, bucket_name):
    s3_client = boto3.client('s3',
                             region_name=os.getenv('AWS_REGION'),
                             aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
                             aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'))
    s3_client.upload_file(file_path, bucket_name, os.path.basename(file_path))
    return f"s3://{bucket_name}/{os.path.basename(file_path)}"

def get_transcript_from_uri(transcript_uri):
    req = requests.get(transcript_uri)
    return req.json()['results']['transcripts'][0]['transcript']

# Function to transcribe multiple audio files using Amazon Transcribe
def transcribe_all_files_amazon(audio_files, labels_list, output_csv_path, language_code, speech_model='default'):
    file_mappings = load_files(audio_files, labels_list)
    audio_paths = []
    truth_text = []
    transcript_outputs = []

    def process_file(file, language_code):
        # Generate a unique job name using UUID
        unique_id = uuid.uuid4()
        job_name = f"transcription-{file['audio'].split('/')[-1]}-{unique_id}"
        file_uri = upload_audio_to_s3(file['audio'], os.getenv('AWS_BUCKET_NAME'))
        transcribe_audio(file_uri, job_name, language_code)

        # Polling for job completion
        while True:
            status = transcribe_client.get_transcription_job(TranscriptionJobName=job_name)
            job_status = status['TranscriptionJob']['TranscriptionJobStatus']
            
            if job_status in ['COMPLETED', 'FAILED']:
                break
            print(f"Waiting for job {job_name} to complete...")
            time.sleep(5)  # Wait for 5 seconds before checking again

        if job_status == 'COMPLETED':
            transcript_uri = status['TranscriptionJob']['Transcript']['TranscriptFileUri']
            transcript = get_transcript_from_uri(transcript_uri)
        else:
            print(f"Transcription job {job_name} failed.")
            transcript = None

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