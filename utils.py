import os
from pydub import AudioSegment
from dotenv import load_dotenv
load_dotenv()

def load_files(audio_files, text_files):
    paths = []
    for text_file, audio_file in zip(text_files, audio_files):
        updated_audio_file = reformat_file_path(audio_file)
        file_mapping = {
            "audio": updated_audio_file,
            "truth": text_file
        }
        paths.append(file_mapping)
    # for file in text_files:
    #     split_path = file.split(".txt")[0]
    #     audio_path = f"{audio_file_dir}/{split_path}"
    #     file_mapping = {
    #         "audio": audio_path,
    #         "truth": f"{text_file_dir}/{file}"
    #     }
    #     paths.append(file_mapping)
    return paths

#needed at times for whisper
def convert_flac_to_mp3(flac_path):
    """Converts a FLAC file to MP3 format and saves it in the 'mp3s' folder."""
    # Ensure the 'mp3s' directory exists
    mp3_dir = "mp3s"
    os.makedirs(mp3_dir, exist_ok=True)
    
    # Construct the MP3 path from the FLAC path
    mp3_path = os.path.join(mp3_dir, os.path.basename(flac_path).replace('.mp3', '.flac'))
    
    # Convert and save the MP3
    audio = AudioSegment.from_file(flac_path, "flac")
    audio.export(mp3_path, format="mp3")
    
    return mp3_path

def reformat_file_path(wrong_path):
    base_dir = os.getenv('BASE_DIR')
    unique_id = wrong_path.split('/')[8]
    file_name = os.path.basename(wrong_path)
    
    # Extract the first two IDs from the file name
    # first_id, second_id, _ = file_name.split('-')
    
    # Construct the correct path, you may need to change this depending on the dataset you are using
    correct_path = os.path.join(base_dir, unique_id, "test", file_name)
    
    return correct_path