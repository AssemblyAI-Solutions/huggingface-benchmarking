import os
import subprocess
from pydub import AudioSegment

from dotenv import load_dotenv
load_dotenv()

def load_files(audio_files, text_files):
    paths = []
    for text_file, audio_file in zip(text_files, audio_files):
        # reformat the audio file path to find the correct path
        updated_audio_file = reformat_file_path(audio_file)
        file_mapping = {
            "audio": updated_audio_file,
            "truth": text_file
        }
        paths.append(file_mapping)
    return paths

def reformat_file_path(wrong_path):
    # First check if the original path exists
    if os.path.exists(wrong_path):
        return wrong_path
        
    file_name = os.path.basename(wrong_path)
    
    # Try /cache path structure, used for running remotely in Modal
    if wrong_path.startswith('/cache/huggingface'):
        path_parts = wrong_path.split('/')
        if len(path_parts) > 6:
            hash_dir = path_parts[6]
            # Try with split directories
            for split_dir in ("test", "dev", "train"):
                candidate = os.path.join(
                    "/cache",
                    "huggingface",
                    "datasets",
                    "downloads",
                    "extracted",
                    hash_dir,
                    split_dir,
                    file_name,
                )
                if os.path.exists(candidate):
                    return candidate
                    
            # Fallback to direct path under hash_dir
            fallback_path = os.path.join(
                "/cache",
                "huggingface",
                "datasets",
                "downloads",
                "extracted",
                hash_dir,
                file_name,
            )
            return fallback_path
            
    # Try local path structure
    base_dir = os.getenv('BASE_DIR', os.path.expanduser("~/.cache/huggingface/datasets/downloads/extracted"))
    
    # Try to find the file in common dataset directory structures
    for split_dir in ("test", "dev", "train"):
        local_candidate = os.path.join(base_dir, split_dir, file_name)
        if os.path.exists(local_candidate):
            return local_candidate

    # If a unique_id can be extracted from the path, try that structure
    try:
        path_parts = wrong_path.split('/')
        if len(path_parts) > 8:
            unique_id = path_parts[8]
            local_path = os.path.join(base_dir, unique_id, "test", file_name)
            if os.path.exists(local_path):
                return local_path
    except IndexError:
        pass
    # If all attempts fail, return the original path
    return wrong_path

# needed at times for whisper
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

# needed at times for google
def convert_32bit_to_16bit(file_path):
    # first check if the file is already 16bit
    if os.path.exists(file_path.replace('.wav', '-16bit.wav')):
        return file_path.replace('.wav', '-16bit.wav')
    else:
        subprocess.run(['ffmpeg', '-i', file_path, '-acodec', 'pcm_s16le', '-ar', '16000', file_path.replace('.wav', '-16bit.wav')])
        return file_path.replace('.wav', '-16bit.wav')