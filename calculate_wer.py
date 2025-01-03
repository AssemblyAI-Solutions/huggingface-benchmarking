from jiwer import wer, cer, compute_measures
import pandas as pd
from whisper_normalizer.basic import BasicTextNormalizer
from pydub import AudioSegment

# Initialize the normalizer
normalizer = BasicTextNormalizer()

def get_audio_duration(audio_path):
    # Load audio file and get its duration in seconds
    audio = AudioSegment.from_file(audio_path)
    return len(audio) / 1000.0

def calculate_wer(csv_path, metrics_output_path, language_code='en'):

    wer_calculations = []
    insertions = []
    deletions = []
    substitutions = []
    durations = []
    df = pd.read_csv(csv_path)

    for index, row in df.iterrows():
        print(row["audio_path"])
        print(row["prediction"])
        try:
            normalized_target = normalizer(row["target"])
            normalized_prediction = normalizer(row["prediction"])
        except Exception as e:
            # if the prediction is empty the normalizer will throw an error. Set the prediction to an empty string.
            normalized_target = row["target"]
            normalized_prediction = ' '

        measures = compute_measures(normalized_target, normalized_prediction)
        word_error_rate = wer(normalized_target, normalized_prediction)
        character_error_rate = cer(normalized_target, normalized_prediction)

        insertions.append(measures['insertions'])
        deletions.append(measures['deletions'])
        substitutions.append(measures['substitutions'])
        if 'zh' in language_code.lower() or 'cmn' in language_code.lower() or 'ja' in language_code.lower():
            wer_calculations.append(character_error_rate)
        else:
            wer_calculations.append(word_error_rate)
        
        # Get duration of the audio file
        duration = get_audio_duration(row["audio_path"])
        durations.append(duration)
    
    df["wer"] = wer_calculations
    df["insertions"] = insertions
    df["deletions"] = deletions
    df["substitutions"] = substitutions
    df["duration"] = durations

    # Calculate the normalized average WER
    total_duration = sum(durations)
    normalized_average_wer = sum(w * d for w, d in zip(wer_calculations, durations)) / total_duration

    # Save results to CSV
    df.to_csv(f"{metrics_output_path}", index=False)
    print(f"Normalized Average WER: {normalized_average_wer}")

    return df, normalized_average_wer

if __name__ == "__main__":
    calculate_wer("table_csvs/deepgram_uk_ua_nova-2.csv", "table_wers/deepgram_uk_ua_nova-2.csv")