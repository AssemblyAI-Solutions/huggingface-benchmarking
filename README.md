# Open Source ASR Benchmarking Tool on Hugging Face Datasets

This tool allows users to benchmark Automatic Speech Recognition (ASR) systems using datasets from Hugging Face. It supports multiple transcribers and provides results in a user-friendly format.

## Features
- Supports multiple ASR providers: AssemblyAI, OpenAI Whisper, Speechmatics, Google, Amazon Transcribe, and Deepgram. Just bring your own API keys.
- Easy integration with Hugging Face datasets.
- Analyzes Word Error Rate (WER) and Character Error Rate (CER) for multiple languages.

## Libraries
- [Hugging Face - Truth Dataset](https://huggingface.co/docs/datasets/index)
- [JiWER - Word Error Rate](https://github.com/jitsi/jiwer)
- [Whisper - Normalization](https://github.com/openai/whisper/tree/main/whisper/normalizers)

## Installation

### Prerequisites
- API keys for each provider
- If you want to use AWS you will need to set up a S3 bucket

### Steps
1. Clone the repository
2. Install dependencies:
   ```bash
   pip3 install -r requirements.txt
   ```
3. Configure API keys:
   - Add your API keys to a `.env` file. Refer to `.env_sample` for guidance.
4. This repo is setup to use the Fleurs, LibriSpeech, and CommonVoice datasets. If you want to use a different dataset, you may need to add the dataset to the`hf_datasets.py` file and update the transcoding functions in `utils.py` to match the dataset. You will also need to confirm the language code is supported in the dataset you are using.
5. If you want to run the benchmark on a subset of the dataset, you can set the `file_limit` variable in `run_benchmark_hf.py` to the number of files you want to run.
6. The script is built to run the files in parallel for each provider. Depending on your API key limits, you may need to adjust the `max_workers` variable in the `ThreadPoolExecutor` to a lower number in each provider script.

## Usage
1. Update the `providers_to_use` list in `run_benchmark_hf.py` to include your transcriber of choice.
2. Run the benchmark script. This will automatically download the full dataset from Hugging Face (note: the default dataset is Fleurs which is several GBs of data).
   ```bash
   python3 run_benchmark_hf.py
   ```
3. View results in the `table_wers` and `table_csvs` directories.
4. Optionally, you can run `compare.py` to generate a summary of the results.