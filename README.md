## Open Source ASR Benchmarking on Hugging Face Datasets

1) Run `pip install -r requirements.txt`

2) Add your respective API keys to a `.env` file (depending on the vendors you want to use) in the respective transcriber scripts. See `.env_sample`

3) Update the `providers` dictionary in `run_benchmark_hf.py` to include your transcriber of choice. Right now we have `assemblyai`, `openai_whisper` and `speechmatics`.

4) Run `run_benchmark_hf.py` to execute benchmarks on hugging face dataset. The default dataset is Fleurs. You can add more datasets to the `hf_datasets.py` file.

5) The results will be saved to the generated `table_wers` and `table_csvs` folders.