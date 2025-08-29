# Dolma Dataset Splitter

This project downloads the `pico-lm/pretokenized-dolma` dataset from Hugging Face, truncates it to the first 20GB, and uploads it to `ThomastheMaker/pretokenized-dolma-20GB`.

## Setup

1. **Activate the virtual environment and install dependencies:**
   ```bash
   chmod +x setup.sh
   ./setup.sh
   ```

2. **Or manually:**
   ```bash
   source myenv/bin/activate
   pip install -r requirements.txt
   ```

## Usage

1. **Set up your Hugging Face token:**
   - Get your token from [Hugging Face Settings](https://huggingface.co/settings/tokens)
   - Create a `.env` file in the project root with:
     ```
     HUGGINGFACE_TOKEN=your_actual_token_here
     ```
   - Or copy from the template: `cp env_template.txt .env` and edit

2. **Run the script:**
   ```bash
   source myenv/bin/activate
   python split.py [target_examples] [custom_dataset_name] [--no-streaming]
   ```

   **Examples:**
   - `python split.py` - Uses default 1M examples with auto-generated name
   - `python split.py 1000000` - Creates 1M examples dataset with auto-generated name
   - `python split.py 500000` - Creates 500K examples dataset with auto-generated name
   - `python split.py 1000000 my-dolma-dataset` - Creates 1M examples dataset named "my-dolma-dataset"
   - `python split.py 2000000 dolma-2m-examples` - Creates 2M examples dataset named "dolma-2m-examples"
   - `python split.py 5000000 ThomasTheMaker/pretokenized-dolma-5M` - Creates 5M examples dataset with full repository path
   - `python split.py 5000000 ThomasTheMaker/pretokenized-dolma-5M --no-streaming` - Uses standard loading mode

## What the script does

1. Downloads the `pico-lm/pretokenized-dolma` dataset
2. **🚀 Uses streaming to process only the examples you need** (much faster than loading 205M examples!)
3. Collects examples until your target count is reached, then stops
4. **Saves the truncated dataset locally first** (safeguard against upload failures)
5. Automatically generates a dataset name based on the target examples (e.g., `ThomastheMaker/pretokenized-dolma-1M`)
6. Creates a new repository on Hugging Face with the generated name
7. Uploads the truncated dataset from local storage
8. Verifies the upload
9. Provides clear paths to both local and remote datasets

## Requirements

- Python 3.8+
- Hugging Face account with write access
- Sufficient disk space for temporary dataset storage
- Internet connection for downloading and uploading

## Notes

- The script uses temporary storage to process the dataset
- Dataset size estimation is approximate and may vary
- The upload process may take some time depending on your internet connection
- Make sure you have sufficient disk space for the original dataset during processing
