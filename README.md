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
   python split.py
   ```

## What the script does

1. Downloads the `pico-lm/pretokenized-dolma` dataset
2. Estimates the dataset size and truncates it to approximately 20GB
3. Creates a new repository `ThomastheMaker/pretokenized-dolma-20GB` on Hugging Face
4. Uploads the truncated dataset
5. Verifies the upload

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
