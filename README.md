# Dolma Dataset Splitter

This project downloads the `pico-lm/pretokenized-dolma` dataset from Hugging Face, truncates it to the first 20GB, and uploads it to `ThomastheMaker/pretokenized-dolma-20GB`.

## Hardware

You will need a lot of storage, roughly 1.5x to 2x the dataset size you are looking to process. For example, Dolma is 780GB, so at least 1.2TB is needed.

Additionally, high bandwidth is recommended (downloading and uploading hundreds of GB of data is no small feat).

With those 2 requirements, I highly recommend a Cloud Virtual Machine. Here are a few options for cheap/free:
- Azure VM - Microsoft Startup Program takes 5m to setup & gives you $1000
- GCP - A free account gives you some good credits too
- Hetzner - Awesome, reliably & awesomely cheap

For context, I use $1000 in Azure credits to run my 8 vCPU, 32GB RAM & 2TB storage. the CPU and RAM are barely used (2 vCPU & 8 GB RAM are enough). But storage is well worth it

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
   - `python split.py 5000000 ThomasTheMaker/pretokenized-dolma-5M --no-streaming` - Uses standard loading mode.

## Tokenizing Datasets

This project also includes a dataset tokenizer that can convert raw text datasets into tokenized format using various Hugging Face tokenizers.

### Performance Notes

**Hardware Requirements for Tokenization:**
- **Storage**: 2-3x the original dataset size (for both source and tokenized versions)
- **RAM**: 16GB+ recommended for large datasets
- **CPU**: Multi-core CPU helps with processing speed
- **Time**: Tokenization can be time-intensive for large datasets

**Real-world Performance Example:**
On my Azure VM (8 vCPU, 32GB RAM, 2TB storage), tokenizing the entire Wikipedia English dataset (~6M articles) takes approximately **12 hours** using the OLMo-7B tokenizer.

### Setup for Tokenization

1. **Install additional dependencies:**
   ```bash
   source myenv/bin/activate
   pip install transformers tqdm
   ```

2. **Login to Hugging Face Hub:**
   ```bash
   huggingface-cli login
   ```

### Usage

1. **Create a configuration file** (e.g., `wikipedia-en.json`):
   ```json
   {
     "dataset": "wikimedia/wikipedia",
     "column": "text",
     "subset": "20231101.en",
     "tokenizer": "allenai/OLMo-7B-0724-hf",
     "output_column": "input_ids",
     "output_dataset": "YourUsername/pretokenized_wiki_en"
   }
   ```

2. **Run the tokenizer:**
   ```bash
   source myenv/bin/activate
   python tokenize-datasets/tokenize_datasets.py wikipedia-en.json
   ```

   **Options:**
   - `--private`: Make the output dataset private
   - `--dry-run`: Process without uploading to Hub

### Configuration Options

- **dataset**: Source dataset name on Hugging Face Hub
- **column**: Text column to tokenize
- **subset**: Dataset subset (optional)
- **tokenizer**: Hugging Face tokenizer model to use
- **output_column**: Name for the tokenized output column
- **output_dataset**: Target dataset name on Hugging Face Hub

### Supported Tokenizers

Any Hugging Face tokenizer can be used, including:
- `allenai/OLMo-7B-0724-hf` (recommended for OLMo models)
- `gpt2` (GPT-2 tokenizer)
- `bert-base-uncased` (BERT tokenizer)
- `t5-base` (T5 tokenizer)
- Custom fine-tuned tokenizers

### Tips for Large Datasets

1. **Monitor progress**: The script shows progress bars and estimated completion time
2. **Use dry-run first**: Test with `--dry-run` before processing large datasets
3. **Consider subsetting**: Process smaller portions first to test your setup
4. **Monitor resources**: Keep an eye on disk space and memory usage
5. **Batch processing**: For very large datasets, consider processing in chunks

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
