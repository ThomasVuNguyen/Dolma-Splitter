#!/usr/bin/env python3
"""
Dataset Splitter and Uploader
Downloads pico-lm/pretokenized-dolma dataset, saves first 20GB, and uploads to ThomastheMaker/pretokenized-dolma-20GB
"""

import os
import sys
import json
import tempfile
import shutil
from pathlib import Path
from typing import Iterator, Dict, Any
import logging
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    from datasets import load_dataset, Dataset, DatasetDict
    from huggingface_hub import HfApi, login
except ImportError as e:
    logger.error(f"Required packages not installed: {e}")
    logger.info("Please install: pip install datasets huggingface_hub")
    sys.exit(1)

def get_dataset_size_gb(dataset: Dataset) -> float:
    """Estimate dataset size in GB"""
    try:
        # Get memory usage in bytes and convert to GB
        memory_usage = dataset.data.nbytes
        size_gb = memory_usage / (1024**3)
        return size_gb
    except Exception as e:
        logger.warning(f"Could not determine exact size: {e}")
        # Fallback: estimate based on number of examples
        num_examples = len(dataset)
        estimated_size_gb = num_examples * 0.001  # Rough estimate: 1KB per example
        return estimated_size_gb

def format_size_gb(size_gb: float) -> str:
    """Format size in GB to human readable string"""
    if size_gb >= 1:
        return f"{size_gb:.2f} GB"
    else:
        return f"{size_gb * 1024:.2f} MB"

def truncate_dataset_to_size(dataset: Dataset, target_size_gb: float) -> Dataset:
    """Truncate dataset to approximately target size in GB"""
    current_size_gb = get_dataset_size_gb(dataset)
    logger.info(f"Current dataset size: {format_size_gb(current_size_gb)}")
    
    if current_size_gb <= target_size_gb:
        logger.info("Dataset is already within target size")
        return dataset
    
    # Calculate how many examples to keep
    ratio = target_size_gb / current_size_gb
    target_examples = int(len(dataset) * ratio)
    
    logger.info(f"Truncating from {len(dataset)} to {target_examples} examples")
    truncated_dataset = dataset.select(range(target_examples))
    
    # Verify final size
    final_size_gb = get_dataset_size_gb(truncated_dataset)
    logger.info(f"Final dataset size: {format_size_gb(final_size_gb)}")
    
    return truncated_dataset

def main():
    """Main function to download, split, and upload dataset"""
    
    # Load environment variables
    load_dotenv()
    
    # Configuration
    source_dataset = "pico-lm/pretokenized-dolma"
    
    # Get target size from command line argument or use default
    if len(sys.argv) > 1:
        try:
            target_size_gb = float(sys.argv[1])
            if target_size_gb <= 0:
                raise ValueError("Target size must be positive")
        except ValueError as e:
            logger.error(f"Invalid target size: {sys.argv[1]}. Please provide a positive number.")
            logger.info("Usage: python split.py [target_size_gb]")
            logger.info("Example: python split.py 10.5")
            sys.exit(1)
    else:
        target_size_gb = 20.0  # Default size
        logger.info(f"No target size specified, using default: {target_size_gb} GB")
    
    # Generate dataset name based on size
    if target_size_gb >= 1:
        size_suffix = f"{int(target_size_gb)}GB"
    else:
        size_suffix = f"{int(target_size_gb * 1024)}MB"
    
    target_dataset = f"ThomastheMaker/pretokenized-dolma-{size_suffix}"
    
    # Check for Hugging Face token
    hf_token = os.getenv('HUGGINGFACE_TOKEN')
    if not hf_token:
        logger.error("HUGGINGFACE_TOKEN not found in environment variables or .env file")
        logger.info("Please create a .env file with your Hugging Face token:")
        logger.info("HUGGINGFACE_TOKEN=your_token_here")
        sys.exit(1)
    
    logger.info(f"Starting dataset processing: {source_dataset} -> {target_dataset}")
    logger.info(f"Target size: {format_size_gb(target_size_gb)}")
    
    try:
        # Step 1: Load the source dataset
        logger.info(f"Loading dataset: {source_dataset}")
        dataset = load_dataset(source_dataset)
        
        if isinstance(dataset, DatasetDict):
            # If it's a DatasetDict, get the first split
            first_split_name = list(dataset.keys())[0]
            dataset = dataset[first_split_name]
            logger.info(f"Using split: {first_split_name}")
        
        logger.info(f"Dataset loaded successfully. Shape: {dataset.shape}")
        
        # Step 2: Truncate to target size
        logger.info(f"Truncating dataset to {format_size_gb(target_size_gb)}")
        truncated_dataset = truncate_dataset_to_size(dataset, target_size_gb)
        
        # Step 3: Create temporary directory for processing
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Step 4: Save truncated dataset
            logger.info("Saving truncated dataset...")
            save_path = temp_path / "dataset"
            truncated_dataset.save_to_disk(str(save_path))
            
            # Step 5: Initialize Hugging Face API with token
            logger.info("Initializing Hugging Face API with token")
            api = HfApi(token=hf_token)
            
            # Verify token is valid
            try:
                user_info = api.whoami()
                logger.info(f"Successfully authenticated as: {user_info['name']}")
            except Exception as e:
                logger.error(f"Failed to authenticate with Hugging Face: {e}")
                logger.error("Please check your HUGGINGFACE_TOKEN in the .env file")
                raise
            
            # Step 6: Upload dataset
            logger.info(f"Uploading dataset to: {target_dataset}")
            
            # Create repository if it doesn't exist
            try:
                api.create_repo(
                    repo_id=target_dataset,
                    repo_type="dataset",
                    exist_ok=True,
                    private=False
                )
                logger.info(f"Repository {target_dataset} created/verified")
            except Exception as e:
                logger.warning(f"Repository creation warning: {e}")
            
            # Upload the dataset
            api.upload_folder(
                folder_path=str(save_path),
                repo_id=target_dataset,
                repo_type="dataset",
                commit_message=f"Initial upload: First {target_size_gb}GB of pretokenized-dolma dataset"
            )
            
            logger.info(f"Dataset successfully uploaded to: {target_dataset}")
            
            # Step 7: Verify upload
            try:
                uploaded_dataset = load_dataset(target_dataset)
                logger.info(f"Upload verification successful. Dataset shape: {uploaded_dataset.shape}")
            except Exception as e:
                logger.warning(f"Upload verification failed: {e}")
    
    except Exception as e:
        logger.error(f"Error during dataset processing: {e}")
        raise

if __name__ == "__main__":
    main()
