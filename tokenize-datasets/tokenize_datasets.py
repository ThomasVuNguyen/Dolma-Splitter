#!/usr/bin/env python3
"""
Dataset Tokenizer Script

This script tokenizes a given dataset using a specified tokenizer and uploads
the result to Hugging Face Hub.
"""

import json
import argparse
import os
from pathlib import Path
from typing import Dict, Any
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer
from huggingface_hub import HfApi
import torch
from tqdm import tqdm


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from JSON file."""
    with open(config_path, 'r') as f:
        return json.load(f)


def load_tokenizer(tokenizer_name: str) -> AutoTokenizer:
    """Load the specified tokenizer."""
    print(f"Loading tokenizer: {tokenizer_name}")
    return AutoTokenizer.from_pretrained(tokenizer_name)


def load_source_dataset(dataset_name: str, subset: str = None) -> Dataset:
    """Load the source dataset."""
    print(f"Loading dataset: {dataset_name}")
    if subset:
        print(f"Subset: {subset}")
        dataset = load_dataset(dataset_name, subset)
    else:
        dataset = load_dataset(dataset_name)
    
    # Get the first split (usually 'train')
    split_name = list(dataset.keys())[0]
    return dataset[split_name]


def tokenize_dataset(dataset: Dataset, tokenizer: AutoTokenizer, 
                    text_column: str, output_column: str, sequence_length: int) -> Dataset:
    """Tokenize the dataset text column and filter by sequence length."""
    print(f"Tokenizing column '{text_column}' to '{output_column}' with sequence_length={sequence_length}")
    
    def tokenize_function(examples):
        # Handle both string and list inputs
        texts = examples[text_column]
        if isinstance(texts, str):
            texts = [texts]
        
        # Tokenize without padding or truncation initially
        tokenized = tokenizer(
            texts,
            padding=False,
            truncation=False,
            return_tensors=None  # Return lists instead of tensors
        )
        
        # Filter and process sequences
        valid_sequences = []
        for token_ids in tokenized["input_ids"]:
            if len(token_ids) >= sequence_length:
                # Truncate to exact sequence_length
                valid_sequences.append(token_ids[:sequence_length])
            # Ignore sequences shorter than sequence_length
        
        return {output_column: valid_sequences}
    
    # Apply tokenization
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        batch_size=1000,
        desc="Tokenizing and filtering dataset",
        remove_columns=dataset.column_names  # Remove all original columns
    )
    
    return tokenized_dataset


def upload_to_hub(dataset: Dataset, dataset_name: str, 
                  private: bool = False) -> None:
    """Upload the tokenized dataset to Hugging Face Hub."""
    print(f"Uploading to Hub: {dataset_name}")
    
    # Check if user is logged in
    api = HfApi()
    try:
        user = api.whoami()
        print(f"Logged in as: {user['name']}")
    except Exception as e:
        print("Not logged in to Hugging Face Hub")
        print("Please run: huggingface-cli login")
        return
    
    # Push to hub
    dataset.push_to_hub(
        dataset_name,
        private=private,
        commit_message="Add tokenized dataset"
    )
    print(f"Successfully uploaded to: https://huggingface.co/datasets/{dataset_name}")


def main():
    parser = argparse.ArgumentParser(description="Tokenize a dataset and upload to HF Hub")
    parser.add_argument("config_file", help="Path to JSON configuration file")
    parser.add_argument("--private", action="store_true", help="Make dataset private")
    parser.add_argument("--dry-run", action="store_true", help="Don't upload to hub")
    
    args = parser.parse_args()
    
    # Load configuration
    try:
        config = load_config(args.config_file)
    except FileNotFoundError:
        print(f"Error: Configuration file '{args.config_file}' not found")
        return 1
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in configuration file: {e}")
        return 1
    
    # Validate required fields
    required_fields = ["dataset", "column", "tokenizer", "output_column", "output_dataset", "sequence_length"]
    for field in required_fields:
        if field not in config:
            print(f"Error: Missing required field '{field}' in configuration")
            return 1
    
    try:
        # Load tokenizer
        tokenizer = load_tokenizer(config["tokenizer"])
        
        # Load source dataset
        source_dataset = load_source_dataset(
            config["dataset"], 
            config.get("subset")
        )
        
        print(f"Dataset loaded with {len(source_dataset)} examples")
        print(f"Columns: {list(source_dataset.column_names)}")
        
        # Tokenize dataset
        tokenized_dataset = tokenize_dataset(
            source_dataset,
            tokenizer,
            config["column"],
            config["output_column"],
            config["sequence_length"]
        )
        
        print(f"Tokenization complete. Dataset shape: {tokenized_dataset.shape}")
        
        # Show sample
        print("\nSample tokenized example:")
        sample = tokenized_dataset[0]
        print(f"Tokenized sequence length: {len(sample[config['output_column']])}")
        print(f"First 20 tokens: {sample[config['output_column']][:20]}")
        print(f"Last 20 tokens: {sample[config['output_column']][-20:]}")
        
        # Upload to hub (unless dry run)
        if not args.dry_run:
            upload_to_hub(
                tokenized_dataset, 
                config["output_dataset"],
                private=args.private
            )
        else:
            print("Dry run mode - not uploading to hub")
            
    except Exception as e:
        print(f"Error during processing: {e}")
        return 1
    
    print("Tokenization process completed successfully!")
    return 0


if __name__ == "__main__":
    exit(main())
