#!/usr/bin/env python3
"""
Dataset Splitter and Uploader
Downloads pico-lm/pretokenized-dolma dataset, copies first N parquet files to reach target size, and uploads to target repository
"""

import os
import sys
import json
import tempfile
import shutil
import time
from pathlib import Path
from typing import Iterator, Dict, Any, List
import logging
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    from datasets import load_dataset, Dataset, DatasetDict
    from huggingface_hub import HfApi, login, hf_hub_download, list_repo_files
except ImportError as e:
    logger.error(f"Required packages not installed: {e}")
    logger.info("Please install: pip install datasets huggingface_hub")
    sys.exit(1)

def check_disk_space(path: str, required_gb: float = 100.0) -> bool:
    """Check if there's enough disk space available"""
    try:
        statvfs = os.statvfs(path)
        free_bytes = statvfs.f_frsize * statvfs.f_bavail
        free_gb = free_bytes / (1024**3)
        logger.info(f"Available disk space at {path}: {free_gb:.2f} GB")
        return free_gb >= required_gb
    except Exception as e:
        logger.warning(f"Could not check disk space: {e}")
        return True  # Assume OK if we can't check

def log_system_info():
    """Log system information for performance analysis"""
    try:
        import psutil
        cpu_count = psutil.cpu_count()
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('.')
        
        logger.info("🖥️  System Information:")
        logger.info(f"   CPU cores: {cpu_count}")
        logger.info(f"   Memory: {memory.total / (1024**3):.1f} GB total, {memory.available / (1024**3):.1f} GB available")
        logger.info(f"   Disk: {disk.free / (1024**3):.1f} GB free")
        
        # Check if we're on SSD or HDD (rough estimate)
        if hasattr(psutil, 'disk_io_counters'):
            io_counters = psutil.disk_io_counters()
            logger.info(f"   Disk I/O: {io_counters.read_bytes / (1024**3):.1f} GB read, {io_counters.write_bytes / (1024**3):.1f} GB written")
            
    except ImportError:
        logger.info("📊 Install psutil for detailed system monitoring: pip install psutil")
    except Exception as e:
        logger.warning(f"Could not get system info: {e}")

def get_parquet_files_info(source_dataset: str, cache_dir: str, target_size_gb: float) -> List[Dict[str, Any]]:
    """Get information about parquet files in the dataset, downloading only what's needed"""
    logger.info("🔍 Getting parquet files information...")
    
    try:
        # List files in the repository
        api = HfApi()
        files = list_repo_files(repo_id=source_dataset, repo_type="dataset")
        
        # Filter for parquet files
        parquet_files = [f for f in files if f.endswith('.parquet')]
        logger.info(f"Found {len(parquet_files)} parquet files in total")
        logger.info(f"Will download files until we reach {target_size_gb} GB")
        
        # Get file sizes and metadata, but only download what we need
        parquet_info = []
        total_size_gb = 0.0
        
        for i, filename in enumerate(parquet_files):
            if total_size_gb >= target_size_gb:
                logger.info(f"🎯 Target size reached ({total_size_gb:.3f} GB), stopping downloads")
                break
                
            try:
                # Download file to get size
                local_path = hf_hub_download(
                    repo_id=source_dataset,
                    repo_type="dataset",
                    filename=filename,
                    cache_dir=cache_dir,
                    local_dir=cache_dir
                )
                
                file_size = os.path.getsize(local_path)
                file_size_gb = file_size / (1024**3)
                total_size_gb += file_size_gb
                
                parquet_info.append({
                    'filename': filename,
                    'local_path': local_path,
                    'size_bytes': file_size,
                    'size_gb': file_size_gb,
                    'index': i
                })
                
                logger.info(f"📁 File {i+1}: {filename} ({file_size_gb:.3f} GB) - Total: {total_size_gb:.3f} GB")
                
            except Exception as e:
                logger.warning(f"Could not get info for {filename}: {e}")
                continue
        
        logger.info(f"✅ Downloaded {len(parquet_info)} files, total size: {total_size_gb:.3f} GB")
        return parquet_info
        
    except Exception as e:
        logger.error(f"Error getting parquet files info: {e}")
        raise

def copy_parquet_files_to_target_size(parquet_info: List[Dict[str, Any]], target_size_gb: float, target_dir: str) -> List[str]:
    """Move already-downloaded parquet files to target directory"""
    logger.info(f"📋 Moving downloaded parquet files to target directory: {target_dir}")
    
    os.makedirs(target_dir, exist_ok=True)
    moved_files = []
    total_size_gb = 0.0
    
    for file_info in parquet_info:
        try:
            # Move file to target directory (more efficient than copy)
            source_path = file_info['local_path']
            target_path = os.path.join(target_dir, file_info['filename'])
            
            # Create subdirectories if needed
            os.makedirs(os.path.dirname(target_path), exist_ok=True)
            
            shutil.move(source_path, target_path)
            
            moved_files.append(file_info['filename'])
            total_size_gb += file_info['size_gb']
            
            logger.info(f"✅ Moved: {file_info['filename']} ({file_info['size_gb']:.3f} GB) - Total: {total_size_gb:.3f} GB")
            
        except Exception as e:
            logger.warning(f"Failed to move {file_info['filename']}: {e}")
            continue
    
    logger.info(f"📊 Final result: {len(moved_files)} files moved, total size: {total_size_gb:.3f} GB")
    return moved_files

def create_dataset_metadata(target_dir: str, source_dataset: str, target_examples: int):
    """Create necessary dataset metadata files"""
    logger.info("📝 Creating dataset metadata files...")
    
    try:
        # Create dataset_info.json
        dataset_info = {
            "builder_name": "parquet",
            "config_name": "default",
            "version": {"version_str": "0.0.0"},
            "splits": {
                "train": {
                    "name": "train",
                    "num_bytes": 0,  # Will be calculated
                    "num_examples": target_examples,
                    "shard_lengths": None,
                    "dataset_name": source_dataset
                }
            },
            "supervised_keys": None,
            "download_size": 0,
            "post_processing_size": None,
            "dataset_size": 0,
            "size_in_bytes": 0
        }
        
        # Calculate total size
        total_size = 0
        for filename in os.listdir(target_dir):
            if filename.endswith('.parquet'):
                file_path = os.path.join(target_dir, filename)
                total_size += os.path.getsize(file_path)
        
        dataset_info["download_size"] = total_size
        dataset_info["dataset_size"] = total_size
        dataset_info["size_in_bytes"] = total_size
        
        # Write dataset_info.json
        with open(os.path.join(target_dir, 'dataset_info.json'), 'w') as f:
            json.dump(dataset_info, f, indent=2)
        
        # Create dataset_state.json
        dataset_state = {
            "_data_files": [
                {
                    "path": filename,
                    "split": "train"
                }
                for filename in os.listdir(target_dir)
                if filename.endswith('.parquet')
            ]
        }
        
        with open(os.path.join(target_dir, 'dataset_state.json'), 'w') as f:
            json.dump(dataset_state, f, indent=2)
        
        logger.info("✅ Dataset metadata files created successfully")
        
    except Exception as e:
        logger.error(f"Error creating metadata files: {e}")
        raise

def main():
    """Main function to download, split, and upload dataset"""
    
    # Load environment variables
    load_dotenv()
    
    # Configuration
    source_dataset = "pico-lm/pretokenized-dolma"
    
    # Get target size in GB from command line argument or use default
    if len(sys.argv) > 1:
        try:
            target_size_gb = float(sys.argv[1])
            if target_size_gb <= 0:
                raise ValueError("Target size must be positive")
        except ValueError as e:
            logger.error(f"Invalid target size: {sys.argv[1]}. Please provide a positive number in GB.")
            logger.info("Usage: python split.py [target_size_gb] [custom_dataset_name]")
            logger.info("Example: python split.py 20.0")
            logger.info("Example: python split.py 20.0 my-custom-dolma-dataset")
            sys.exit(1)
    else:
        target_size_gb = 20.0  # Default: 20 GB
        logger.info(f"No target size specified, using default: {target_size_gb} GB")
    
    # Get custom dataset name or full repository path if specified
    custom_dataset_name = None
    full_repo_path = None
    
    if len(sys.argv) > 2:
        user_input = sys.argv[2]
        
        # Check if user provided full repository path (contains '/')
        if '/' in user_input:
            full_repo_path = user_input
            logger.info(f"Using full repository path: {full_repo_path}")
            # Extract just the dataset name for local folder
            custom_dataset_name = user_input.split('/')[-1]
            logger.info(f"Local folder will be: {custom_dataset_name}")
        else:
            custom_dataset_name = user_input
            logger.info(f"Using custom dataset name: {custom_dataset_name}")
    else:
        # Generate default dataset name based on size
        custom_dataset_name = f"pretokenized-dolma-{target_size_gb:.0f}GB"
        logger.info(f"Using default dataset name: {custom_dataset_name}")
    
    # Automatically use ./datasets folder for both cache and save
    base_datasets_dir = "./datasets"
    os.makedirs(base_datasets_dir, exist_ok=True)
    
    # Cache directory for Hugging Face processing
    custom_cache_dir = f"{base_datasets_dir}/cache"
    os.makedirs(custom_cache_dir, exist_ok=True)
    logger.info(f"Using cache directory: {custom_cache_dir}")
    
    # Local save directory for the final dataset
    local_save_dir = f"{base_datasets_dir}/{custom_dataset_name}"
    os.makedirs(local_save_dir, exist_ok=True)
    logger.info(f"Using local save directory: {local_save_dir}")
    
    # Update environment variables
    os.environ['HF_DATASETS_CACHE'] = custom_cache_dir
    os.environ['TRANSFORMERS_CACHE'] = custom_cache_dir
    
    # Check for Hugging Face token first
    hf_token = os.getenv('HUGGINGFACE_TOKEN')
    if not hf_token:
        logger.error("HUGGINGFACE_TOKEN not found in environment variables or .env file")
        logger.info("Please create a .env file with your Hugging Face token:")
        logger.info("HUGGINGFACE_TOKEN=your_token_here")
        logger.info("Optional: VERIFY_UPLOAD=true to verify upload by downloading (default: false)")
        sys.exit(1)
    
    # Get the actual username from Hugging Face API
    try:
        temp_api = HfApi(token=hf_token)
        user_info = temp_api.whoami()
        actual_username = user_info['name']
        logger.info(f"✅ Authenticated as: {actual_username}")
    except Exception as e:
        logger.warning(f"Could not get username from API: {e}")
        actual_username = "ThomastheMaker"  # Fallback
        logger.info(f"Using fallback username: {actual_username}")
    
    # Use the full repository path if provided, otherwise construct from username and dataset name
    if full_repo_path:
        target_dataset = full_repo_path
    else:
        target_dataset = f"{actual_username}/{custom_dataset_name}"
    
    logger.info(f"Starting dataset processing: {source_dataset} -> {target_dataset}")
    logger.info(f"Target size: {target_size_gb} GB")
    
    # Check disk space before proceeding
    logger.info("Checking available disk space...")
    if not check_disk_space(".", 200.0):  # Need at least 200GB in current directory
        logger.error("Insufficient disk space in current directory. Please free up space.")
        sys.exit(1)
    logger.info("✅ Sufficient disk space available")
    
    # Log system information for performance analysis
    log_system_info()
    
    try:
        # Step 1: Get information about parquet files
        logger.info(f"Analyzing parquet files in: {source_dataset}")
        parquet_info = get_parquet_files_info(source_dataset, custom_cache_dir, target_size_gb)
        
        if not parquet_info:
            logger.error("No parquet files found in the dataset")
            sys.exit(1)
        
        # Step 2: Move downloaded parquet files to target directory
        logger.info(f"Moving downloaded parquet files to target directory...")
        moved_files = copy_parquet_files_to_target_size(parquet_info, target_size_gb, local_save_dir)
        
        if not moved_files:
            logger.error("No files were moved successfully")
            sys.exit(1)
        
        # Step 3: Create dataset metadata files
        logger.info("Creating dataset metadata...")
        create_dataset_metadata(local_save_dir, source_dataset, len(moved_files))
        
        # Step 4: Initialize Hugging Face API with token
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
        
        # Step 5: Upload dataset from local storage
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
        
        # Upload the dataset from local storage
        try:
            api.upload_folder(
                folder_path=local_save_dir,
                repo_id=target_dataset,
                repo_type="dataset",
                commit_message=f"Initial upload: First {len(moved_files)} parquet files (~{target_size_gb:.1f} GB) of pretokenized-dolma dataset"
            )
            logger.info(f"✅ Dataset successfully uploaded to: {target_dataset}")
        except Exception as e:
            logger.error(f"Failed to upload dataset: {e}")
            logger.info(f"Dataset is still available locally at: {local_save_dir}")
            logger.info("You can retry the upload later or upload manually")
            raise
        
        # Step 6: Verify upload (optional - can be disabled to avoid re-downloading)
        verify_upload = os.getenv('VERIFY_UPLOAD', 'false').lower() == 'true'
        if verify_upload:
            try:
                uploaded_dataset = load_dataset(target_dataset)
                logger.info(f"✅ Upload verification successful. Dataset loaded successfully.")
            except Exception as e:
                logger.warning(f"Upload verification failed: {e}")
        else:
            logger.info("📝 Upload verification skipped (set VERIFY_UPLOAD=true in .env to enable)")
        
        # Final success message
        logger.info("🎉 Dataset processing completed successfully!")
        logger.info(f"Local copy saved at: {local_save_dir}")
        logger.info(f"Hugging Face dataset: {target_dataset}")
        logger.info(f"Files uploaded: {len(moved_files)} parquet files")
    
    except Exception as e:
        logger.error(f"Error during dataset processing: {e}")
        
        # Clean up temporary files if possible
        try:
            if 'custom_cache_dir' in locals():
                logger.info("Cleaning up temporary files...")
                shutil.rmtree(custom_cache_dir, ignore_errors=True)
        except Exception as cleanup_error:
            logger.warning(f"Could not clean up temporary files: {cleanup_error}")
        
        # Provide helpful error messages for common issues
        if "No space left on device" in str(e):
            logger.error("This error indicates insufficient disk space.")
            logger.error("Solutions:")
            logger.error("1. Free up disk space (at least 200GB recommended)")
            logger.error("2. Check if /tmp directory has enough space")
            logger.error("3. Consider using a different machine with more storage")
        
        raise

if __name__ == "__main__":
    main()
