#!/bin/bash

# Activate virtual environment
echo "Activating virtual environment..."
source myenv/bin/activate

# Install required packages
echo "Installing required packages..."
pip install -r requirements.txt

echo ""
echo "Setup complete! Next steps:"
echo "1. Create a .env file with your Hugging Face token:"
echo "   cp env_template.txt .env"
echo "   # Edit .env and add your actual token"
echo "2. Run: python split.py [target_examples] [custom_dataset_name]"
echo "   Examples:"
echo "   - python split.py        # Default 1M examples"
echo "   - python split.py 1000000     # 1M examples dataset"
echo "   - python split.py 1000000 my-dataset     # Custom name"
echo "   - python split.py 5000000 ThomasTheMaker/pretokenized-dolma-5M     # Full repo path"
echo "   All datasets will be stored in ./datasets folder automatically"
