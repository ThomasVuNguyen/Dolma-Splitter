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
echo "2. Run: python split.py [target_size_gb]"
echo "   Examples:"
echo "   - python split.py        # Default 20GB"
echo "   - python split.py 10     # 10GB dataset"
echo "   - python split.py 5.5    # 5.5GB dataset"
