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
echo "2. Run: python split.py"
