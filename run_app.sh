#!/bin/bash

# Bedrock Chat Application Runner
# This script activates the Poetry environment and runs the Flask application

echo "🚀 Starting Bedrock Chat Application..."
echo "📦 Activating Poetry environment..."

# Change to the project directory
cd /home/dilusha/cml/polaris/bedrock_test/streamlit_test_chat_app

# Activate Poetry environment and run the application
source .venv/bin/activate && python app.py

echo "✅ Application stopped." 