#!/bin/bash

# Quick start script for GLD Price Prediction Application

echo "=========================================="
echo "GLD Price Prediction - Quick Start"
echo "=========================================="
echo ""

# Check if dependencies are installed
if ! python -c "import streamlit" 2>/dev/null; then
    echo "Installing dependencies..."
    pip install -r requirements.txt
    echo ""
fi

# Run the Streamlit app
echo "Starting Streamlit application..."
echo ""
echo "The app will open in your browser automatically."
echo "If not, navigate to: http://localhost:8501"
echo ""
echo "Press Ctrl+C to stop the server."
echo ""

streamlit run app.py
