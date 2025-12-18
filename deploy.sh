#!/bin/bash
# Deployment script for FMMFOREX web interface

set -e

echo "=== FMMFOREX Deployment Script ==="
echo ""

# Check Python version
echo "Checking Python version..."
python --version

# Install dependencies
echo ""
echo "Installing dependencies..."
pip install -r requirements.txt

# Create necessary directories
echo ""
echo "Creating data directory..."
mkdir -p data

# Run tests
echo ""
echo "Running tests..."
python -m pytest tests/ -v

echo ""
echo "=== Deployment Complete ==="
echo ""
echo "To start the web interface, run:"
echo "  python -m src.cli web"
echo ""
echo "The web interface will be available at http://localhost:5000"
echo ""
echo "To run a backtest from CLI:"
echo "  python -m src.cli download-data"
echo "  python -m src.cli backtest --policy heuristic"
echo ""
