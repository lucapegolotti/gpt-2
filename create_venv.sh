#!/bin/sh

# Name of the virtual environment directory
ENV_NAME="gpt2-venv"

# Path to Python interpreter (optional)
PYTHON_BIN="python3"

# Check if virtualenv is installed
if ! command -v virtualenv >/dev/null 2>&1; then
    echo "virtualenv is not installed. Installing it now..."
    $PYTHON_BIN -m pip install --user virtualenv
fi

# Create the virtual environment
virtualenv -p $PYTHON_BIN $ENV_NAME

# Activate the virtual environment (for interactive shells, source this manually)
echo "To activate the virtual environment, run:"
echo ". $ENV_NAME/bin/activate"
