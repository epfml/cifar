#!/bin/bash
set -e  # exit on error

# Install any missing dependencies that are not available in the base image yet.
pip install --user -r requirements.txt

# Run the training script.
python train.py
