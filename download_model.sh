#!/bin/bash

# Google Drive file ID for the pre-trained model
FILE_ID="1HQSpgJSOE7nv6zJJ6WrvrEMEg9YM7dFo"
# Output directory and filename for the downloaded model
OUTPUT_DIR="."
OUTPUT_FILE="$OUTPUT_DIR/gpt2-model.pt"

# Check if 'gdown' is installed. It's a utility for downloading from Google Drive.
if ! command -v gdown &> /dev/null
then
    echo "gdown could not be found. Please install it using pip:"
    echo "pip install gdown"
    exit 1
fi

# Create the output directory if it does not already exist
mkdir -p "$OUTPUT_DIR"

echo "Downloading model checkpoint to $OUTPUT_FILE..."
# Use gdown to download the file by its ID and save it to the specified output path.
gdown --id "$FILE_ID" -O "$OUTPUT_FILE"

# Check the exit status of the previous command (gdown) to see if the download was successful.
if [ $? -eq 0 ]; then
    echo "Download complete!"
    echo "The model is saved at: $OUTPUT_FILE"
else
    echo "Error: Download failed. Please check your internet connection or verify the Google Drive file ID."
fi