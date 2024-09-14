#!/bin/bash

set -eo pipefail
shopt -s dotglob nullglob

# Define variables
URL="https://d8it4huxumps7.cloudfront.net/files/66e31d6ee96cd_student_resource_3.zip"
ZIP_FILE="66e31d6ee96cd_student_resource_3.zip"
EXTRACT_DIR="student_resource 3"

# Function to clean up
cleanup() {
    echo "Cleaning up..."
    rm -f "$ZIP_FILE"
    rm -rf "$EXTRACT_DIR" __MACOSX sample_data src
}

# Set trap to call cleanup function on exit
trap cleanup EXIT

# Download and unzip the file
echo "Downloading and unzipping file..."
wget -q "$URL" -O "$ZIP_FILE" && unzip -q "$ZIP_FILE" || { echo "Download or unzip failed. Exiting."; exit 1; }

# Move files from extracted directory to current directory
if [[ -d "$EXTRACT_DIR" ]]; then
    echo "Moving files from $EXTRACT_DIR..."
    mv "$EXTRACT_DIR"/* "$EXTRACT_DIR"/.[!.]* . 2>/dev/null || true
    rmdir "$EXTRACT_DIR" 2>/dev/null || echo "Note: Unable to remove '$EXTRACT_DIR'. It might not be empty."
else
    echo "Warning: '$EXTRACT_DIR' not found."
fi

# Move files from src folder to parent directory
if [[ -d src ]]; then
    echo "Moving files from src folder..."
    mv src/* src/.[!.]* . 2>/dev/null || true
    rmdir src 2>/dev/null || echo "Note: Unable to remove 'src' folder. It might not be empty."
else
    echo "Note: 'src' folder not found."
fi

echo "Setup complete. Current directory contents:"
ls -la
