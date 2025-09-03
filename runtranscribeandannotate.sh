#!/bin/bash

# --- Configuration ---
# Set the directory containing your audio files
TARGET_DIR=""

# Set the options for insanely-fast-whisper (optional, you can keep them inline)
MODEL_NAME="openai/whisper-large-v3"
DIARIZATION_MODEL="pyannote/speaker-diarization"
DEVICE_ID="mps"
# --- End Configuration ---

# Check if the target directory exists
if [ ! -d "$TARGET_DIR" ]; then
  echo "Error: Directory '$TARGET_DIR' not found."
  exit 1
fi

echo "Starting processing in directory: $TARGET_DIR"

# Find files (adjust -maxdepth and -type if needed)
# -maxdepth 1: Only find files directly in TARGET_DIR, not subdirectories
# -type f: Only find regular files
# -print0: Print filenames separated by null characters (safe for special chars)
find "$TARGET_DIR" -maxdepth 1 -type f -print0 | while IFS= read -r -d $'\0' audio_file; do

    # Construct the output transcript path
    transcript_path="${audio_file}.txt"

    echo "-----------------------------------------------------"
    echo "Processing: $audio_file"
    echo "Outputting to: $transcript_path"
    echo "-----------------------------------------------------"

    # Optional: Uncomment the block below to skip if the .txt file already exists
    # if [ -f "$transcript_path" ]; then
    #   echo "Skipping: Output file '$transcript_path' already exists."
    #   continue # Skip to the next file
    # fi

    # Run the command for the current file
    PYTORCH_ENABLE_MPS_FALLBACK=1 pdm run insanely-fast-whisper \
        --model-name "$MODEL_NAME" \
        --diarization_model "$DIARIZATION_MODEL" \
        --device-id "$DEVICE_ID" \
        --file-name "$audio_file" \
        --transcript-path "$transcript_path"

    # Check the exit status of the command
    if [ $? -ne 0 ]; then
        echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
        echo "Error processing file: $audio_file"
        echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
        # Decide if you want to stop on error or continue
        # exit 1 # Uncomment this line to stop the script on the first error
    fi
    echo # Add a newline for better separation in output

done

echo "====================================================="
echo "Finished processing all files in $TARGET_DIR"
echo "====================================================="
