#!/bin/bash

# GCS_SYNC_ARGS.SH
#
# Description: Synchronizes a Google Cloud Storage directory to a local directory
#              using 'gsutil rsync' in a continuous loop, taking configuration
#              from command-line arguments.
#
# Usage:
#    ./gcs_sync_args.sh <GCS_SOURCE> <LOCAL_DEST> [INTERVAL_SECONDS]
#

if [ "$#" -lt 2 ]; then
    echo "Usage: $0 <GCS_SOURCE> <LOCAL_DEST> [INTERVAL_SECONDS]"
    echo ""
    echo "  <GCS_SOURCE>      The Google Cloud Storage source path (must end with a slash)."
    echo "  <LOCAL_DEST>      The local destination path."
    echo "  [INTERVAL_SECONDS] Synchronization interval in seconds (Default: 300)."
    exit 1
fi

GCS_SOURCE="$1"
LOCAL_DEST="$2"
INTERVAL_SECONDS=${3:-1800}

sync_gcs_to_local() {
    echo "--- Starting synchronization at $(date '+%Y-%m-%d %H:%M:%S') ---"
    
    # Create the local destination if it doesn't exist
    if [ ! -d "$LOCAL_DEST" ]; then
        echo "Local destination '$LOCAL_DEST' does not exist. Creating it."
        mkdir -p "$LOCAL_DEST"
    fi

    echo "Syncing '$GCS_SOURCE' to '$LOCAL_DEST'..."
    
    gsutil -m rsync -r -d "$GCS_SOURCE" "$LOCAL_DEST"
    SYNC_STATUS=$?
    
    if [ $SYNC_STATUS -eq 0 ]; then
        echo "Synchronization successful."
    else
        echo "!!! ERROR: Synchronization failed with exit code $SYNC_STATUS." >&2
        echo "Please ensure 'gsutil' is authenticated and the path is correct." >&2
    fi
    
    echo "--- Synchronization finished. ---"
}


echo "Starting GCS Synchronization Monitor..."
echo "Source: $GCS_SOURCE"
echo "Destination: $LOCAL_DEST"
echo "Interval: $INTERVAL_SECONDS seconds."
echo ""

trap 'echo -e "\nExiting sync monitor..."; exit 0' SIGINT

while true; do
    sync_gcs_to_local
    
    echo "Waiting for $INTERVAL_SECONDS seconds before next sync..."
    sleep "$INTERVAL_SECONDS"
done
