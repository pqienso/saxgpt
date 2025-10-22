#!/bin/bash

set -e

echo "=========BEGINNING PIPELINE EXECUTION=========="

source .venv/demucs/bin/activate
python -m src.data.pipeline.1_download_audio "$@"
python -m src.data.pipeline.2_split_stems "$@"
deactivate

source .venv/encodec/bin/activate
python -m src.data.pipeline.3_create_metadata "$@"
python -m src.data.pipeline.4_clip_audio "$@"
python -m src.data.pipeline.5_augment_audio "$@"
python -m src.data.pipeline.6_tokenize "$@"
python -m src.data.pipeline.7_create_datasets "$@"
deactivate

echo "=========PIPELINE EXECUTION COMPLETE=========="
