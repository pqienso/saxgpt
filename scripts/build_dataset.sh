#!/bin/bash

set -e

source .venv/demucs/bin/activate
python -m src.data.create_stems "$@"
deactivate

source .venv/encodec/bin/activate
python -m src.data.create_metadata "$@"
python -m src.data.create_codes "$@"
python -m src.data.create_datasets "$@"
deactivate
