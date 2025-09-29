#!/bin/bash
set -e

python3.9 -m venv .venv/demucs
source .venv/demucs/bin/activate
pip install -r requirements/demucs.txt
deactivate

python3 -m venv .venv/encodec
source .venv/encodec/bin/activate
pip install -r requirements/encodec.txt
deactivate
