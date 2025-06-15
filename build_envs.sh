#!/bin/bash

conda env create -f demucs.yml
conda activate demucs
pip install -U git+https://github.com/facebookresearch/demucs#egg=demucs
conda deactivate

conda env create -f encodec.yml
