#! /usr/bin/env bash

set -e
#conda deactivate  # make sure we're not in base environ
#module load miniconda/3.2021.10  # load miniconda module
#conda activate pytorch  # activate pytorch conda environment

python3 main.py --max_len 6 --distractors_num 0 --num_workers 1
