#!/bin/bash

INPUT_DIR=$1

python resolvent/SVD_save.py $INPUT_DIR
python resolvent/DMD_save.py $INPUT_DIR
python resolvent/resolvent_analysis.py $INPUT_DIR
python resolvent/plot_gain.py $INPUT_DIR
python resolvent/plot_peaks.py $INPUT_DIR