#!/bin/bash

# Note, this run script is specific to our local server.
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate bnelearndocs
cd ~/docs-bnelearn/_build/html
python -m http.server