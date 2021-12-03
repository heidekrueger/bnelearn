#!/bin/bash

source activate bnelearn-docs
cd ~/docs-bnelearn/_build/html
python -m http.server