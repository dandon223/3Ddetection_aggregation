#!/bin/bash
source .env

ENV_FILE=environment.yml
ENV_NAME=$(head -n 1 "$ENV_FILE" | sed -r 's/^name: //')

"$CONDA_EXE" env create -f "$ENV_FILE"
"$CONDA_EXE" env config vars set LD_PRELOAD="$LD_PRELOAD":$(cat ~/.conda/environments.txt | grep "/$ENV_NAME\$")/lib/libstdc++.so.6.0.29 -n "$ENV_NAME"
source "$CONDA_PREFIX/bin/activate" "$ENV_NAME"

python setup.py develop