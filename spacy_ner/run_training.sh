#!/bin/bash
TIMESTAMP=`date +%Y-%m-%d_%H-%M-%S`

if [[ $# -ne 3 ]] ; then
  echo 'Usage: run_training.sh <cuda> <data_path> <model>'
  exit 1
fi

cuda="$1"
data_path="$2"
model="$3"

python -m spacy project run train . --vars.gpu "$cuda" --vars.run_name "$TIMESTAMP-cuda-$cuda" --vars.data_path "$data_path" --vars.model "$model"
