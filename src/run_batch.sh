#!/bin/bash

set -o xtrace
set -o nounset
set -o pipefail

# 1. model name
MODEL_NAMES=$(cat models.txt | sed '/^# /d' | xargs)
# 2. batch sizes
BATCH_SIZES="1"
# 3. backends
BACKENDS="inductor null"

_work () {
    docker run --gpus all --network host \
        -v ~/data:/data \
        --env-file env \
        pytorch-cuda-experiments \
        --model_name $1 \
        --n_iter 50 \
        --batch_size $2 \
        --logdir /data \
        --backend $3
}
export -f _work

parallel -j 1 _work \
    ::: $MODEL_NAMES \
    ::: $BATCH_SIZES \
    ::: $BACKENDS 

