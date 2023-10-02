#!/bin/bash

docker run -it --gpus all --network host \
    -v ~/src:/src \
    -v ~/huggingface:/root/huggingface \
    -v ~/data:/data \
    -p 6006:6006 \
    cuda-pytorch-nightly /bin/bash
