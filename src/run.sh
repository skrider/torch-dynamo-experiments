#!/bin/bash

docker run -it --gpus all --network host \
    -v ~/src:/src \
    -v ~/huggingface:/root/huggingface \
    -v ~/data:/data \
    -p 7001:7001 \
    cuda-pytorch-nightly /bin/bash
