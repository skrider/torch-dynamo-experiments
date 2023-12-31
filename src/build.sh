#!/bin/bash

set -o xtrace
set -o nounset
set -o pipefail
set -o errexit

export DOCKER_BUILDKIT=1 

docker build . --target base --tag cuda-pytorch-nightly

