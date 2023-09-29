#!/bin/bash

set -o xtrace
set -o nounset
set -o pipefail
set -o errexit

echo "set -o vi" >> ~/.bashrc
echo "export EDITOR=vim" >> ~/.bashrc

mkdir -p ~/huggingface
mkdir -p ~/src
