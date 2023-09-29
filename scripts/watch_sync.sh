#!/bin/bash

set -o nounset
set -o pipefail
set -o errexit

while true; do
    find ./src -type f | entr -cds ./scripts/sync.sh
done
