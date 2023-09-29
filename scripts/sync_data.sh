#!/bin/bash

set -o xtrace
set -o nounset
set -o pipefail
set -o errexit

rsync --verbose --archive --progress --rsh="ssh" ${HOST_PREFIX}0:~/data ./data

