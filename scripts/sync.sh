#!/bin/bash

set -o xtrace
set -o nounset
set -o pipefail
set -o errexit

rsync --verbose --archive --progress --rsh="ssh" ./src ${HOST_PREFIX}0:

