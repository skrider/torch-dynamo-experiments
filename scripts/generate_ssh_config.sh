#!/bin/bash

set -o nounset
set -o pipefail
set -o errexit

HELP=$(cat << EOF
This script generates ssh config for accessing hosts in the configured cluster.
EOF
)

TF_OUT=$1

N_HOSTS=$(cat $TF_OUT | jq -r '.node_ids.value | length')

for i in $(seq 0 $(($N_HOSTS - 1))); do
	echo "Host $HOST_PREFIX$i"
	echo "	HostName $(cat $TF_OUT | jq -r ".node_ips.value[$i]")"
	echo "	User ubuntu"
	echo "	ProxyJump ec2-user@$(cat $TF_OUT | jq -r '.public_ip.value')"
	echo "	ForwardAgent yes"
done

