#!/usr/bin/env bash

CONFIG=$1
GPUS=$2
PORT=${PORT:-28509}

ROOT_DIR=$(cd $(dirname $0)/.. && pwd)
export PYTHONPATH=$ROOT_DIR:$PYTHONPATH
echo "Using PYTHONPATH=$PYTHONPATH"
python -m debugpy --listen 0.0.0.0:9999 --wait-for-client -m torch.distributed.run \
    --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/train.py $CONFIG --launcher pytorch ${@:3} --deterministic
