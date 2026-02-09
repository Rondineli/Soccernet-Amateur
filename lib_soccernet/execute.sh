#!/usr/bin/env bash

set -ex

CONFIG_MODEL="${1:-configs/contextawarelossfunction/json_soccernet_calf_resnetpca512_amateur_model.py}"
DATASET_ROOT_DIR=${2:-/workspace/datasets/amateur-dataset/}

OUTPUT="configs_${CONFIG_MODEL##*/}"
OUTPUT="${OUTPUT%.py}.json"

OUTPUT=${3:-OUTPUT}

ANNOTATION_PATH=${4:-/datasets/amateur/test_annotations.json}

VIDEO_ID=${5:-""}


. /root/miniconda3/etc/profile.d/conda.sh && \
    conda activate osl-action-spotting && \
    # Evaluate only test datasource features
    python3 ./cal_infer.py "$CONFIG_MODEL" \
    --output $OUTPUT \
    --video-id $VIDEO_ID \
    --cfg-options \
        dataset.test.data_root=$DATASET_ROOT_DIR \
        dataset.test.path=$ANNOTATION_PATH
