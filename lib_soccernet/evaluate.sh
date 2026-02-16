#!/bin/bash
set -ex
CONFIG_MODEL="${1:-configs/contextawarelossfunction/json_soccernet_calf_resnetpca512_amateur_model.py}"
MAX_EPOCHS=${2:-10}
METRIC=${4:-loose}
ANNOTATION=${3:-/datasets/amateur/test_annotations.json}

RESULT_DIR="/tmp/osl_eval"
mkdir -p "$RESULT_DIR" || echo "path /tmp/osl_eval already exists"

CONFIG_NAME=$(basename "$CONFIG_MODEL" .py)


rm -rf outputs/contextawarelossfunction/json_soccernet_calf_resnetpca512/results_spotting_test* || echo "ok"
rm -rf outputs/contextawarelossfunction/json_soccernet_calf_resnetpca512_amateur_model/results_spotting_test* || echo "ok"
rm -rf outputs/contextawarelossfunction/json_soccernet_calf_resnetpca512_amateur_model_st_2/results_spotting_test* || echo "ok"

rm -rf outputs/contextawarelossfunction/json_soccernet_calf_resnetpca512_amateur_model_no_tf/results_spotting_test* || echo "ok"

. /root/miniconda3/etc/profile.d/conda.sh && \
    conda activate osl-action-spotting

# Evaluate only test datasource features
python tools/infer.py "$CONFIG_MODEL" \
  --cfg-options \
    dataset.test.data_root=/workspace/datasets/amateur-dataset/ \
    dataset.test.path=$ANNOTATION

# run evaluation to retrive mAP
python tools/evaluate.py "$CONFIG_MODEL" \
  --cfg-options \
    dataset.test.data_root=/workspace/datasets/amateur-dataset/ \
    dataset.test.path=$ANNOTATION \
    dataset.test.metric=$METRIC