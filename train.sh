#!/bin/bash
set -x
CONFIG_MODEL="${1:-configs/contextawarelossfunction/json_soccernet_calf_resnetpca512_amateur_model.py}"
MAX_EPOCHS=${2:-10}
DATASET_ROOT_DIR=${3:-/workspace/datasets/amateur-dataset/}

RESULT_DIR="/tmp/osl_eval"
mkdir -p "$RESULT_DIR" || echo "path /tmp/osl_eval already exists"

CONFIG_NAME=$(basename "$CONFIG_MODEL" .py)

# train the model in lower epochs
python tools/train.py "$CONFIG_MODEL" \
  --cfg-options \
    dataset.train.data_root=$DATASET_ROOT_DIR \
    dataset.valid.data_root=$DATASET_ROOT_DIR \
    dataset.train.path=$DATASET_ROOT_DIR/train_amateur_annotations.json \
    dataset.valid.path=$DATASET_ROOT_DIR/valid_amateur_annotations.json \
    training.max_epochs=$MAX_EPOCHS

# test_annotations.json contains the test and challeng features in the amateur dataset
python tools/infer.py "$CONFIG_MODEL" \
  --cfg-options \
    dataset.test.data_root=$DATASET_ROOT_DIR \
    dataset.test.path=$DATASET_ROOT_DIR/test_annotations.json

# run evaluation to retrive mAP@10s
python tools/evaluate.py "$CONFIG_MODEL" \
  --cfg-options \
      dataset.test.data_root=$DATASET_ROOT_DIR \
      dataset.test.path=$DATASET_ROOT_DIR/test_annotations.json \
      dataset.test.metric=loose > "$RESULT_DIR/${CONFIG_NAME}_test_annotations_eval.txt"

# run evaluation to retrive mAP@5s
python tools/evaluate.py "$CONFIG_MODEL" \
  --cfg-options \
    dataset.test.data_root=$DATASET_ROOT_DIR \
    dataset.test.path=/test_annotations.json \
    dataset.test.metric=tight > "$RESULT_DIR/${CONFIG_NAME}_test_annotations_eval.txt"

# Evaluate only test datasource features
python tools/infer.py "$CONFIG_MODEL" \
  --cfg-options \
    dataset.test.data_root=$DATASET_ROOT_DIR \
    dataset.test.path=$DATASET_ROOT_DIR/test_amateur_annotations.json

# run evaluation to retrive mAP@10s
python tools/evaluate.py "$CONFIG_MODEL" \
  --cfg-options \
    dataset.test.data_root=$DATASET_ROOT_DIR \
    dataset.test.path=$DATASET_ROOT_DIR/test_amateur_annotations.json \
    dataset.test.metric=loose > "$RESULT_DIR/${CONFIG_NAME}_amateur_test_eval.txt"

# run evaluation to retrive mAP@5s
python tools/evaluate.py "$CONFIG_MODEL" \
  --cfg-options \
    dataset.test.data_root=$DATASET_ROOT_DIR \
    dataset.test.path=$DATASET_ROOT_DIR/test_amateur_annotations.json \
    dataset.test.metric=tight > "$RESULT_DIR/${CONFIG_NAME}_amateur_test_eval.txt"
