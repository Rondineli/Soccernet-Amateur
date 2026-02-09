#!/usr/bin/env bash


set -ex

CWD_DIR=${1:-/OSL-ActionSpotting}
PATH_VIDEO=${2}
VIDEO_ID=${3}

export OMP_NUM_THREADS=4
export TF_NUM_INTRAOP_THREADS=4
export TF_NUM_INTEROP_THREADS=2

if [[ -z $2 || -z $3 ]]; then
    echo "Missing video feature paramater"
    exit 0
fi

OUTPUT="${VIDEO_ID}.npy"

. /root/miniconda3/etc/profile.d/conda.sh &&
conda activate oslactionspotting && 
python3 tools/features/extract_features.py \
    --path_video ${PATH_VIDEO} \
    --path_features /datasets/amateur/download/$VIDEO_ID/1_ResNet.npy \
    --PCA tools/features/pca_512_TF2.pkl \
    --PCA_scaler tools/features/average_512_TF2.pkl \
    --GPU 0