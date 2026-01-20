#!/bin/bash

set -ex

DATASETDIR="${1:-/opt/projects/datasets/SoccerNet-Amateur/}"
USER="${2:-admin}"
PASS="${3:-4m4t3urS0cc3RN3t}"

mkdir -p $DATASETDIR || echo "Maybe $DATASETDIR already exists"

echo "Downloading dataset to $DATASETDIR"

#aws s3 cp s3://soccernet-v2-amateur/test.zip $DATASETDIR
#aws s3 cp s3://soccernet-v2-amateur/valid.zip $DATASETDIR
#aws s3 cp s3://soccernet-v2-amateur/train.zip $DATASETDIR
#aws s3 cp s3://soccernet-v2-amateur/challenge.zip $DATASETDIR
#aws s3 cp s3://soccernet-v2-amateur/annotations.zip $DATASETDIR

cd $DATASETDIR

wget --user $USER --password $PASS https://d3tdwb735roscv.cloudfront.net/test.zip
wget --user $USER --password $PASS https://d3tdwb735roscv.cloudfront.net/valid.zip
wget --user $USER --password $PASS https://d3tdwb735roscv.cloudfront.net/challenge.zip
wget --user $USER --password $PASS https://d3tdwb735roscv.cloudfront.net/train.zip
wget --user $USER --password $PASS https://d3tdwb735roscv.cloudfront.net/annotations.zip

for a in $(ls |grep .zip); do
    unzip $a
    rm -rf $a
done
