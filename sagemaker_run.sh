#!/bin/bash
# Run on a SageMaker JupyterLab terminal AFTER uploading: this bundle + student_resource zip + train_source2.tsv
# Usage: bash sagemaker_run.sh
set -e
cd ~
nproc; free -g | head -2
[ -d student_resource ] || unzip -q student_resource-*.zip
[ -f train_source2.tsv ] && mv -f train_source2.tsv student_resource/dataset/train/ || true
ls -lh student_resource/dataset/test student_resource/dataset/train
python3 -m venv .venv && . .venv/bin/activate
pip install -q -r code/business_entity_resolution/requirements.txt
cd code/business_entity_resolution/src
# model already trained on the Mac (models/lgb.txt); skip `train` unless you want to retrain here (~5 min):
#   python -u pipeline.py train 2>&1 | grep -vE "Warning|eval_set"
nohup python -u pipeline.py test > ../../../test.log 2>&1 &
echo "started. follow with:  tail -f ~/test.log"
