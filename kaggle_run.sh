#!/bin/bash
# Kaggle: run from a notebook cell as  !bash /kaggle/input/<dataset>/kaggle_run.sh <dataset>
# The dataset must contain: code/, models/, student_resource/ (auto-extracted from the zip).
set -e
DS=/kaggle/input/${1:?dataset folder name}
cd /kaggle/working
cp -r $DS/code $DS/models . 2>/dev/null || true
ln -sfn $DS/student_resource student_resource
nproc; free -g | head -2; ls student_resource/dataset/test
pip install -q -r code/business_entity_resolution/requirements.txt 2>&1 | grep -vi "warning" || true
cd code/business_entity_resolution/src
python -u pipeline.py test 2>&1 | grep --line-buffered -vE "Warning|eval_set"
ls -lh /kaggle/working/output
