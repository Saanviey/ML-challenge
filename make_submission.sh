#!/bin/bash
# Build <team>_submission.zip in the layout the organizers require. Usage: ./make_submission.sh <team_name>
set -e
TEAM=${1:?team name}
cd "$(dirname "$0")"
python3 student_resource/utils/validate_submission.py --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir student_resource/dataset/test
rm -rf /tmp/sub && mkdir -p /tmp/sub/output /tmp/sub/code/business_entity_resolution
cp output/matching_results.tsv output/candidate_pairs.tsv /tmp/sub/output/
cp -r code/business_entity_resolution/src code/business_entity_resolution/README.md code/business_entity_resolution/requirements.txt /tmp/sub/code/business_entity_resolution/
rm -rf /tmp/sub/code/business_entity_resolution/src/__pycache__
cp Documentation_template.md /tmp/sub/
(cd /tmp/sub && zip -qr "$OLDPWD/${TEAM}_submission.zip" .)
ls -lh "${TEAM}_submission.zip" && unzip -l "${TEAM}_submission.zip"
