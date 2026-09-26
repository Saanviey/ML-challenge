# Business Entity Resolution — pipeline

Reproduces `output/matching_results.tsv` and `output/candidate_pairs.tsv` from the raw TSVs.

## Setup
```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
```
Expects the data at `../../student_resource/dataset/{train,test}/` (i.e. this folder sits next to `student_resource/`).

## Run
```bash
cd src
python pipeline.py train 400000   # trains the LightGBM matcher on a 400k-entity sample of train -> ../models/
python pipeline.py test           # blocks + matches the test set -> ../../output/*.tsv
cd ../../../student_resource && python3 utils/validate_submission.py --matching ../output/matching_results.tsv --candidate ../output/candidate_pairs.tsv --test-dir dataset/test
```

## Stages (src/)
| file | role |
|---|---|
| `common.py` | loading, text normalisation (ASCII fold, legal-suffix strip, street abbreviations) |
| `block_faiss.py` | candidate generation: char n-gram TF-IDF -> TruncatedSVD -> FAISS top-k, per country, on name / address / name+address |
| `prune.py` | each S2/S3 record keeps its top-2 S1 parents by fused similarity (this is the set in `candidate_pairs.tsv`) |
| `features.py` | pairwise string, numeric-token and rank/margin features |
| `match.py` | LightGBM pair classifier; one-parent-per-record decision with a threshold tuned on macro F0.5 |
| `pipeline.py` | end-to-end train / test driver, writes the two output files |
| `score.py` | the official macro F0.5 metric for local validation |
| `block.py`, `make_dev_split.py` | exact (slow) sparse blocker and dev-split builder used for development only |
