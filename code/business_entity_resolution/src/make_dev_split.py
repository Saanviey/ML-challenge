"""Build a realistic dev split from train.

Sample N source-1 entities. Keep every S2/S3 record whose truth is in the sample,
plus the same *fraction* of unmatched (distractor) S2/S3 rows so the candidate pool
has the same match/noise ratio as the full data.
"""
import sys, numpy as np, pandas as pd, os
from common import load_train, gt_to_dict, invert_gt, DEV

N = int(sys.argv[1]) if len(sys.argv) > 1 else 200_000
SEED = 0
os.makedirs(DEV, exist_ok=True)
s1, s2, s3, gt = load_train()
rng = np.random.default_rng(SEED)
keep = set(rng.choice(s1.entity_id.values, N, replace=False))
frac = N / len(s1)
gtd = gt_to_dict(gt); inv = invert_gt(gtd)

def sub(src, name):
    if src is None: return
    truth = src.entity_id.map(inv).fillna("")
    matched = truth.isin(keep)
    unmatched = (truth == "") & (rng.random(len(src)) < frac)
    out = src[matched | unmatched].reset_index(drop=True)
    out.to_parquet(f"{DEV}/{name}.parquet")
    print(name, len(out), "matched", int(matched.sum()), "distractors", int(unmatched.sum()))

s1[s1.entity_id.isin(keep)].reset_index(drop=True).to_parquet(f"{DEV}/s1.parquet")
gt[gt.source1_entity_id.isin(keep)].reset_index(drop=True).to_parquet(f"{DEV}/gt.parquet")
sub(s2, "s2"); sub(s3, "s3")
print("s1", N)
