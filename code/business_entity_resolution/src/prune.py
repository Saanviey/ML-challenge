"""Prune candidates: each S2/S3 record keeps only its top-m S1 parents by a cheap fused score."""
import numpy as np, pandas as pd

def fused(w: pd.DataFrame) -> pd.Series:
    s = w[["sim_name", "sim_addr", "sim_both"]].fillna(0.0)
    return s.sim_both + 0.5 * s.sim_name + 0.5 * s.sim_addr

def prune(w: pd.DataFrame, m: int, min_score: float = 0.0) -> pd.DataFrame:
    w = w.assign(fs=fused(w))
    w["r"] = w.groupby("oid").fs.rank(ascending=False, method="first")
    return w[(w.r <= m) & (w.fs >= min_score)].drop(columns=["r"])

if __name__ == "__main__":
    from common import DEV, gt_to_dict
    from score import blocking_recall
    w = pd.read_parquet(f"{DEV}/cands_s3.parquet"); gt = pd.read_parquet(f"{DEV}/gt.parquet"); s3 = pd.read_parquet(f"{DEV}/s3.parquet")
    gtd = gt_to_dict(gt); avail = set(s3.entity_id)
    for m in (1, 2, 3, 5, 10):
        for ms in (0.0, 0.3, 0.5):
            p = prune(w, m, ms)
            r = blocking_recall(p.groupby("s1_id").oid.agg(set).to_dict(), gtd, avail)
            print(f"top-{m:2d} min={ms:.1f}  recall={r['pair_recall']:.4f}  avg_cands/S1={r['avg_cands']:.2f}  pairs={r['total_pairs']:,}")
