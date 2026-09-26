"""Train a LightGBM pair classifier on dev candidates, sweep decision threshold on macro F0.5."""
import sys, time, numpy as np, pandas as pd
from sparse_dot_topn import sp_matmul_topn  # must load before lightgbm (OpenMP clash on macOS)
import lightgbm as lgb
from common import DEV, gt_to_dict, invert_gt
from features import add_features, FEATS
from score import macro_f05

def decide(pairs: pd.DataFrame, prob: np.ndarray, thr: float, s1_ids, o_ids) -> dict:
    """Each S2/S3 record goes to at most one S1: its argmax, if prob >= thr. Returns s1_id -> set(oid)."""
    d = pd.DataFrame({"s1_idx": pairs.s1_idx.values, "o_idx": pairs.o_idx.values, "p": prob})
    d = d.sort_values("p", ascending=False).drop_duplicates("o_idx")
    d = d[d.p >= thr]
    d["s1_id"] = s1_ids[d.s1_idx.values]; d["oid"] = o_ids[d.o_idx.values]
    return d.groupby("s1_id").oid.agg(set).to_dict()

def train_eval(pairs, s1, other, gt, seed=0):
    t0 = time.time()
    gtd = gt_to_dict(gt); inv = invert_gt(gtd)
    f = add_features(pairs, s1, other)
    s1_ids = s1.entity_id.values; o_ids = other.entity_id.values
    f["s1_id"] = s1_ids[f.s1_idx.values]
    f["y"] = (pd.Series(o_ids[f.o_idx.values]).map(inv).fillna("").values == f.s1_id.values).astype(int)
    print(f"features: {len(f):,} pairs, pos={f.y.sum():,} ({time.time()-t0:.0f}s)", flush=True)
    # split by S1 entity so the two halves share no entities
    rng = np.random.default_rng(seed)
    ents = s1.entity_id.values; tr_ents = set(rng.choice(ents, len(ents) // 2, replace=False))
    tr = f[f.s1_id.isin(tr_ents)]; va = f[~f.s1_id.isin(tr_ents)]
    m = lgb.LGBMClassifier(n_estimators=600, learning_rate=0.05, num_leaves=63, subsample=0.8, subsample_freq=1,
                           colsample_bytree=0.8, min_child_samples=50, verbose=-1, n_jobs=8)
    m.fit(tr[FEATS], tr.y, eval_set=[(va[FEATS], va.y)], callbacks=[lgb.early_stopping(50, verbose=False)])
    p = m.predict_proba(va[FEATS])[:, 1]
    print(f"trained {m.best_iteration_} iters ({time.time()-t0:.0f}s)")
    avail = set(other.entity_id); truths = {a: s for a, s in gtd.items() if a not in tr_ents}
    best = None
    for thr in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95):
        r = macro_f05(decide(va, p, thr, s1_ids, o_ids), truths, avail)
        print(f"thr={thr:.2f}  macroF05={r['macro_f05']:.4f}  P={r['pair_precision']:.3f} R={r['pair_recall']:.3f}")
        if best is None or r["macro_f05"] > best[1]: best = (thr, r["macro_f05"])
    imp = sorted(zip(m.feature_importances_, FEATS), reverse=True)[:12]
    print("top features:", ", ".join(f"{n}={v}" for v, n in imp))
    print("BEST", best)
    return m, best

if __name__ == "__main__":
    s1 = pd.read_parquet(f"{DEV}/s1.parquet"); s3 = pd.read_parquet(f"{DEV}/s3.parquet"); gt = pd.read_parquet(f"{DEV}/gt.parquet")
    pairs = pd.read_parquet(f"{DEV}/cands_s3.parquet")
    print(f"pruned candidates: {len(pairs):,}")
    m, best = train_eval(pairs, s1, s3, gt)
    m.booster_.save_model(f"{DEV}/lgb_s3.txt")
