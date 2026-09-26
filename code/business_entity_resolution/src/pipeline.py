"""End-to-end driver.
  python pipeline.py train        -> trains the matcher on the dev split (data/dev, both sources) -> models/
  python pipeline.py test [thr]   -> blocks + matches the test set -> output/matching_results.tsv, candidate_pairs.tsv
"""
import sys, os, json, time, gc, numpy as np, pandas as pd
from common import ROOT, OUT, DEV, DATA, read_tsv, gt_to_dict, invert_gt
from block import block   # NOTE: sparse_dot_topn must be imported before lightgbm (OpenMP clash on macOS)
from features import string_features, rank_features, FEATS
from match import decide
from score import macro_f05, blocking_recall

MODELS = os.path.join(ROOT, "models"); os.makedirs(MODELS, exist_ok=True); os.makedirs(OUT, exist_ok=True)
TOPM = 2

def train(seed=0):
    t0 = time.time()
    s1 = pd.read_parquet(f"{DEV}/s1.parquet"); gt = pd.read_parquet(f"{DEV}/gt.parquet")
    parts, ids, off = [], [], 0
    for n in ("s2", "s3"):
        o = pd.read_parquet(f"{DEV}/{n}.parquet")
        w = block(s1, o, TOPM); w["o_idx"] += off; off += len(o)
        parts.append(string_features(w, s1, o)); ids.append(o.entity_id.values); del o, w; gc.collect()
    f = rank_features(pd.concat(parts, ignore_index=True)); del parts
    o_ids = np.concatenate(ids); s1_ids = s1.entity_id.values
    gtd = gt_to_dict(gt); inv = invert_gt(gtd); avail = set(o_ids.tolist())
    tmp = pd.DataFrame({"s1_id": s1_ids[f.s1_idx.values], "oid": o_ids[f.o_idx.values]})
    print("blocking:", blocking_recall(tmp.groupby("s1_id").oid.agg(set).to_dict(), gtd, avail), flush=True); del tmp
    f["y"] = (pd.Series(o_ids[f.o_idx.values]).map(inv).fillna("").values == s1_ids[f.s1_idx.values]).astype(int)
    print(f"features: {len(f):,} pairs, pos={int(f.y.sum()):,} ({time.time()-t0:.0f}s)", flush=True)
    rng = np.random.default_rng(seed)
    va_idx = set(rng.choice(len(s1), len(s1) // 4, replace=False).tolist())
    m_va = f.s1_idx.isin(va_idx); tr = f[~m_va]; va = f[m_va]
    import lightgbm as lgb
    m = lgb.LGBMClassifier(n_estimators=1500, learning_rate=0.05, num_leaves=127, subsample=0.8, subsample_freq=1,
                           colsample_bytree=0.8, min_child_samples=50, verbose=-1, n_jobs=-1)
    m.fit(tr[FEATS], tr.y, eval_set=[(va[FEATS], va.y)], callbacks=[lgb.early_stopping(50, verbose=False)])
    p = m.predict_proba(va[FEATS])[:, 1]
    va_ids = set(s1_ids[list(va_idx)]); vt = {a: s for a, s in gtd.items() if a in va_ids}
    best = (0.5, -1.0)
    for thr in np.arange(0.3, 0.96, 0.05):
        r = macro_f05(decide(va, p, thr, s1_ids, o_ids), vt, avail)
        print(f"thr={thr:.2f} macroF05={r['macro_f05']:.4f} P={r['pair_precision']:.3f} R={r['pair_recall']:.3f}")
        if r["macro_f05"] > best[1]: best = (float(round(thr, 2)), r["macro_f05"])
    imp = sorted(zip(m.feature_importances_, FEATS), reverse=True)[:10]
    print("top features:", ", ".join(f"{n}={v}" for v, n in imp))
    m.booster_.save_model(f"{MODELS}/lgb.txt"); json.dump({"thr": best[0], "dev_f05": best[1], "trees": int(m.best_iteration_)}, open(f"{MODELS}/thr.json", "w"))
    print(f"saved model ({m.best_iteration_} trees), best thr={best[0]:.2f} dev macroF05={best[1]:.4f}  ({time.time()-t0:.0f}s)")

def id_lists(s1_idx: np.ndarray, o_idx: np.ndarray, o_ids: np.ndarray, n1: int) -> list:
    """Comma-joined sorted id list per S1 position (empty string when none), from integer pairs."""
    order = np.lexsort((o_idx, s1_idx)); s = s1_idx[order]; o = o_ids[o_idx[order]]
    bounds = np.flatnonzero(np.r_[True, s[1:] != s[:-1]]); ends = np.r_[bounds[1:], len(s)]
    out = [""] * n1
    for b, e in zip(bounds, ends): out[s[b]] = ",".join(sorted(o[b:e].tolist()))
    return out

def test(thr=None):
    t0 = time.time()
    s1 = read_tsv(f"{DATA}/test/test_source1.tsv"); n1 = len(s1)
    parts, ids, off = [], [], 0
    for n in ("source2", "source3"):
        o = read_tsv(f"{DATA}/test/test_{n}.tsv")
        w = block(s1, o, TOPM); w["o_idx"] += off; off += len(o)
        fw = string_features(w, s1, o); fw.to_parquet(f"{OUT}/pairs_{n}.parquet")
        print(f"{n}: {len(fw):,} candidate pairs with string features ({time.time()-t0:.0f}s)", flush=True)
        parts.append(fw); ids.append(o.entity_id.values); del o, w, fw; gc.collect()
    f = rank_features(pd.concat(parts, ignore_index=True)); del parts; gc.collect()
    o_ids = np.concatenate(ids); s1_ids = s1.entity_id.values
    print(f"features: {len(f):,} pairs ({time.time()-t0:.0f}s)", flush=True)
    import lightgbm as lgb
    booster = lgb.Booster(model_file=f"{MODELS}/lgb.txt")
    thr = float(thr) if thr is not None else json.load(open(f"{MODELS}/thr.json"))["thr"]
    p = booster.predict(f[FEATS].values.astype(np.float32))
    d = pd.DataFrame({"s1_idx": f.s1_idx.values, "o_idx": f.o_idx.values, "p": p})
    d = d.sort_values("p", ascending=False).drop_duplicates("o_idx"); d = d[d.p >= thr]
    m_lists = id_lists(d.s1_idx.values, d.o_idx.values, o_ids, n1)
    c_lists = id_lists(f.s1_idx.values, f.o_idx.values, o_ids, n1)
    pd.DataFrame({"source1_entity_id": s1_ids, "matched_entity_ids": m_lists}).to_csv(f"{OUT}/matching_results.tsv", sep="\t", index=False)
    pd.DataFrame({"source1_entity_id": s1_ids, "candidate_entity_ids": c_lists}).to_csv(f"{OUT}/candidate_pairs.tsv", sep="\t", index=False)
    print(f"wrote {n1:,} rows; matched pairs={len(d):,} avg/S1={len(d)/n1:.2f}; candidates={len(f):,} avg/S1={len(f)/n1:.2f}; thr={thr}  ({time.time()-t0:.0f}s)")

if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "train": train()
    elif cmd == "test": test(sys.argv[2] if len(sys.argv) > 2 else None)
