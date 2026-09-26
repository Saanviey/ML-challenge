"""Benchmark vectoriser configs for the exact sparse blocker: time vs recall, India 'both' channel on dev."""
import time, sys, numpy as np, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sparse_dot_topn import sp_matmul_topn
from common import DEV, gt_to_dict, norm_name, norm_addr
from score import blocking_recall

s1 = pd.read_parquet(f"{DEV}/s1.parquet"); s3 = pd.read_parquet(f"{DEV}/s3.parquet"); gt = pd.read_parquet(f"{DEV}/gt.parquet")
ctry = sys.argv[1] if len(sys.argv) > 1 else "India"
a = s1[s1.country == ctry]; b = s3[s3.country == ctry]
ta = (a.business_name.map(norm_name) + " | " + a.business_address.map(norm_addr)).values
tb = (b.business_name.map(norm_name) + " | " + b.business_address.map(norm_addr)).values
gtd = gt_to_dict(gt); avail = set(b.entity_id.tolist())
aids = set(a.entity_id.tolist())
gtd = {k: v & avail for k, v in gtd.items() if k in aids}

CONFIGS = {
    "char_wb 2-4 (current)": dict(analyzer="char_wb", ngram_range=(2, 4)),
    "char_wb 3-3":           dict(analyzer="char_wb", ngram_range=(3, 3)),
    "char_wb 3-4 max200k":   dict(analyzer="char_wb", ngram_range=(3, 4), max_features=200_000),
    "word 1-1":              dict(analyzer="word", ngram_range=(1, 1), token_pattern=r"\S+"),
    "word 1-2":              dict(analyzer="word", ngram_range=(1, 2), token_pattern=r"\S+"),
    "w12 maxdf.05":          dict(analyzer="word", ngram_range=(1, 2), token_pattern=r"\S+", max_df=0.05),
    "w12 maxdf.01":          dict(analyzer="word", ngram_range=(1, 2), token_pattern=r"\S+", max_df=0.01),
    "w12 maxdf.002":         dict(analyzer="word", ngram_range=(1, 2), token_pattern=r"\S+", max_df=0.002),
}
only = sys.argv[2] if len(sys.argv) > 2 else None
for name, cfg in CONFIGS.items():
    if only and not name.startswith(only): continue
    t0 = time.time()
    v = TfidfVectorizer(min_df=2, dtype=np.float32, sublinear_tf=True, **cfg).fit(np.concatenate([ta, tb]))
    A = v.transform(tb); B = v.transform(ta).T.tocsr(); t1 = time.time()
    rows, cols, vals = [], [], []
    for s in range(0, A.shape[0], 50_000):
        C = sp_matmul_topn(A[s:s+50_000], B, top_n=10, threshold=0.05, sort=True, n_threads=8).tocoo()
        rows.append(C.row + s); cols.append(C.col); vals.append(C.data)
    r, c, val = map(np.concatenate, (rows, cols, vals)); t2 = time.time()
    w = pd.DataFrame({"s1_id": a.entity_id.values[c], "oid": b.entity_id.values[r], "sim": val})
    rec10 = blocking_recall(w.groupby("s1_id").oid.agg(set).to_dict(), gtd)["pair_recall"]
    w["rk"] = w.groupby("oid").sim.rank(ascending=False, method="first")
    rec2 = blocking_recall(w[w.rk <= 2].groupby("s1_id").oid.agg(set).to_dict(), gtd)["pair_recall"]
    print(f"{name:24s} nnz/row={A.nnz/A.shape[0]:6.1f} vocab={len(v.vocabulary_):>8,}  vec={t1-t0:5.0f}s  matmul={t2-t1:5.0f}s  recall@10={rec10:.4f}  recall@top2={rec2:.4f}", flush=True)
