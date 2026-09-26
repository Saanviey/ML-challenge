"""Blocking: word uni+bigram TF-IDF (max_df prunes identity-free common words), exact top-k S1
neighbours for every S2/S3 record within its country, on three channels (name / address / both).
Channels are unioned on integer indices, each record keeps its top-m parents by a fused score.
Memory-safe: text kept as Arrow-backed pandas strings, per-country/channel corpora built lazily,
vectoriser fitted on a sample, query side transformed + searched in chunks.
"""
import sys, time, gc, numpy as np, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sparse_dot_topn import sp_matmul_topn
from common import norm_name, norm_addr

VEC = dict(analyzer="word", ngram_range=(1, 2), token_pattern=r"\S+", min_df=2, max_df=0.05)
K = {"name": 5, "addr": 5, "both": 10}
import os
FIT_N, CHUNK, THREADS, THR = 600_000, 200_000, max(1, os.cpu_count() or 8), 0.05

def make_vec(**kw):
    return TfidfVectorizer(dtype=np.float32, sublinear_tf=True, **{**VEC, **kw})

def topk(vec, texts_q: pd.Series, BT, k):
    rows, cols, vals = [], [], []
    for s in range(0, len(texts_q), CHUNK):
        A = vec.transform(texts_q.iloc[s:s+CHUNK])
        C = sp_matmul_topn(A, BT, top_n=k, threshold=THR, sort=True, n_threads=THREADS).tocoo()
        rows.append((C.row + s).astype(np.int32)); cols.append(C.col.astype(np.int32)); vals.append(C.data.astype(np.float32))
    return np.concatenate(rows), np.concatenate(cols), np.concatenate(vals)

def channel_text(nn: pd.Series, na: pd.Series, chan: str) -> pd.Series:
    return nn if chan == "name" else na if chan == "addr" else (nn + " | " + na)

def block(s1: pd.DataFrame, other: pd.DataFrame, topm=2, seed=0) -> pd.DataFrame:
    """Returns pruned candidate pairs as positions into s1/other: s1_idx, o_idx, sim_name, sim_addr, sim_both, fs."""
    t0 = time.time(); rng = np.random.default_rng(seed); n1 = len(s1)
    s1n = s1.business_name.map(norm_name).astype("string[pyarrow]"); s1a = s1.business_address.map(norm_addr).astype("string[pyarrow]")
    on = other.business_name.map(norm_name).astype("string[pyarrow]"); oa = other.business_address.map(norm_addr).astype("string[pyarrow]")
    print(f"normalised {n1:,} + {len(other):,} in {time.time()-t0:.0f}s", flush=True)
    s1c = s1.country.values; oc = other.country.values
    parts = []
    for ctry in sorted(set(s1c) | set(oc)):
        ai = np.flatnonzero(s1c == ctry); bi = np.flatnonzero(oc == ctry)
        if len(ai) == 0 or len(bi) == 0: continue
        keys, sims = [], {}
        for chan in ("name", "addr", "both"):
            ta = channel_text(s1n.iloc[ai], s1a.iloc[ai], chan); tb = channel_text(on.iloc[bi], oa.iloc[bi], chan)
            if len(ta) + len(tb) > FIT_N:
                pick = rng.choice(len(ta) + len(tb), FIT_N, replace=False)
                corpus = pd.concat([ta, tb], ignore_index=True).iloc[np.sort(pick)]
            else:
                corpus = pd.concat([ta, tb], ignore_index=True)
            v = make_vec().fit(corpus); del corpus
            BT = v.transform(ta).T.tocsr()
            r, c, val = topk(v, tb, BT, K[chan])
            key = bi[r].astype(np.int64) * n1 + ai[c]
            keys.append(key); sims[chan] = (key, val)
            print(f"{ctry:8s} {chan:5s} pairs={len(r):>11,} {time.time()-t0:.0f}s", flush=True)
            del v, BT, ta, tb, r, c; gc.collect()
        ukey = np.unique(np.concatenate(keys)); del keys
        cols = {}
        for chan, (key, val) in sims.items():
            col = np.zeros(len(ukey), np.float32); col[np.searchsorted(ukey, key)] = val; cols[chan] = col
        del sims
        fs = cols["both"] + 0.5 * cols["name"] + 0.5 * cols["addr"]
        oidx = (ukey // n1).astype(np.int64); sidx = (ukey % n1).astype(np.int64); del ukey
        order = np.lexsort((-fs, oidx))
        oidx, sidx, fs = oidx[order], sidx[order], fs[order]
        first = np.r_[True, oidx[1:] != oidx[:-1]]
        grp_start = np.maximum.accumulate(np.where(first, np.arange(len(oidx)), 0))
        keep = (np.arange(len(oidx)) - grp_start) < topm
        parts.append(pd.DataFrame({"s1_idx": sidx[keep].astype(np.int32), "o_idx": oidx[keep].astype(np.int32),
                                   "sim_name": cols["name"][order][keep], "sim_addr": cols["addr"][order][keep],
                                   "sim_both": cols["both"][order][keep], "fs": fs[keep]}))
        print(f"{ctry:8s} union={len(oidx):,} pruned={int(keep.sum()):,}  ({time.time()-t0:.0f}s)", flush=True)
        del cols, fs, oidx, sidx, order, first, grp_start, keep; gc.collect()
    w = pd.concat(parts, ignore_index=True)
    print(f"candidate pairs: {len(w):,}  ({time.time()-t0:.0f}s)", flush=True)
    return w

if __name__ == "__main__":
    from common import DEV, gt_to_dict
    from score import blocking_recall
    src = sys.argv[1] if len(sys.argv) > 1 else "s3"; topm = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    s1 = pd.read_parquet(f"{DEV}/s1.parquet"); o = pd.read_parquet(f"{DEV}/{src}.parquet"); gt = pd.read_parquet(f"{DEV}/gt.parquet")
    w = block(s1, o, topm); w.to_parquet(f"{DEV}/cands_{src}.parquet")
    gtd = gt_to_dict(gt); avail = set(o.entity_id.tolist())
    w["s1_id"] = s1.entity_id.values[w.s1_idx]; w["oid"] = o.entity_id.values[w.o_idx]
    print("PRUNED", blocking_recall(w.groupby("s1_id").oid.agg(set).to_dict(), gtd, avail))
