"""Fast blocking: char n-gram TF-IDF -> TruncatedSVD (dense) -> FAISS inner-product top-k, per country.
Channels: name, address, name+address. Scales to the full test set on a laptop.
"""
import sys, time, numpy as np, pandas as pd, faiss
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from common import norm_name, norm_addr

def embed(texts_fit, texts_all, dim, ngram=(2, 4), max_feat=400_000, fit_n=300_000, seed=0):
    v = TfidfVectorizer(analyzer="char_wb", ngram_range=ngram, min_df=2, max_features=max_feat, dtype=np.float32, sublinear_tf=True)
    rng = np.random.default_rng(seed)
    fit = texts_fit if len(texts_fit) <= fit_n else texts_fit[rng.choice(len(texts_fit), fit_n, replace=False)]
    v.fit(fit)
    svd = TruncatedSVD(dim, algorithm="randomized", n_iter=4, random_state=seed).fit(v.transform(fit))
    def tf(t):
        out = np.empty((len(t), dim), dtype=np.float32)
        for s in range(0, len(t), 200_000):
            out[s:s+200_000] = svd.transform(v.transform(t[s:s+200_000]))
        faiss.normalize_L2(out); return out
    return tf

def knn(index_vecs, query_vecs, k, exact=False):
    d = index_vecs.shape[1]
    if exact or len(index_vecs) < 50_000:
        idx = faiss.IndexFlatIP(d)
    else:
        nlist = int(4 * np.sqrt(len(index_vecs)))
        q = faiss.IndexFlatIP(d); idx = faiss.IndexIVFFlat(q, d, nlist, faiss.METRIC_INNER_PRODUCT)
        idx.train(index_vecs[np.random.default_rng(0).choice(len(index_vecs), min(len(index_vecs), 60 * nlist), replace=False)])
        idx.nprobe = max(8, nlist // 32)
    idx.add(index_vecs)
    D, I = idx.search(query_vecs, k)
    return D, I

def block(s1: pd.DataFrame, other: pd.DataFrame, k=10, dim=256, exact=False) -> pd.DataFrame:
    t0 = time.time()
    s1 = s1.assign(nn=s1.business_name.map(norm_name), na=s1.business_address.map(norm_addr))
    other = other.assign(nn=other.business_name.map(norm_name), na=other.business_address.map(norm_addr))
    s1["nb"] = s1.nn + " | " + s1.na; other["nb"] = other.nn + " | " + other.na
    print(f"normalised in {time.time()-t0:.0f}s", flush=True)
    outs = []
    for ctry in sorted(set(s1.country) | set(other.country)):
        a = s1[s1.country == ctry]; b = other[other.country == ctry]
        if a.empty or b.empty: continue
        for field, chan in (("nn", "name"), ("na", "addr"), ("nb", "both")):
            tf = embed(pd.concat([a[field], b[field]]).values, None, dim)
            A = tf(a[field].values); B = tf(b[field].values)
            D, I = knn(A, B, k, exact)
            r = np.repeat(np.arange(len(b)), k); c = I.ravel(); d = D.ravel(); ok = c >= 0
            outs.append(pd.DataFrame({"s1_id": a.entity_id.values[c[ok]], "oid": b.entity_id.values[r[ok]], "sim": d[ok], "chan": chan}))
            print(f"{ctry:8s} {chan:5s} pairs={ok.sum():>9,} {time.time()-t0:.0f}s", flush=True)
    p = pd.concat(outs, ignore_index=True)
    w = p.pivot_table(index=["s1_id", "oid"], columns="chan", values="sim", aggfunc="max").reset_index()
    for ch in ("name", "addr", "both"):
        if ch not in w: w[ch] = np.nan
    w = w.rename(columns={"name": "sim_name", "addr": "sim_addr", "both": "sim_both"})
    print(f"unique candidate pairs: {len(w):,}  ({time.time()-t0:.0f}s)")
    return w

if __name__ == "__main__":
    from common import DEV, gt_to_dict
    from score import blocking_recall
    src = sys.argv[1] if len(sys.argv) > 1 else "s3"
    s1 = pd.read_parquet(f"{DEV}/s1.parquet"); s3 = pd.read_parquet(f"{DEV}/{src}.parquet"); gt = pd.read_parquet(f"{DEV}/gt.parquet")
    k = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    dim = int(sys.argv[3]) if len(sys.argv) > 3 else 256
    w = block(s1, s3, k, dim)
    w.to_parquet(f"{DEV}/cands_{src}_faiss.parquet")
    gtd = gt_to_dict(gt); avail = set(s3.entity_id)
    print("ALL   ", blocking_recall(w.groupby("s1_id").oid.agg(set).to_dict(), gtd, avail))
    for ch in ("sim_name", "sim_addr", "sim_both"):
        c = w[w[ch].notna()].groupby("s1_id").oid.agg(set).to_dict()
        print(f"{ch:8s}", blocking_recall(c, gtd, avail))
