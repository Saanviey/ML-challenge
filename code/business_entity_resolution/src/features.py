"""Pairwise features for (s1 record, other record) candidate pairs. Vectorised (rapidfuzz cpdist), float32, chunked."""
import re, numpy as np, pandas as pd
from rapidfuzz import fuzz, process
from rapidfuzz.distance import JaroWinkler
from common import norm_name, norm_addr

_num = re.compile(r"\d+")
CHUNK = 1_000_000

def _nums(s): return set(_num.findall(s))

def _jacc(a, b):
    if not a and not b: return -1.0
    if not a or not b: return 0.0
    return len(a & b) / len(a | b)

def _cp(scorer, x, y):
    return process.cpdist(x, y, scorer=scorer, workers=-1, dtype=np.float32)

def _string_feats(n1, n2, a1, a2, oid):
    f = {}
    f["n_tsr"] = _cp(fuzz.token_set_ratio, n1, n2)
    f["n_tsort"] = _cp(fuzz.token_sort_ratio, n1, n2)
    f["n_partial"] = _cp(fuzz.partial_ratio, n1, n2)
    f["n_jw"] = _cp(JaroWinkler.similarity, n1, n2)
    t1 = [x.split() for x in n1]; t2 = [y.split() for y in n2]
    f["n_first_tok"] = np.fromiter((float(bool(x) and bool(y) and x[0] == y[0]) for x, y in zip(t1, t2)), np.float32, len(n1))
    f["n_tok_jacc"] = np.fromiter((_jacc(set(x), set(y)) for x, y in zip(t1, t2)), np.float32, len(n1))
    f["n_len1"] = np.fromiter((len(x) for x in n1), np.float32, len(n1)); f["n_len2"] = np.fromiter((len(x) for x in n2), np.float32, len(n2))
    f["a_tsr"] = _cp(fuzz.token_set_ratio, a1, a2)
    f["a_tsort"] = _cp(fuzz.token_sort_ratio, a1, a2)
    f["a_partial"] = _cp(fuzz.partial_ratio, a1, a2)
    f["a_tok_jacc"] = np.fromiter((_jacc(set(x.split()), set(y.split())) for x, y in zip(a1, a2)), np.float32, len(a1))
    N1 = [_nums(x) for x in a1]; N2 = [_nums(y) for y in a2]
    f["a_num_jacc"] = np.fromiter((_jacc(x, y) for x, y in zip(N1, N2)), np.float32, len(a1))
    f["a_num_sub"] = np.fromiter((float(x <= y or y <= x) if (x and y) else -1.0 for x, y in zip(N1, N2)), np.float32, len(a1))
    f["a_empty"] = np.fromiter((float(not x or not y) for x, y in zip(a1, a2)), np.float32, len(a1))
    f["a_len2"] = np.fromiter((len(x) for x in a2), np.float32, len(a2))
    f["src"] = np.fromiter((int(o[1]) for o in oid), np.float32, len(oid))
    return pd.DataFrame(f)

def string_features(pairs: pd.DataFrame, s1: pd.DataFrame, other: pd.DataFrame) -> pd.DataFrame:
    """pairs has s1_idx, o_idx (positions into s1/other), sim_*. Returns pairs + per-pair string features (no rank feats)."""
    an = s1.business_name.map(norm_name).astype("string[pyarrow]"); aa = s1.business_address.map(norm_addr).astype("string[pyarrow]")
    bn = other.business_name.map(norm_name).astype("string[pyarrow]"); ba = other.business_address.map(norm_addr).astype("string[pyarrow]")
    oid = other.entity_id
    pairs = pairs.reset_index(drop=True)
    parts = []
    for s in range(0, len(pairs), CHUNK):
        si = pairs.s1_idx.values[s:s+CHUNK]; oi = pairs.o_idx.values[s:s+CHUNK]
        parts.append(_string_feats(an.iloc[si].fillna("").tolist(), bn.iloc[oi].fillna("").tolist(),
                                   aa.iloc[si].fillna("").tolist(), ba.iloc[oi].fillna("").tolist(), oid.iloc[oi].tolist()))
    f = pd.concat([pairs, pd.concat(parts, ignore_index=True)], axis=1)
    for c in ("sim_name", "sim_addr", "sim_both"):
        f[c] = f[c].fillna(0.0).astype(np.float32)
    return f

def rank_features(f: pd.DataFrame) -> pd.DataFrame:
    """Competition features; must run over the COMPLETE pair set (all sources) so groups are whole."""
    f["score0"] = (f.sim_both + 0.01 * (f.n_tsr + f.a_tsr)).astype(np.float32)
    g = f.groupby("o_idx").score0
    f["o_rank"] = g.rank(ascending=False, method="first").astype(np.float32)
    f["o_margin"] = (f.score0 - g.transform("max")).astype(np.float32)
    f["o_ncand"] = g.transform("size").astype(np.float32)
    g = f.groupby("s1_idx").score0
    f["s_rank"] = g.rank(ascending=False, method="first").astype(np.float32)
    f["s_margin"] = (f.score0 - g.transform("max")).astype(np.float32)
    f["s_ncand"] = g.transform("size").astype(np.float32)
    return f

def add_features(pairs, s1, other):
    return rank_features(string_features(pairs, s1, other))

FEATS = ["sim_name", "sim_addr", "sim_both", "n_tsr", "n_tsort", "n_partial", "n_jw", "n_first_tok", "n_tok_jacc",
         "n_len1", "n_len2", "a_tsr", "a_tsort", "a_partial", "a_tok_jacc", "a_num_jacc", "a_num_sub", "a_empty", "a_len2",
         "src", "score0", "o_rank", "o_margin", "o_ncand", "s_rank", "s_margin", "s_ncand"]
