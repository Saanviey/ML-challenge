"""Official metric: macro F0.5 per source-1 entity (singletons score 1.0 if predicted empty)."""
import numpy as np

def f05(pred: set, truth: set) -> float:
    if not truth and not pred: return 1.0
    if not truth or not pred: return 0.0
    tp = len(pred & truth)
    if tp == 0: return 0.0
    p = tp / len(pred); r = tp / len(truth)
    b2 = 0.25
    return (1 + b2) * p * r / (b2 * p + r)

def macro_f05(preds: dict, truths: dict, available_ids: set | None = None) -> dict:
    """preds/truths: s1_id -> set. If available_ids given, truth is restricted to ids
    present in the dev pool (so we don't punish for records we never had)."""
    scores, ps, rs = [], [], []
    for a, t in truths.items():
        if available_ids is not None: t = t & available_ids
        p = preds.get(a, set())
        scores.append(f05(p, t))
        if t and p:
            tp = len(p & t); ps.append(tp / len(p)); rs.append(tp / len(t))
    return {"macro_f05": float(np.mean(scores)), "n": len(scores),
            "pair_precision": float(np.mean(ps)) if ps else 0.0,
            "pair_recall": float(np.mean(rs)) if rs else 0.0}

def blocking_recall(cands: dict, truths: dict, available_ids: set | None = None) -> dict:
    tot = found = 0; sizes = []
    for a, t in truths.items():
        if available_ids is not None: t = t & available_ids
        c = cands.get(a, set()); sizes.append(len(c))
        tot += len(t); found += len(t & c)
    return {"pair_recall": found / max(tot, 1), "avg_cands": float(np.mean(sizes)), "total_pairs": int(np.sum(sizes))}

if __name__ == "__main__":
    assert f05(set(), set()) == 1.0
    assert f05({"a"}, set()) == 0.0
    assert abs(f05({"a", "b", "c"}, {"a", "b", "d"}) - 0.6667) < 1e-3  # README example
    print("score.py self-test ok")
