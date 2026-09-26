"""Shared helpers: paths, loading, text normalisation, ground-truth parsing."""
import os, re
import pandas as pd
from anyascii import anyascii as unidecode

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
DATA = os.path.join(ROOT, "student_resource", "dataset")
DEV = os.path.join(ROOT, "data", "dev")
OUT = os.path.join(ROOT, "output")

def read_tsv(path):
    return pd.read_csv(path, sep="\t", keep_default_na=False, dtype=str)

def load_train():
    d = os.path.join(DATA, "train")
    s1 = read_tsv(f"{d}/train_source1.tsv")
    s2p = f"{d}/train_source2.tsv"
    s2 = read_tsv(s2p) if os.path.exists(s2p) else None
    s3 = read_tsv(f"{d}/train_source3.tsv")
    gt = read_tsv(f"{d}/train_ground_truth.tsv")
    return s1, s2, s3, gt

def load_test():
    d = os.path.join(DATA, "test")
    return (read_tsv(f"{d}/test_source1.tsv"),
            read_tsv(f"{d}/test_source2.tsv"),
            read_tsv(f"{d}/test_source3.tsv"))

def gt_to_dict(gt):
    """source1_entity_id -> set of matched ids (may be empty)."""
    return {a: set(x for x in l.split(",") if x) for a, l in zip(gt.source1_entity_id, gt.matched_entity_ids)}

def invert_gt(gtd):
    """matched id -> source1 id."""
    return {x: a for a, s in gtd.items() for x in s}

# ---- normalisation ----
_LEGAL = r"\b(inc|incorporated|llc|l l c|ltd|limited|pvt|private|co|corp|corporation|company|plc|llp|sarl|sas|sa|eurl|sci|gmbh|pc|p c|the)\b"
_ABBR = {
    "street": "st", "avenue": "ave", "road": "rd", "drive": "dr", "lane": "ln", "court": "ct",
    "boulevard": "blvd", "place": "pl", "circle": "cir", "highway": "hwy", "parkway": "pkwy",
    "terrace": "ter", "north": "n", "south": "s", "east": "e", "west": "w", "apartment": "apt",
    "suite": "ste", "floor": "fl", "building": "bldg", "number": "no", "rue": "r",
}

def basic(s: str) -> str:
    s = unidecode(s or "").lower()
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return " ".join(s.split())

def norm_name(s: str) -> str:
    s = basic(s)
    s = re.sub(_LEGAL, " ", s)
    return " ".join(s.split())

def norm_addr(s: str) -> str:
    s = basic(s)
    toks = [_ABBR.get(t, t) for t in s.split()]
    return " ".join(toks)

def tokset(s: str) -> str:
    return " ".join(sorted(set(s.split())))
