# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** [TEAM NAME]
**Team Members:** [MEMBER 1], [MEMBER 2], [MEMBER 3]
**Submission Date:** [DATE]

---

## 1. Executive Summary

A two-stage pipeline: exact character n-gram TF-IDF retrieval per country on three text channels, pruned to the top-2 parent entities per Source 2/3 record, followed by a LightGBM pair classifier with rank-and-margin features and a one-parent-per-record decision rule tuned directly on macro F0.5. The candidate set averages [X] records per Source 1 entity at [Y]% recall on a 200k-entity held-out split; validation macro F0.5 is [Z]. Everything runs on CPU, uses no external data, and is fully reproducible from the raw TSVs.

---

## 2. Methodology

### 2.1 Problem Analysis

Findings from EDA on the training set (2.21M S1, 5.03M S2, 5.29M S3):

- **Match structure.** Each S1 entity has 0 to 11 matches (mean 3.5, mode 3); 5.6% are singletons. About 25% of S2/S3 records (1.34M each) match nothing (distractors).
- **Country is exact.** 100% of true pairs share the S1 entity's country string. Country is treated as an open string label; France (test only) is handled by the same code path with no hard-coding.
- **Noise is mechanical, not semantic.** Injected typos (`Rbihdge` for `Ridge`), token reordering, legal-suffix moves/drops, street abbreviations (`Ave`/`Avenue`), state abbreviation vs full name (`TN`/`Tennessee`, `HR`/`Haryana`), native-script state names (தமிழ்நாடு, महाराष्ट्र), added plot numbers, diacritics, hashtag-style handles.
- **Address is the anchor.** Many true matches have unrelated names but identical addresses (`Premier Cashew` vs `Ectokor`, `Kunal Solutions LLP` vs `Synjax`). Exact normalised-address match is 95% precise but recovers only 6% of pairs, so fuzzy address retrieval is required.
- **Neither field alone decides.** 157k S1 rows share an address with another S1 (multi-tenant addresses); exact normalised-name match is only 59% precise. Blocking must combine both, and the matcher must resolve the competition.
- **Missing fields.** 3.3% of S3 addresses are empty; S1 addresses are never empty.

### 2.2 Solution Strategy

**Approach Type:** Blocking + Classifier
**Core Innovation:** Treating candidate generation as a per-record competition (each S2/S3 record belongs to at most one S1 entity) rather than a per-entity shortlist. Retrieval is wide (top-10 on three channels), then each record keeps only its top-2 parents by a fused score. This yields a very small candidate set with negligible recall loss, and the same competition structure is fed to the classifier as rank/margin features, which dominate feature importance.

---

## 3. Candidate Generation (Blocking)

- **Blocking keys used:** Character n-gram (`char_wb`, [NGRAM]) TF-IDF cosine similarity, sublinear TF, fitted per country. Three channels: normalised name, normalised address, name + address concatenated. Exact top-10 nearest neighbours per record via sparse matrix multiplication (`sparse_dot_topn`), no approximation. Union of channels, then per-record pruning to the top-2 S1 parents by fused score `sim_both + 0.5·sim_name + 0.5·sim_addr`.
- **Normalisation:** ASCII folding (anyascii, permissive licence), lowercase, non-alphanumerics to spaces, legal-suffix removal on names, street-word abbreviation mapping on addresses (incl. French `rue`).
- **Candidate pairs generated:** [TOTAL] pairs for [N_S1] test entities = [X] per entity (reduction ratio [R] vs the 17.3T naive pairs).
- **How you ensured true matches were not lost:** Measured pair recall against ground truth on a 200k-entity dev split built with the same distractor ratio as the full data. Per-channel and post-pruning recall:

| Stage (dev, Source 3) | Pair recall | Cands / S1 |
| --- | --- | --- |
| Name channel, top-10 | 87.4% | 24.0 |
| Address channel, top-10 | 90.8% | 23.2 |
| Name+address channel, top-10 | 98.9% | 24.0 |
| Union of three channels | 99.4% | 57.1 |
| Union, pruned to top-2 parents per record | 97.9% | 4.8 |
| Union, pruned to top-1 | 97.3% | 2.4 |

We also evaluated a dense alternative (TruncatedSVD-256 + FAISS IVF): union recall 96.8% but only 85.4% after pruning, because compression degrades the ranking of the true parent. It was rejected; the exact sparse path is kept.

---

## 4. Matching Model

**Features used (27):**
- Name features: blocking cosine (name channel), token_set_ratio, token_sort_ratio, partial_ratio, Jaro-Winkler, first-token equality, token Jaccard, lengths of both names.
- Address features: blocking cosine (address channel), token_set_ratio, token_sort_ratio, partial_ratio, token Jaccard, digit-run Jaccard (house numbers / PIN codes), digit-run subset flag, empty-address flag, address length.
- Other: combined-channel cosine; source id (2/3); fused score; per-record rank, margin to best and candidate count; per-entity rank, margin and candidate count.

**Model type:** LightGBM binary classifier ([N_TREES] trees, 127 leaves, lr 0.05, early stopping on an entity-disjoint validation fold). No country feature, so the model transfers unchanged to France.
**Decision rule:** each S2/S3 record is assigned to its argmax-probability S1 entity only; assignment kept if probability ≥ threshold.
**Threshold selection method:** sweep on macro F0.5 (singletons included) over the entity-disjoint validation fold; best at [THR].

---

## 5. Results & Error Analysis

- **F_0.5 Score (macro):** [Z] on the dev split ([N] entities, both sources). Pair precision [P], pair recall [R].
- **Public leaderboard:** [LB] ([DATE]).
- **Common false positives (wrong merges):** [FILL AFTER ERROR ANALYSIS]
- **Common false negatives (missed matches):** [FILL AFTER ERROR ANALYSIS] Dominated by blocking loss (2.1% of true pairs never enter the candidate set): records with both name and address heavily corrupted, and multi-tenant addresses where the true parent ranks 3rd or lower.

---

## 6. Conclusion

[FILL]

---

## Appendix

### A. Code Artefacts

`code/business_entity_resolution/src/`:

| file | role |
| --- | --- |
| `common.py` | loading, normalisation |
| `block.py` | per-country char n-gram TF-IDF exact top-k retrieval on three channels (chunked, memory-safe) |
| `prune.py` | top-2 parents per record (produces the `candidate_pairs.tsv` set) |
| `features.py` | 27 pairwise features (vectorised rapidfuzz) |
| `match.py` | LightGBM matcher, one-parent decision rule, threshold sweep |
| `pipeline.py` | entry point: `python pipeline.py train` then `python pipeline.py test` writes both output files |
| `score.py` | official macro F0.5 metric |
| `make_dev_split.py`, `block_faiss.py`, `bench_block.py` | dev-split builder, rejected dense blocker, vectoriser benchmark |

Reproduce: see `code/business_entity_resolution/README.md`.

### B. Additional Results

**Experiment log**

| Date | Change | Dev macro F0.5 | Cands/S1 | Public LB |
| --- | --- | --- | --- | --- |
| 2026-09-26 | Baseline: 3-channel TF-IDF top-10, top-2 prune, LightGBM, thr 0.6 (S3 only) | 0.9787 | 4.8 | — |

**Threshold sweep (dev, Source 3)**

| thr | macro F0.5 | P | R |
| --- | --- | --- | --- |
| 0.30 | 0.9773 | 0.992 | 0.985 |
| 0.50 | 0.9784 | 0.994 | 0.984 |
| 0.60 | 0.9787 | 0.995 | 0.983 |
| 0.80 | 0.9778 | 0.997 | 0.979 |
| 0.95 | 0.9695 | 0.999 | 0.965 |
