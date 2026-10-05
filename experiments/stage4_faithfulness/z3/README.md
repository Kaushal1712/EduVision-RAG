# Z3 — human verification set (DEV, unlabelled)

These labels will be the **reference set** for Stage 4. They will later be used to check how well
the Z2 alignment proxy and any future claim-level LLM judge agree with a human. Neither Z2 nor the
existing judge is ground truth.

## Files

| File | Use |
|---|---|
| `z3_annotation_sheet.md` | **Read this to annotate.** Per sample: question, prior conversation (follow-ups), full answer, the claim, its citation, and every evidence chunk given to the model — original text, unedited. |
| `z3_labels.csv` | **Fill in this one.** Columns `A_claim_support`, `B_citation_status`, `C_note` are empty. |
| `z3_metadata.csv` | Z2 scores, existing-judge output and selection reason per sample. **Open only after labelling**, so they do not anchor your labels. |
| `z3_annotation_set.json` | Everything above in one file (for later analysis). |

Regenerate with `venv/bin/python experiments/stage4_faithfulness/z3_prepare_annotation.py`
(deterministic; uses only existing DEV answers, Z1/Z2 outputs and the cached BGE-M3 model).

## How to annotate (about 1–1.5 minutes per sample)

Judge **only the claim** shown under "Claim to label", using **only** the evidence chunks shown
(E1–E5) and, for follow-ups, the prior conversation. Do not use your own knowledge of web development
or of the videos. Transcripts are machine-translated speech; ignore wording and grammar.

**A. Claim support** — is the claim's content stated in the evidence (or prior conversation)?
- `Supported` — everything substantive in the claim is stated there.
- `Partially supported` — some of it is stated, some is not (e.g. an added example, reason or detail).
- `Unsupported` — the evidence does not state it (even if it is true in general).
- `Contradicted` — the evidence says something incompatible with it.

**B. Citation status** — about the citation attached to this claim:
- `Correctly supports the claim` — the cited chunk (marked "← cited by this claim") supports the claim
  (fully or for its main point).
- `Points to evidence but does not support the claim` — the cited chunk is one of E1–E5 but does not
  support the claim (another chunk might).
- `No citation` — the claim has no citation.
- `Citation points outside provided evidence` — the citation matches none of E1–E5.
- `Cannot determine` — use sparingly, e.g. transcript too garbled to decide.

**C. Note** — optional, one short sentence (e.g. "adds a code example not in E2").

Use the exact label strings above. Label every sample; do not skip.

## Selection (39 samples, 38 queries)

All samples are factual sentences (≥ 4 words) from existing DEV answers: 38 from the frozen Stage 2.6
run (`answers_s26_cand_t044_guard_dev.json`) and 1 from the Stage 3 control re-run (EVB-054, the known
case). At most one sentence per query, except EVB-018 (two deliberately different sentences: its
uncited `<link>`-tag explanation and its misaligned code example). Order is shuffled (seed 2026) so
strata are not grouped.

| Primary reason | n | Rule |
|---|---:|---|
| Known Z1/Z2 cases | 6 | EVB-018 (×2), EVB-054, EVB-163, EVB-134, EVB-086 |
| Z2 misaligned | 4 | cited chunk ranked 2nd+ by both dense and lexical; largest margin first |
| Z2 aligned | 6 | cited chunk ranked 1st by both; spread over the cited-score range |
| Z2 mixed | 3 | ranked 1st by one signal only |
| Uncited factual sentence | 7 | spread over the best-evidence score range |
| Judge-flagged sentence | 5 | sentence matched to a judge `unsupported_claims` entry |
| Z2 aligned but judge-flagged | 2 | disagreement cases |
| Hinglish / follow-up top-up | 2 / 1 | to reach ≥ 5 Hinglish and ≥ 4 follow-ups |
| Controls | 3 | near-verbatim support (EVB-045); answer to an out-of-scope question (EVB-157); weakest best-evidence match (EVB-020) |

Coverage across all samples (a sample can count in several rows):

| Stratum | n |
|---|---:|
| Z2 misaligned / mixed / aligned / uncited | 8 / 6 / 14 / 11 |
| Judge-flagged sentence | 10 |
| Answer judged hallucinated | 8 |
| Z2–judge disagreement (Z2 misaligned but judge "fully supported", or Z2 aligned but judge-flagged) | 12 |
| Hinglish | 6 |
| Follow-up | 4 |
| Ordinary English (in-scope, not follow-up) | 29 |
| Out-of-scope question | 1 |

Notes: claims are machine-extracted sentences; markdown from the answer (e.g. `**`) can remain in the
claim text. The full answer is shown so the claim can be read in context. No label has been pre-filled.
