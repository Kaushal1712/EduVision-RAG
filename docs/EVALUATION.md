# EduVision evaluation: journey and final metrics

This document records how the production configuration (**Stage 2.6**) was chosen, the final metrics, and
what the evaluation can and cannot support. Per-stage reports with full detail are linked below; they are
kept as history and describe the system as it was when each stage ran.

**Read this first.** Every label and grade in this evaluation was produced by AI models: the benchmark
queries and gold spans by an AI annotator (checked against Whisper transcriptions and video frames), the
answer grades by an LLM judge, and the Stage 4 verification labels by an AI annotator. **Nothing has been
verified by a person.** The numbers are useful for comparing configurations under the same procedure. They
are not human-validated accuracy figures.

---

## 1. Benchmark: eduvision-bench-v1.1

Files: `eval/benchmark/eduvision_bench_v1.json`, `eval/benchmark/corpus_manifest.json`. Schema, split rule
and annotation process: [eval/benchmark/README.md](../eval/benchmark/README.md).

| | DEV | TEST | Total |
|---|---:|---:|---:|
| In-scope queries (scored) | 60 | 92 | 152 |
| Out-of-scope: unrelated | 4 | 7 | 11 |
| Out-of-scope: near-miss (web-dev topic the course does not teach) | 5 | 8 | 13 |
| **All queries** | **69** | **107** | **176** |

The 176 queries include 24 Hinglish (Romanized Hindi) and 15 follow-ups with fixed history. Gold is given as
**timestamp spans** on the video timeline (`video_id`, `start_time`, `end_time`), not chunk IDs, so the same
gold applies to any transcription or chunking. A retrieved chunk is relevant when it overlaps a gold span.

Annotation status: all 176 items are `video_checked`, meaning an AI annotator checked them against
Whisper `medium` transcripts, a second-opinion Whisper `large-v2` pass and video frames. **0 items are
human-`verified`.**

**Split discipline.** DEV is used for every selection. TEST is held out and was never used to select
anything. TEST results were reported for the Stage 2 retrieval comparison, the Stage 2.5 answer evaluation
and the locked Stage 2.6 evaluation. No Stage 3 or Stage 4 variant was evaluated on TEST.

## 2. Metric definitions

**Retrieval** (`eval/run_benchmark.py`, `eval/bench_metrics.py`): Recall@k, MRR@10 and nDCG@10 over
in-scope queries, using span overlap as relevance. Paired comparisons (`eval/compare_results.py`) report
a seeded paired-bootstrap 95% CI.

**Answers** (`eval/answer_eval.py`): the full production `pipeline.ask()` with GPT-4o-mini, 2 generation
repeats per query. Each answer is graded by the judge `gpt-4.1-2025-04-14` (temperature 0, JSON output).

| Metric | Definition |
|---|---|
| Correct / correct-or-partial | Judge's correctness grade, measured against a **reference**: the Stage 1.5 Whisper `medium` annotation transcript of the gold spans (text from neither system under test) |
| Fully grounded | Share of answered in-scope responses the judge rates as fully supported by the evidence given to the model |
| Hallucination | Share of in-scope answers with a substantive claim supported by neither the evidence nor the reference |
| False refusal | In-scope question refused (no-LLM gate, or the answer contains "could not find") |
| OOS answered | Out-of-scope question answered instead of refused |
| Evidence has gold | Some chunk passed to the LLM overlaps a gold span |
| Citation inside gold span | A parsed `[Video: … @ MM:SS]` citation lies in a gold video within [gold start − 5 s, gold end + 5 s] |

## 3. Journey

| Stage | Question | Outcome | Report |
|---|---|---|---|
| 1 | Build a timestamp-anchored retrieval benchmark and baseline | v1.0: 176 queries, gold drafted from production transcripts | `eval/benchmark/README.md`, `eval/results/stage1_baseline_bench-v1.0.json` |
| 1.5 | Re-check every gold span against the videos | v1.1: gold now also covers regions where the production transcript was unusable (33% of gold time); frozen baseline | `eval/benchmark/README.md`, `eval/results/stage1_5_baseline_bench-v1.1.json` |
| 2 | Does stronger ASR improve retrieval? | Whisper `large-v2` translate (greedy) chosen on DEV; TEST R@10 0.711 → 0.899 | [experiments/stage2_transcription/README.md](../experiments/stage2_transcription/README.md) |
| 2.5 | Chunking variants, timestamp error, end-to-end answers | Chunking kept at 5/1/5 s. Answers: correct 63% → 79%, false refusals 15% → 3%, but **OOS answered 6.7% → 13.3%**, and 2/20 Hinglish in-scope queries were refused by the 0.50 gate | same README, `eval/results/stage2_5_*` |
| 2.6 | Fix the abstention regressions | Threshold 0.50 → **0.44** plus prompt **rule 8**; selection locked on DEV, then TEST run once | `experiments/stage2_6_abstention/locked_selection.json`, `eval/results/stage2_6_*` |
| 3 | Hybrid retrieval / reranking | Negative: no variant improved end-to-end answers; Stage 2.6 retained | [experiments/stage3_retrieval/README.md](../experiments/stage3_retrieval/README.md) |
| 4 | Citation integrity and claim-level faithfulness (offline) | Citations 100% valid; ~42% of factual sentences uncited; claim-level faithfulness not reliably measurable; nothing promoted | [experiments/stage4_faithfulness/README.md](../experiments/stage4_faithfulness/README.md) |
| Finalization | Wire Stage 2.6 into production | Index shipped, config/prompt switched, consistency test, rebuild safety, UI safety fixes | [experiments/finalization/FINALIZATION_AUDIT.md](../experiments/finalization/FINALIZATION_AUDIT.md) (pre-fix audit) |

### Stage 2.6 selection (DEV only)

The Stage 2 candidate (new index, threshold 0.50, no rule 8) answered too many out-of-scope questions and
refused some Hinglish questions at the gate. Variants compared on DEV (`eval/results/stage2_6_answer_quality_dev.json`):

| DEV | Correct | Correct/partial | Fully grounded | Hallucination | OOS answered | False refusals |
|---|---:|---:|---:|---:|---:|---:|
| Previous production (base index, 0.50) | 47.5% | 82.5% | 60.0% | 28.3% | 0.0% | 20.8% |
| Stage 2 candidate (0.50, no rule 8) | 72.5% | 95.8% | 75.0% | 15.8% | 22.2% | 3.3% |
| 0.44, no rule 8 | 74.2% | 96.7% | 75.4% | 15.8% | 22.2% | 1.7% |
| 0.50 + rule 8 | 78.3% | 95.0% | 81.0% | 13.3% | 11.1% | 3.3% |
| **0.44 + rule 8 (selected)** | 75.8% | 96.7% | 77.8% | 14.2% | 11.1% | 2.5% |

- The threshold **0.44** is the midpoint of the DEV plateau (0.435–0.451). In that range no in-scope
  query is refused at the gate, and the out-of-scope gate decisions are unchanged
  (`eval/results/stage2_6_threshold_sweep_dev.json`).
- **Rule 8** halved out-of-scope answering on DEV (22.2% → 11.1%).
- "0.50 + rule 8" scored slightly higher on DEV correctness and hallucination. It was rejected because it
  still refused a Hinglish in-scope query at the gate (12.5% Hinglish false refusals). These DEV differences
  are within noise at this sample size. One global threshold reaches zero DEV gate refusals for both
  languages, so language-specific thresholds were not justified.

## 4. Final metrics (locked TEST)

### Retrieval

Source: `eval/results/stage2_comparison_lv2g_translate.json`. The threshold does not change ranking, so
Stage 2.6 retrieval is the same as the Stage 2 candidate's.

| TEST (92 scored) | Previous production | Stage 2.6 | Δ (95% CI) |
|---|---:|---:|---|
| Recall@5 | 0.668 | 0.830 | +0.161 |
| Recall@10 | 0.711 | **0.899** | +0.188 (+0.11, +0.27) |
| MRR@10 | 0.677 | **0.785** | +0.108 (+0.02, +0.19) |
| nDCG@10 | 0.636 | 0.776 | +0.140 |

### Answers

Source: `eval/results/stage2_6_answer_quality_test_locked.json`. Each column covers 184 in-scope answers and
30 out-of-scope answers (2 repeats).

| TEST | Previous production | Stage 2 candidate | **Stage 2.6 (production)** |
|---|---:|---:|---:|
| Correct | 63.0% | 78.8% | **81.5%** |
| Correct or partial | 84.8% | 96.7% | **96.2%** |
| Fully grounded | 73.7% | 82.0% | **85.9%** |
| Hallucination (in-scope) | 17.9% | 15.8% | **10.3%** |
| OOS answered | 6.7% | 13.3% | **6.7%** (2 of 30) |
| False refusals | 15.2% | 3.3% | **3.8%** |
| Evidence has gold | 77.2% | 90.2% | 91.3% |
| Citation inside gold span | 86.5% | 80.3% | **77.4%** |
| Median citation start error | 4.8 s | 6.0 s | 6.4 s |

Stage 2.6 TEST slices (small; indicative only):

| Slice | In-scope queries | Correct | Hallucination | False refusals | Citation in gold |
|---|---:|---:|---:|---:|---:|
| English | 80 | 80.6% | 10.0% | 4.4% | 76.5% |
| Hinglish | 12 | 87.5% | 12.5% | 0.0% | 83.3% |
| Follow-ups | 9 | 77.8% | 11.1% | 11.1% | 100% |

### Interpreting the citation and timestamp numbers

The citation-in-gold-span rate and the median start error are **worse** for Stage 2.6 than for the
previous system. Part of this is a labelling artefact. Stage 1 drafted gold spans from the previous system's
chunk times: 108 of 217 gold spans start exactly at one of its chunk boundaries, where it scores 0 s error by
construction. On spans placed independently of either system, Stage 2.6 is more precise: median start error
8.4 s vs 16.6 s on TEST, and 8.7 s vs 17.7 s overall. Stage 2.6 chunks are also longer (median 11.5 s vs
7.5 s). The citation figure is reported as measured.

## 5. Stages 3 and 4 in brief

**Stage 3: hybrid retrieval and reranking (DEV only, negative).** Variants tested: exact dense search, BM25,
BGE-M3 sparse, dense+BM25 RRF and ColBERT top-20 reranking. ColBERT gave the best retrieval (R@10 0.900 →
0.947, CI excludes 0), but **hallucination rose from 13–14% to 26%**, and correct answers fell from 77% to 69%.
The finding: retrieval metrics alone are not sufficient for promotion, and every retrieval change must be
judged end to end. 13 ColBERT judge records are missing because the API quota ran out. The rejection does
not depend on them.

**Stage 4: citation integrity and faithfulness (DEV, offline, no API calls).**

- **Z1:** 626 of 626 citations (100%) point at evidence the model was given; no fabricated timestamps.
  About 58% of factual sentences carry a citation (heuristic). Rule 8 makes no difference to either figure.
- **Z2:** the BGE-M3 claim–citation proxy separates grossly misaligned citations (AUROC 0.97–0.99). It does
  not judge whether a claim is supported.
- **Z3:** 39 stratified sentences, AI-labelled in one pass (not human ground truth). Z2 found 6 of 7
  citation problems. The existing LLM judge agreed poorly with the labels at claim level (κ 0.07–0.20).
  Neither instrument is reliable enough for claim-level decisions.

## 6. Legacy smoke suites

`eval/evaluate.py` (17 cases) and `eval/eval_followup.py` (5 cases) passed 17/17 and 5/5 on the **previous**
production system, graded manually. They call the OpenAI API and have **not** been re-run on Stage 2.6. They
are kept as smoke tests, not as evidence for the final configuration.

## 7. Limitations of the evaluation

- **AI-only labels and grades.** No human verification at any level (gold spans, answer grades, Z3 labels).
- **Correlated annotation.** The gold spans and the judge's reference were both placed with Whisper models.
  A passage every Whisper model mishears would be missed. On the 85 items whose gold never moved in
  Stage 1.5, the retrieval gain is smaller but still positive (R@10 +0.08, CI +0.02 to +0.15).
- **Single judge.** One LLM judge was used. Judge-only variance was not measured separately; a
  control re-run in Stage 3 bounds the total noise (hallucination Δ +0.017, CI −0.05 to +0.08).
- **Sample size.** 92 TEST and 60 DEV in-scope queries, and 15 TEST out-of-scope queries. Differences of a few
  points, and all slice results, are within noise.
- **Claim-level faithfulness is not measured** (Stage 4), and citation coverage is incomplete.
- **ANN non-determinism.** ChromaDB HNSW can differ from exact search for a few queries between processes.
  No metric effect was observed in Stage 3.
- **Reproducibility.** The result files pin the index fingerprint and the prompt SHA. A Whisper rebuild on
  other hardware may not reproduce the shipped index.

## 8. Reproducing (costs)

```bash
# free, local: retrieval metrics for DEV and TEST (opens ChromaDB, which rewrites index files)
venv/bin/python eval/run_benchmark.py --label <label>
venv/bin/python eval/compare_results.py eval/results/<control>.json eval/results/<label>.json

# PAID (OpenAI): answers + judge. Use --split dev; TEST is held out and already locked.
venv/bin/python eval/answer_eval.py generate --label <label> --split dev --repeats 2
venv/bin/python eval/answer_eval.py judge --label <label>
venv/bin/python eval/answer_eval.py summarize <label> --split dev
```

To evaluate a non-production index, point the tools at it with `CHROMA_DB_PATH=<abs path>
ACTIVE_COLLECTION=<name>`. The locked TEST results are final for this configuration. Re-running TEST to
tune anything would invalidate it as a held-out set.
