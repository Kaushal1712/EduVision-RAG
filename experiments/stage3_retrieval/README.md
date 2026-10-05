# Stage 3: hybrid retrieval and reranking — final summary

**Status: closed (2026-10-05) with a negative result. Stage 2.6 is retained.**
DEV only; TEST was not used. Production code, configuration and the frozen Stage 2.6 files were not changed.

## 1. Objective

Find out whether hybrid retrieval (dense + lexical) or reranking improves **end-to-end** answer
quality (correct, grounded, low hallucination, unchanged abstention) over the frozen Stage 2.6
candidate. Retrieval metrics alone were not enough for promotion.

## 2. Frozen Stage 2.6 reference

Configuration: see section 9. Source: `experiments/stage2_6_abstention/locked_selection.json`.

| Retrieval | R@5 | R@10 | MRR@10 | nDCG@10 |
|---|---:|---:|---:|---:|
| DEV (60 scored) | 0.892 | 0.900 | 0.766 | 0.773 |
| TEST (92 scored, locked) | 0.830 | 0.899 | 0.785 | 0.776 |

| Answers (judge gpt-4.1-2025-04-14) | Correct | Correct/partial | Fully grounded | Hallucination | OOS answered | False refusals | Evidence has gold | Citation in gold span |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| DEV | 75.8% | 96.7% | 77.8% | 14.2% | 11.1% | 2.5% | 93.3% | 77.8% |
| TEST (locked) | 81.5% | 96.2% | 85.9% | 10.3% | 6.7% | 3.8% | 91.3% | 77.4% |

## 3. Experiments (all DEV, against exact dense unless stated)

Every retrieval variant was run twice in separate processes; all repeat runs were identical. (The
HNSW control itself differed between processes in one lower-ranked chunk for EVB-103, with no metric
change.)
In every variant each chunk kept its BGE-M3 dense cosine, so the 0.44 gate and per-chunk evidence
filter were unchanged; the query-level gate was the exact-dense top-1. No gate decision changed in
any variant (0/69 DEV queries).

| Variant | Retrieval (DEV) | Evidence / answers | Decision | Reason |
|---|---|---|---|---|
| **Exact dense** (brute-force cosine over the same embeddings) | Identical to the Stage 2.6 HNSW search on all 60 scored queries (R@10 0.900, MRR 0.766); deterministic; 45 vs 53 ms p50 | Evidence identical on 60/60 queries | **Keep as experimental baseline** (not a promotion; no functional change) | HNSW approximation does not affect the baseline (only EVB-161, an OOS query, differs in its top-10). Pool recall @20 0.961, @50 0.992. |
| **BM25** (k1 1.2, b 0.75, `[a-z0-9]+` tokens, no stemming) | R@10 0.842 (Δ −0.058), MRR 0.702; exact-term MRR 0.78 → 0.88; paraphrase and follow-up collapse | Evidence recall 0.892 → 0.819. Adds gold that passes the 0.44 filter for EVB-002, -115, -120. Not answer-evaluated | **Diagnostic only** | Worse ranker overall; useful only as a candidate source for exact/rare terms. |
| **BGE-M3 sparse** (lexical weights, cached model) | R@10 0.839, MRR 0.681 (R@5 Δ −0.108, CI excludes 0); Hinglish MRR 0.675 → 0.854 | Evidence recall 0.778. Adds filter-passing gold only for EVB-115; its Hinglish gain (EVB-172) has dense cosine 0.35 and is filtered. Not answer-evaluated | **Reject** | Weaker than BM25 and adds nothing BM25 does not; its distinct gain never reaches the LLM. |
| **Dense + BM25 RRF** (k ∈ {20, 40, 60}, predeclared; best k=20) | R@10 0.922 (+0.022, CI [−0.028, +0.075]), MRR 0.804; R@5 0.892 → 0.836; follow-up R@5 0.77 → 0.10 | Evidence recall 0.892 → 0.850; median timestamp error 6.99 → 8.90 s on queries both hit. Not answer-evaluated | **Reject** | No meaningful R@10 gain; lexical matches displace dense evidence from the top-5, lowering evidence recall. |
| **ColBERT top-20** (exact dense top-20 reordered by BGE-M3 multi-vector MaxSim) | R@10 **0.947** (+0.047, CI [+0.008, +0.097]), MRR 0.811, 0 queries lost (EVB-002, EVB-120 recovered) | Evidence has gold 93.3% → 96.7%, citation in gold span 77.8% → 83.3%, false refusals 2.5% → 0%. **Hallucination 13–14% → 26%** (paired Δ +0.11 to +0.13, CI excludes 0); correct 77% → 69% and fully grounded 79% → 69% (Δ −0.08, CI upper bound 0). Median timestamp error 6.52 → 7.46 s | **Reject** | Better retrieval and evidence, but measurably less faithful answers. |

ColBERT answer figures are over the 115 in-scope records judged in all arms (two controls: the frozen
Stage 2.6 DEV answers and a same-day re-run). Noise floor (control re-run vs frozen): hallucination
Δ +0.017, CI [−0.050, +0.083]; correct Δ +0.008.

Latency per query (warm, M3 Pro): dense 45 ms p50; BM25 +0.1 ms; BGE-M3 sparse +41 ms; ColBERT
+43 ms query encoding + 0.6 ms scoring for 20 candidates, plus 283 MB of passage vectors and an
18.8 s index build. Sparse and ColBERT ran as a separate model pass; one shared pass was not measured.

## 4. Conclusion

No tested retrieval or reranking variant improved end-to-end answer quality over Stage 2.6.
**Stage 2.6 remains the selected configuration.**

## 5. Main finding

**ColBERT improved retrieval and evidence metrics but increased hallucination and reduced answer
faithfulness.** Retrieval metrics (R@k, MRR) and gold-in-evidence alone are not sufficient for
promotion in this system; every retrieval change must be judged end to end.

Other measured findings:
- Candidate coverage is not the bottleneck: the dense top-50 already holds gold for nearly every DEV
  query (recall 0.992). Ordering and evidence selection are what change answers.
- Lexical retrieval helps exact and rare technical terms but hurts paraphrases and follow-ups
  (rewritten follow-ups are generic text).
- The 0.44 dense gate is stable under all variants; OOS answering at gate level never changed.
- For EVB-171 and EVB-172 (follow-up, one Hinglish) the gold has dense cosine 0.395–0.438, below
  the 0.44 evidence filter, so no reranker can pass it to the LLM.

## 6. Not run: ColBERT evidence-order isolation pilot

Prepared but intentionally NOT RUN because it requires additional paid API calls and is not
required for the current promotion decision.

- Arm `colbert_dense20_denseorder` in `answer_eval_stage3.py` (ColBERT evidence set, re-sorted by
  dense cosine before generation) is implemented and unit-tested.
- 15-query DEV pilot selected in `pilot_option1_dev.json` (estimated 186 API calls, ≈ $0.30).
- Running it needs an `--ids-file` filter in `answer_eval_stage3.py`, which was not added.

## 7. Limitations

- The ColBERT answer evaluation has **13 missing judge records** (repeat 1: 5 in-scope follow-ups,
  8 out-of-scope) because the OpenAI quota was exhausted. The ColBERT result is therefore
  preliminary in completeness. It is already sufficient to reject ColBERT as a promotion candidate,
  because the hallucination increase in the 115 judged in-scope records is statistically meaningful
  against both controls and well outside the measured control noise.
- **The evidence-order hypothesis was not tested and is not proven.** Whether ColBERT's regression
  comes from evidence order or evidence set is unknown. Descriptively, the increase appears both
  where ColBERT changed the evidence set (92 records) and where it only reordered it (22 records).
- DEV is small: 60 scored in-scope queries, 9 out-of-scope, 8 Hinglish, 8 rare-term and 5 follow-up
  queries. Slice results are indicative only.
- Gold labels were placed with Whisper medium (Stage 1.5) and checked by an AI annotator, not a human.
- One LLM judge (gpt-4.1, temperature 0) was used; judge-only variance on identical answers was not
  measured separately (the control re-run noise includes it).
- BM25, BGE-M3 sparse and RRF were not answer-evaluated; they were rejected on retrieval and
  evidence results.

## 8. Exit criteria (met)

Stage 3 is complete, with Stage 2.6 retained, because no tested retrieval change demonstrated an
end-to-end improvement over Stage 2.6 on DEV. TEST was therefore not run for any Stage 3 variant.

## 9. Configuration carried forward to Stage 4 (= Stage 2.6, unchanged)

| Component | Setting |
|---|---|
| ASR | Whisper `large-v2`, task `translate`, `temperature=0.0` (no fallback), `condition_on_previous_text=False`, `fp16=False`, language auto-detected |
| Cleaning / chunking | production `cleaner.clean_transcript`; 5 segments per chunk, 1-segment overlap, break on gaps > 5 s |
| Embeddings | BGE-M3 dense (`BAAI/bge-m3`): passages fp16, batch 16, max_length 512; queries fp32, max_length 512, no instruction prefix |
| Index | ChromaDB cosine, collection `lv2g_translate` at `experiments/stage2_transcription/artifacts/lv2g_translate/vector_db`; 1,391 chunks; fingerprint `2a3b259ded116d30f6aedf8b45b069f1143c52bc1ef8cfbdd9c106725cdf61a7` |
| Retrieval | production `retrieval/retriever.retrieve()` (ChromaDB HNSW), top-k 10, no hybrid, no reranking |
| Gate / evidence | similarity threshold **0.44** (refuse without an LLM call when no chunk passes; only passing chunks are evidence); up to **5** evidence chunks in ranked order |
| Generation | GPT-4o-mini, temperature 0.2, production `SYSTEM_PROMPT` + rule 8 grounding guard (`experiments/stage2_6_abstention/guard_prompt.txt`, SHA-256 `9b294ecd05aa9bc350d5056005774e533e1700373fd5b5bd939046fbb133b4e3`) |
| Follow-ups | existing `pipeline._rewrite_query_for_retrieval`, unchanged |
| Locked selection | `experiments/stage2_6_abstention/locked_selection.json` (SHA-256 `69599b24f218ca6e95a1555b92b84f5c63ff87a31570d198bccf0671bcef91e0`) |

Exact dense search is functionally equivalent to the HNSW search on DEV and may be used as a
deterministic implementation, but it was not adopted as a change.

## Files

- Code: `run_stage3.py` (retrieval variants), `compare_stage3.py`, `bm25.py`, `m3sparse.py`,
  `fusion.py`, `colbert.py`, `answer_eval_stage3.py`, `answer_compare_stage3.py`;
  tests in `tests/test_stage3_retrieval.py`.
- Results: `eval/results/stage3_*_dev*.json`, `eval/results/stage3_comparison_*_dev.json`,
  `eval/results/answers_s3_control_dev.json`, `eval/results/answers_s3_colbert20_dev.json`.
- Artifacts (git-ignored): `artifacts/` (BGE-M3 sparse passage weights).

Reproduce a retrieval variant: `venv/bin/python experiments/stage3_retrieval/run_stage3.py --variant <name>`
then `compare_stage3.py eval/results/stage3_exact_dense_dev_r3.json eval/results/stage3_<name>_dev.json`.
