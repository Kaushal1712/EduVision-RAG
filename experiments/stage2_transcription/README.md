# Stage 2: transcription A/B experiment

**Question.** Does a stronger transcription improve EduVision retrieval, especially
for queries whose answers sit where the production transcript is `[unclear audio]`
or missing?

**Status.** Isolated experiment. Production transcripts, chunks, embeddings, index,
code and configuration are unchanged; see "Isolation" below. Nothing here is used by
the app.

> **Later status (finalization, 2026-10-05).** The `lv2g_translate` index built here was selected in
> Stage 2.6 and is now production. A byte-identical copy ships as `data/vector_db_lv2g_translate/`, and
> `build_candidate.py` is the authoritative rebuild path (`ingestion/REBUILD.md`). The text below is
> unchanged and describes the experiment as it ran.

## Design

| | CONTROL (production) | CANDIDATE |
|---|---|---|
| ASR | Whisper `base`, task `transcribe`, language auto (`hi`), default decoding (temperature fallback 0→1, `best_of=5`, `condition_on_previous_text=True`) | Whisper `large-v2`, see "Candidate" |
| Cleaning | `ingestion/cleaner.clean_transcript` | same function, unchanged |
| Chunking | `_build_chunks_from_segments`: 5 segments, overlap 1, 5 s gap break | same function, same parameters |
| Embedding | BGE-M3 dense, fp16, batch 16, max_length 512, of the raw chunk text | same functions (`_load_bge_model`, `_embed_chunks`); parity check: cosine 1.000000 on 40 production chunks |
| English display text | `normalize_chunks` (quality filter + GPT-4o-mini translation of non-English chunks) | same function, with the cache disabled (it is keyed by chunk id, which collides) |
| Index | `data/vector_db`, collection `eduvision_chunks_v2` | `artifacts/<name>/vector_db`, collection `<name>`, same metadata |
| Retrieval, threshold, top-k, rewrite | production code | **the same production code**, pointed at the experimental index with the `CHROMA_DB_PATH` / `ACTIVE_COLLECTION` environment variables |
| Benchmark | eduvision-bench-v1.1, 152 scored queries | same |

Production's Tutorial #1 also contains a one-off patch (08:30–16:00 re-transcribed
with Whisper `medium`, see `archive/patch_t1_stage15.py`). The candidate
re-transcribes all of Tutorial #1.

**Why `large-v2` and not `medium`.** Stage 1.5 placed the gold spans by reading Whisper
`medium` (translate) transcripts. A `medium` candidate would be graded against labels
derived from its own output. `large-v2` was used in Stage 1.5 only as a second opinion
on 38 spans. Both are Whisper models, so some correlation remains (see limitations).

## Candidate selected: `lv2g_translate`

Whisper `large-v2`, task `translate`, `temperature=0.0` with no fallback,
`condition_on_previous_text=False`, `fp16=False`, language auto-detected (`hi` on
all 18 videos). CPU (Apple M3 Pro), openai-whisper 20250625, torch 2.13.0.

How it was chosen (pilot, DEV only):

1. Production-identical decoding (temperature fallback, `best_of=5`, previous-text
   conditioning) with `large-v2` ran below 0.33× real time on CPU and was aborted.
   Greedy decoding without fallback was adopted for both variants.
2. On T03, T04, T05, T08 and T18, both tasks were transcribed and indexed as a
   hybrid (production chunks for the other 13 videos). On the 13 DEV queries with
   gold in those videos:
   - control: R@10 0.564, MRR 0.497
   - `transcribe`: R@10 0.718, MRR 0.718; 2 recovered, 1 lost; 1.0× real time;
     GPT normalizer turned 255 of 396 clear chunks into `[unclear audio]`
   - `translate`: R@10 0.936, MRR 1.000; 4 recovered, 0 lost; 3.2× real time;
     no normalizer translation needed

   TEST was not used for the selection.

## Results (full corpus, benchmark v1.1)

Files: `eval/results/stage2_lv2g_translate.json` (candidate),
`eval/results/stage2_comparison_lv2g_translate.json` (paired comparison with the frozen
Stage 1.5 baseline), `eval/results/stage2_run_variation.json`,
`eval/results/stage2_sensitivity_*`, and `transcript_quality_lv2g_translate.json`.

| TEST (n=92) | Control | Candidate | Δ (95% CI) |
|---|---:|---:|---|
| Recall@5 | 0.668 | 0.830 | +0.161 |
| Recall@10 | 0.711 | 0.899 | +0.188 (+0.11, +0.27) |
| MRR@10 | 0.677 | 0.785 | +0.108 (+0.02, +0.19) |
| nDCG@10 | 0.636 | 0.776 | +0.140 |

Coverage slices are taken from the control run. On TEST, R@10 changed as follows:
- full (n=54): 0.843 → 0.910
- partial (n=33): 0.573 → 0.894
- none (n=5): 0.200 → 0.800

Known costs and caveats:

- **Timestamps.** On TEST queries both systems hit, the median start error rose
  from 4.7 s to 7.5 s, and chunk median length grew from 7.5 s to 11.5 s.
- **Rare-term queries.** Flat: n = 7–8 per split, CI spans 0.
- **Annotation correlation.** Gold was located in Stage 1.5 with Whisper `medium`
  translate. On the 85 items whose gold was never moved, the gain is smaller but
  still positive (R@10 +0.08, CI +0.02 to +0.15).
- **Stricter relevance.** Gains hold at a minimum overlap of 3 s and 5 s.

## Stage 2.5 validation

Analyses: `stage2_5_analysis.py` → `eval/results/stage2_5_analysis.json`.
Answer quality: `eval/answer_eval.py` → `eval/results/answers_s25_*.json` and
`eval/results/stage2_5_answer_quality_{dev,test}.json`.

**Chunking / boundary refinement.** DEV only; the candidate was kept unchanged.

| Variant | R@10 | MRR@10 | Median start error |
|---|---:|---:|---:|
| w5o1, gap 5 s (candidate) | 0.900 | 0.766 | 6.5 s |
| w3o1 | 0.889 | 0.691 | 8.2 s |
| w4o1 | 0.906 | 0.749 | 6.9 s |
| w5o2 | 0.928 | 0.780 | 6.9 s |
| gap 3 s | 0.900 | 0.765 | 7.5 s |
| gap 2 s | 0.906 | 0.778 | 6.5 s |

None of the variants clearly improves the trade-off. The candidate's segments are nearly contiguous, so the gap threshold has almost no effect.

**Timestamp error is confounded by the gold labels.** 108 of 217 gold spans start exactly at a
production chunk boundary, because Stage 1 drafted spans from production chunk times. On those
spans control scores a median start error of 0.0 s by construction. On spans set independently
of any system, the candidate is more precise:

| Split | Control | Candidate |
|---|---:|---:|
| All | 17.7 s | 8.7 s |
| TEST | 16.6 s | 8.4 s |

**Downstream answers.** Full `pipeline.ask()` with GPT-4o-mini; judge `gpt-4.1-2025-04-14`,
2 repeats. TEST in-scope answers:

| TEST | Control | Candidate |
|---|---:|---:|
| Correct | 63% | 79% |
| False refusals | 15% | 3% |
| Fully grounded | 74% | 82% |
| Hallucination | 18% | 16% |
| Out-of-scope questions answered | 6.7% | 13.3% |
| Citation inside the gold span | 87% | 80% |

**Hinglish.** The English index lowers top-1 similarity for Hinglish queries (median 0.611 →
0.576). 2 of 20 Hinglish in-scope queries fall under the unchanged 0.50 gate and are refused
without an LLM call; EVB-010 has the correct chunk at rank 1 at 0.480.

## Reproduce

```bash
E=experiments/stage2_transcription
venv/bin/python $E/build_candidate.py parity
venv/bin/python $E/build_candidate.py transcribe --name <name> --model large-v2 --task transcribe --greedy
venv/bin/python $E/build_candidate.py index --name <name>
CHROMA_DB_PATH="$PWD/$E/artifacts/<name>/vector_db" ACTIVE_COLLECTION=<name> \
    venv/bin/python eval/run_benchmark.py --label stage2_<name>
venv/bin/python eval/compare_results.py eval/results/stage1_5_baseline_bench-v1.1.json \
    eval/results/stage2_<name>.json --out eval/results/stage2_comparison_<name>.json
venv/bin/python $E/transcript_quality.py --name <name>
```

Pilot (subset of videos, other videos filled from the committed production index):
`... transcribe ... --videos 3 4 5 8 18` then `... index ... --videos 3 4 5 8 18 --fill-from-production`.

## Isolation

- Every artifact is written under `artifacts/<name>/` (git-ignored).
- `cleaner.clean_transcript` saves to its module-level `TRANSCRIPTS_DIR`. The build
  script points that name at the artifacts directory and asserts where the file landed.
- The production index is never opened. Pilot fill-in reads a `git archive` copy of
  the committed index (`artifacts/_production_index_snapshot/`).
- SHA-256 of all 73 production transcript/chunk/embedding/cache files and of the
  production code were recorded before the experiment and compared afterwards.
