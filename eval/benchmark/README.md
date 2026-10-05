# EduVision timestamp-anchored retrieval benchmark (Stage 1)

This directory holds the ground truth used to measure EduVision's retrieval.
It complements, and does not replace, the end-to-end regression suites
`eval/evaluate.py` (17 single-turn cases) and `eval/eval_followup.py` (5 follow-up cases).

| File | Purpose |
|---|---|
| `eduvision_bench_v1.json` | 176 queries with gold timestamp spans |
| `corpus_manifest.json` | Video identity: `video_id`, tutorial number, filename, duration, sha256 |

Code: `eval/bench_metrics.py` (metrics), `eval/bench_data.py` (validation),
`eval/run_benchmark.py` (runner), `eval/bench_tools.py` (annotation CLI).
Results: `eval/results/<label>.json`.

## Why timestamps, not chunk IDs

Chunk IDs encode the chunk position (`<video_id>_chunk_0042`). They change whenever
the corpus is re-transcribed or re-chunked. The place in the video where a topic is
explained does not. Gold is therefore a list of spans on the video timeline:

```json
"gold": [{"video_id": "07_forms_and_input_tags_in_html_sigma_web_development_course_tutorial_7",
          "start_time": 404.9, "end_time": 436.9}]
```

**Relevance rule.** A retrieved chunk is relevant when it has the same `video_id` as
a gold span and its `[start_time, end_time]` overlaps that span by more than
`min_overlap_s` (default 0: any positive overlap; touching endpoints do not count).

**Video identity.** `video_id` is derived from the `.mp4` filename by
`ingestion/video_processor.py::_make_video_id`, and retrieval returns it in chunk
metadata. The manifest pins each video's sha256. Gold timestamps stay valid across
re-transcription and re-chunking. They must be re-checked if a video file is
replaced, trimmed or re-encoded (`python eval/bench_tools.py manifest --check`).

## Query schema

| Field | Values |
|---|---|
| `id` | `EVB-NNN`, stable |
| `split` | `dev` (tuning in later stages) / `test` (held out; report only) |
| `query` | user text exactly as typed |
| `history`, `followup` | prior turns for follow-ups (`followup` is true exactly when `history` is non-empty) |
| `category` | `in_scope`, `out_of_scope` (unrelated), `near_miss_oos` (web-dev topic the course does not cover) |
| `language` | `en`, `hinglish` (Romanized Hindi) |
| `phrasing` | `exact_term`, `paraphrase`, `rare_term` (in-scope only; `null` otherwise) |
| `difficulty` | `easy`, `medium`, `hard` (annotator judgement; garbled evidence or multi-span gold ⇒ harder) |
| `gold` | list of `{video_id, start_time, end_time}`; non-overlapping per video; empty for OOS |
| `annotation.status` | `draft`, `verified`, `needs_annotation` |
| `annotation.evidence` | transcript excerpt that justifies the span |
| `source` | original case in `eval/evaluate.py` / `eval/eval_followup.py`, if any |

Follow-up `history` assistant turns are fixed, hand-written context so the input is
reproducible. They are not recorded model outputs.

**Split rule.** Within each (category, followup, language, status) stratum, in id
order, positions `i % 5 ∈ {1, 3}` are `dev` and the rest are `test`. Do not move
items between splits after results have been looked at.

## Annotation status

| Status | Meaning | Scored? |
|---|---|---|
| `draft` | Gold read from the production transcripts only (v1.0). None remain in v1.1. | yes |
| `video_checked` | Gold checked against the video's own audio and frames by an **AI annotator**, not yet by a person (see below). | yes |
| `verified` | Gold checked against the video by a person. None yet. | yes |
| `needs_annotation` | In-scope, but no span could be located confidently; `annotation.notes` says why. | no (abstention only) |

### How v1.1 was checked (Stage 1.5)

v1.0 gold came from the production transcripts, so it could only cover regions where
those transcripts are readable. For v1.1 every item was re-checked against the videos:

1. `python eval/bench_tools.py asr --model medium` re-transcribed every video's audio
   with Whisper `medium` (English translation, no conditioning on previous text). The
   output goes to `eval/benchmark/annotation_asr/` (git-ignored). It is an annotation
   aid only; the production transcripts, chunks and index were not changed.
2. Each video's annotation transcript was read end to end with the existing gold
   spans marked inline. Every span was confirmed, re-bounded, extended with other
   passages that answer the same question, or replaced. The 8 `needs_annotation`
   items were located the same way. The retriever was never consulted.
3. Spot checks with frames (`python eval/bench_tools.py frames …`) confirmed on-screen
   code for demonstrations. The source videos are 640×360, so only large, zoomed
   text is legible.
4. A second, independent model (Whisper `large-v2`) re-transcribed every newly
   located or replaced span, plus a seeded random sample of 25 other spans
   (`annotation.verification.second_opinion_large_v2`).
5. Out-of-scope items were re-checked by searching all 18 annotation transcripts
   for their key terms (`annotation.verification.check`).

Every changed item keeps `annotation.previous_gold` and its old `draft_evidence`, and
`annotation.verification.change` is one of `confirmed`,
`boundaries_or_spans_corrected`, `spans_added`, `replaced` or `located`.

Because both annotation models are Whisper, a region every Whisper model mishears
would still be missed. A human pass (status `verified`) remains the stronger standard.

### Workflow for a human annotator

```bash
venv/bin/python eval/bench_tools.py todo                      # what is left
venv/bin/python eval/bench_tools.py show 7 404.9 436.9        # transcript around a span
venv/bin/python eval/bench_tools.py find "rowspan" --video 5  # where a term is said
```

1. Open the video at `start_time` and watch through `end_time`.
2. Move the boundaries to where the answer actually starts and ends. Merge spans that touch.
3. Add any other place in the corpus that answers the question equally well.
4. Set `annotation.status` to `verified` and record what changed in `annotation.notes`.
5. For `needs_annotation` items, add gold spans (or delete the item if the video does
   not cover the topic) and set the status to `verified`.
6. Run `venv/bin/python eval/bench_tools.py validate`, then bump `benchmark_version`.
   Results from different benchmark versions are not comparable.

## Transcript-coverage slice

Each result file records, per scored query, `gold_usable_text_coverage`: the share of
the gold time for which the production index has usable (not `[unclear audio]`) text.
It also gives a breakdown `by.transcript_coverage` = `none` / `partial` / `full`.
A transcription change should mainly move the `none` and `partial` slices, so compare
them separately instead of only the overall averages.

In v1.0, 0% of gold time lay in regions without usable production text. Gold could
only be read there, which hid exactly what a transcription upgrade would fix. In v1.1
it is 33% (8 items `none`, 51 `partial`, 92 `full`), close to the corpus-wide share
of `[unclear audio]` chunks.

## Known limitations of v1.1

- **AI annotator, not a person.** Queries, v1.0 spans and the v1.1 checks were all
  done by the same AI agent that built the tooling. The checks used the videos' own
  audio (two independent Whisper models) and frames, but nobody has watched the
  videos. Human verification is still recommended before decisions that hinge on
  small differences.
- **Query set composition.** Queries were written in v1.0 from readable transcript
  regions. v1.1 added gold in unreadable regions for those queries, and located the
  8 core-topic items. However, topics that are only taught inside unreadable regions
  have no query of their own (for example box-sizing and margin collapse in T18, and
  the backend explanation in T01's second half). A v1.2 extension with such queries
  would make the transcription-sensitive slice larger.
- **Scale.** With about 150 scored queries, differences of a few points between
  systems are within noise. Prefer per-query comparisons.
- **ANN non-determinism.** ChromaDB's HNSW search returned a top-10 that differs from
  exact search for roughly 4–7% of queries, and which queries are affected varies
  between processes. Each result file records `exact_topk_overlap` per query.
