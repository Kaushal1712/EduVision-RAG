# Rebuilding the EduVision index

The production index is the Stage 2.6 selection, shipped as a verified artifact:

| | |
|---|---|
| Directory | `data/vector_db_lv2g_translate/` |
| Collection | `lv2g_translate`, 1,391 chunks |
| Fingerprint | `2a3b259ded116d30f6aedf8b45b069f1143c52bc1ef8cfbdd9c106725cdf61a7` (`eval/run_benchmark._index_fingerprint`) |
| Source | `experiments/stage2_transcription/artifacts/lv2g_translate/vector_db/` (byte-identical copy) |

Do not rebuild it in place. A rebuild produces a **new** index in a non-production directory, which is
verified and evaluated before it could ever replace the shipped one. Even with identical settings, Whisper
output can differ across hardware and library versions, so a rebuild may not reproduce the fingerprint.

## Safety guarantees (`ingestion/build_safety.py`)

- **Explicit target.** Every build names its output directory. The runtime `CHROMA_DB_PATH` is never used
  as a build target. Refused targets (the path, anything inside it, or any directory containing it): the
  shipped index, `data/vector_db/` (previous production index; only `allow_legacy_index=True` unlocks it),
  the verified Stage 2.6 source index, and whatever `CHROMA_DB_PATH` resolves to.
  Enforced in `indexer.build_index` (v1), `indexer_v2.main` and `build_candidate.py transcribe/index`.
- **No silent reuse of old outputs.** Ingestion stages skip work when their output exists, but only if:
  - transcripts: the cached file records the same Whisper model and decoding options as the current
    settings (`transcriber.transcription_options()`). The existing `data/transcripts/` files are
    Whisper `base` with no recorded options, so they are never reused for a Stage 2.6 build;
  - cleaned transcripts, chunks, embeddings: the cached file is not older than the file it came from.
  Otherwise `StaleOutputError` is raised: pass `force=True` / `--force`, or use fresh directories.
- **Translation cache off by default** in `indexer_v2` (`--use-translation-cache` to opt in): it is keyed by
  `chunk_id`, which collides between transcriptions.

## Reproducing the Stage 2.6 index (authoritative path)

The shipped index was assembled by `experiments/stage2_transcription/build_candidate.py`. That script is
the authoritative build path: production ingestion shares its transcription settings, cleaner, chunker and
embedder, but `indexer_v2.py` differs in index assembly (it reuses saved embeddings and the v2 metadata
flow), so it does not reproduce the shipped artifact exactly.

Settings (all represented in `config/settings.py` and checked by `python -m config.consistency`):
Whisper `large-v2`, task `translate`, temperature 0, no fallback, `condition_on_previous_text=False`,
language auto-detected; chunking 5 segments / overlap 1 / break on gaps > 5 s; BGE-M3 dense, fp16,
batch 16, max_length 512, of the raw chunk text; normalizer without cache; collection metric cosine.

```bash
# audio: data/processed/<video_id>.wav, extracted by ingestion/video_processor.py from videos/*.mp4
E=experiments/stage2_transcription
NAME=lv2g_translate_rebuild_<date>          # any new name; "lv2g_translate" is refused
venv/bin/python $E/build_candidate.py transcribe --name $NAME --model large-v2 --task translate --greedy
venv/bin/python $E/build_candidate.py index --name $NAME
# → $E/artifacts/$NAME/vector_db, collection $NAME (all 18 videos; nothing outside artifacts/$NAME)
```

Verify before any promotion: fingerprint and count (`eval/run_benchmark._index_fingerprint` on a copy,
since opening ChromaDB rewrites its files), then the retrieval benchmark on DEV with
`CHROMA_DB_PATH=<abs path> ACTIVE_COLLECTION=$NAME venv/bin/python eval/run_benchmark.py --label <label>`.
Promoting a rebuilt index is a separate, explicit decision: copy it to a new `data/` directory and update
`STAGE26_VECTOR_DB_DIR` / `ACTIVE_COLLECTION` defaults and `config/consistency.py` together.

## Legacy staged pipeline

`video_processor → transcriber → cleaner → chunker → embedder → indexer_v2` still works, now with the
guards above. It writes intermediate outputs to `data/transcripts/` and `data/processed/`, which hold the
Whisper `base` outputs; a Stage 2.6-settings run therefore stops at the transcriber unless `force` is
given. `indexer_v2` must be called with `--db-path <non-production dir> --collection <name>`.
