# EduVision RAG — Finalization Audit (inspection only)

> **Point-in-time snapshot.** This audit describes the repository before finalization steps 1–7. Those
> steps shipped the Stage 2.6 index, switched the runtime defaults, added rule 8 and the consistency test,
> added rebuild safety and fixed the UI safety findings (G2, G4, G10). Current state: `README.md` and
> `docs/EVALUATION.md`. The remaining open items are listed there under known limitations.

Date: 2026-10-05 · Scope: whole repository · Mode: read-only inspection, offline unit tests only.
No OpenAI/API calls, no TEST benchmark runs, no code/config/index changes, no commits.

Severity scale: **BLOCKER** (must fix before finalizing) · **HIGH** · **MEDIUM** · **LOW** · **OK**.

---

## A. Executive summary

The selected configuration (Stage 2.6, `experiments/stage2_6_abstention/locked_selection.json`) is **not what
the production app runs**. The app, through `config/settings.py` defaults and the local `.env`, still serves the
pre-Stage-2 baseline: Whisper-`base` index `eduvision_chunks_v2` in `data/vector_db`, threshold **0.50**, and a
system prompt **without Rule 8**. The Stage 2.6 index exists only in a git-ignored experiment folder, and the
production ingestion code cannot rebuild it. These are blockers.

Otherwise the code base is in reasonable shape:
- no secrets are tracked;
- no shell-injection or path-traversal exposure was found, and there is no file upload;
- the runtime does not import Whisper or ffmpeg;
- graceful handling of missing videos works;
- all 144 offline unit tests pass.

The main non-blocking work is:
- **Repository hygiene:** the whole Stage 1–4 evaluation harness, tests and benchmark are untracked.
- **Outdated documentation:** the README describes the old configuration and metrics.
- **Deployment details:** the Python version is not pinned; a relative DB path depends on the working directory.
- **Minor safety and UX items.**

**Verdict: needs fixes before finalization** (see section K).

---

## B. Current production architecture (as implemented)

| Stage | Module | What it does now | Status |
|---|---|---|---|
| Audio extraction | `ingestion/video_processor.py` | `ffmpeg`/`ffprobe` via `subprocess.run([...])`, argument list, no shell | OK |
| Transcription | `ingestion/transcriber.py` | `whisper.load_model(WHISPER_MODEL)` (env: `base`), `model.transcribe(...)` with default task (*transcribe*), default temperature fallback, default `condition_on_previous_text`, `word_timestamps=False` | does not match Stage 2.6 (HIGH, see D/E) |
| Cleaning | `ingestion/cleaner.py` | segment cleaning | OK (same as Stage 2.6) |
| Chunking | `ingestion/chunker.py` | 5 segments, overlap 1, hard break on gaps > 5 s | OK (matches Stage 2.6) |
| Normalization | `ingestion/normalizer.py` | quality filter and GPT-4o-mini translation of non-English chunks (cached) | OK; Stage 2.6 index needed no translation |
| Embeddings | `ingestion/embedder.py` | BGE-M3 dense, fp16, batch 16, max_length 512 | OK |
| Vector DB | `ingestion/indexer.py` (v1), `ingestion/indexer_v2.py` (v2) | ChromaDB persistent client; v2 collection name hard-coded `eduvision_chunks_v2` | see D/E |
| Retrieval | `retrieval/retriever.py` | BGE-M3 query (fp32), ChromaDB HNSW cosine, top-k, per-chunk `below_threshold` flag | OK (algorithm = Stage 2.6) |
| Threshold / abstention | `generation/generator.py` `generate` | no LLM call when no chunk ≥ threshold | OK (logic); value wrong (D) |
| Evidence selection | `generator._select_evidence` | first `MAX_LLM_EVIDENCE` (5) above-threshold chunks, ranked order | OK |
| Generation | `generator.generate` | GPT-4o-mini, `TEMPERATURE` 0.2, `MAX_TOKENS` 1024, `SYSTEM_PROMPT` rules 1–7 | Rule 8 missing (D) |
| Follow-ups | `pipeline._rewrite_query_for_retrieval`, prior-context block in generator | regex-gated GPT rewrite (temperature 0), falls back to original query | OK (unchanged, as selected) |
| Citations | model-written `[Video: "…" @ MM:SS]` | free text, not parsed or validated in production | OK per Stage 4 Z1 (100% valid on DEV) |
| Go-to-Timestamp | `app/ui.py` `_render_source_card`, `_video_path`, `st.video(start_time=…)` | button per source card; hidden when `videos/` or the file is absent | OK |
| Search Evidence | `pipeline.search` → `retrieve` | retrieval only, above/below threshold groups | OK (follows the threshold setting) |
| UI | `app/ui.py` (Streamlit) | chat tab, search tab, sidebar filter, cached pipeline, 10-question session counter | see G |

Runtime import chain: `app/ui.py → pipeline → retrieval, generation, ingestion.indexer (get_chroma_client) →
ingestion.chunker/cleaner/transcriber/embedder` (module-level imports only; Whisper and FlagEmbedding load lazily).
Verified: importing `pipeline` does not load `whisper`, `FlagEmbedding` or `tqdm`.

---

## C. Stage 2.6 configuration verification

| Item | Stage 2.6 (locked) | Production runtime today | Severity |
|---|---|---|---|
| ASR | Whisper large-v2, task translate, temperature 0, no fallback, `condition_on_previous_text=False` | index built from Whisper `base`, task transcribe, default decoding; `.env` `WHISPER_MODEL=base` | **BLOCKER** (via index) |
| Chunking | 5 / 1 / gap > 5 s | same | OK |
| Embeddings | BGE-M3 dense | same | OK |
| Collection / index | `lv2g_translate`, 1,391 chunks, fingerprint `2a3b259d…`, at `experiments/stage2_transcription/artifacts/lv2g_translate/vector_db` (git-ignored, 19 MB) | `eduvision_chunks_v2` in `data/vector_db` (tracked) via `ACTIVE_COLLECTION` default | **BLOCKER** |
| top_k | 10 | `RETRIEVAL_TOP_K` default 10 (not set in `.env`) | OK |
| Threshold | 0.44 | **0.50** (`.env` `SIMILARITY_THRESHOLD=0.50`; code default also 0.50) | **BLOCKER** |
| Evidence chunks | up to 5 | `MAX_LLM_EVIDENCE` default 5 | OK |
| Generator | GPT-4o-mini, temperature 0.2 | `.env` `OPENAI_MODEL=gpt-4o-mini`, `TEMPERATURE=0.2` | OK |
| System prompt | production prompt + Rule 8 (`guard_prompt.txt`, SHA-256 `9b294ecd…`) | `generator.SYSTEM_PROMPT` has rules 1–7 only; Rule 8 appears nowhere in production code (it was injected at evaluation time by `answer_eval.py --system-prompt-file`) | **BLOCKER** |
| Follow-up rewrite | unchanged | unchanged | OK |

Accidental fallback risk: because the code defaults are the old values, **any environment without the right
variables silently runs the baseline**. This applies to a fresh deployment, a missing secret, or a different
working directory. Finalizing therefore needs both new defaults and explicit env values.

---

## D. Critical issues

| # | Finding | Evidence | Severity |
|---|---|---|---|
| D1 | Production runtime does not run Stage 2.6 (index, threshold, prompt) | section C | **BLOCKER** |
| D2 | Stage 2.6 index is not deployable: it lives only under the git-ignored `experiments/stage2_transcription/artifacts/` | `artifacts/.gitignore` `*`; `data/vector_db` holds only the old collections | **BLOCKER** |
| D3 | Rule 8 exists only in `experiments/stage2_6_abstention/guard_prompt.txt`, not in `generation/generator.py` | grep for the Rule 8 text in production code: no match | **BLOCKER** |
| D4 | Production ingestion cannot rebuild the Stage 2.6 index: the transcriber has no translate, temperature or `condition_on_previous_text` options; `indexer_v2.py` hard-codes `eduvision_chunks_v2` and expects v1 embeddings. The index was built by `experiments/stage2_transcription/build_candidate.py` | `transcriber.py:279-285`, `indexer_v2.py:48,61` | **HIGH** (reproducibility; not a runtime blocker if the built index is shipped) |

---

## E. High-priority cleanup

| # | Finding | Severity |
|---|---|---|
| E1 | **Evaluation work is untracked.** The benchmark (`eval/benchmark/`), the harness (`run_benchmark.py`, `bench_*.py`, `compare_results.py`, `answer_eval.py`), the 144 unit tests (`tests/`), all experiments and their reports are untracked: 117 non-ignored files, about 34 MB, much of it bulky per-run JSON. Nothing that justifies Stage 2.6 is in git. | HIGH |
| E2 | **README and config documentation describe the old system:** threshold 0.50, Whisper `base`, `eduvision_chunks_v2`, 1,510 / 1,528 chunks, only the legacy 17/17 and 5/5 evaluation. It claims "Whisper word-level alignment", but the code uses `word_timestamps=False`. It doesn't mention Rule 8, the benchmark, or the Stage 1–4 results. It lists Python "3.10+" while the environment is 3.14. | HIGH |
| E3 | **Opening ChromaDB rewrites the tracked `data/vector_db` files** (observed in Stages 1–2). Running the app or the benchmark against the tracked index dirties the working tree. 50 binary index files are tracked. A policy is needed for the new index: ship as tracked binary, read-only copy, or build artifact. | HIGH |
| E4 | **`.env` overrides will defeat a code fix.** `.env` sets `SIMILARITY_THRESHOLD=0.50` and `WHISPER_MODEL=base`, so changing code defaults alone won't switch the runtime. `.env` and any deployment secrets must change together. | HIGH |

---

## F. Medium / low-priority cleanup

| # | Finding | Severity |
|---|---|---|
| F1 | **Answers can be hidden.** `generator.generate` sets `not_found` when the answer contains "could not find" anywhere (`generator.py:426`). The UI then replaces the whole answer with the not-found box (`ui.py:837-845`), so a partial answer that mentions a gap is hidden. Stage 2.6 metrics were measured with this logic, so changing it changes evaluated behaviour; re-evaluate before changing. | MEDIUM |
| F2 | **Stale `.env` variable:** `TOP_K_RESULTS=5` is unused (settings reads `RETRIEVAL_TOP_K`, default 10; `TOP_K_RESULTS` is a code alias), which misleads readers. `ACTIVE_COLLECTION` and `MAX_LLM_EVIDENCE` are not in `.env`. | MEDIUM |
| F3 | **Working-directory dependence:** `.env` sets `CHROMA_DB_PATH=data/vector_db` (relative), so the DB path depends on the working directory. The code default is absolute (`ROOT_DIR/data/vector_db`), but the `.env` value overrides it. | MEDIUM |
| F4 | **Stale self-tests:** the `__main__` blocks in `pipeline.py` (line 771) and `generator.py` (line 690) open the v1 collection `eduvision_chunks`, expect 235 documents / 2 videos and hard-code `data/...` paths. `retriever.py` docstrings describe a 2-video corpus. | LOW |
| F5 | **Legacy v1 code:** `CHROMA_COLLECTION_NAME = "eduvision_chunks"` and the v1 builder in `ingestion/indexer.py`. Production uses only `get_chroma_client` from that module. | LOW |
| F6 | **Stale comment:** `config/settings.py` threshold comment (calibrated on a 2-video corpus, 2026-08-28). | LOW |
| F7 | **Legacy evaluation scripts:** `eval/evaluate.py` and `eval/eval_followup.py` (tracked) call the API and were validated on the old corpus. They should be kept but marked as legacy smoke suites. | LOW |
| F8 | **One-off script:** `archive/patch_t1_stage15.py` (tracked) was a one-time transcript patch for the old index. | LOW |
| F9 | **Inaccurate README claim:** the README says the UI imports nothing from ingestion; `pipeline` imports `ingestion.indexer.get_chroma_client` (which pulls chunker, cleaner and transcriber modules). Harmless (lazy heavy imports) but inaccurate. | LOW |
| F10 | **Broad catches:** `except Exception` in `pipeline._get_openai_client_if_available` and the rewrite path. These are intentional graceful degradation; logging is present for the rewrite. | LOW |
| F11 | **No app logging:** the Streamlit app does not configure logging, so library warnings go to default handlers. | LOW |
| F12 | **Retrieval nondeterminism:** HNSW search can differ between processes in rare cases (Stage 3: one lower-ranked chunk on 1 of 69 queries, no metric effect). | LOW |
| F13 | **Uncited sentences:** about 42% of factual sentences carry no citation (Stage 4 Z1, heuristic); citations that do exist always resolve (100%). Known and documented, not a code defect. | LOW |
| F14 | **Go-to-Timestamp after the index switch:** the Stage 2.6 index reuses production `video_filename` metadata (same build path); verify that buttons still resolve to `videos/*.mp4` after switching. | LOW (verify) |

---

## G. Security findings

| # | Finding | Severity |
|---|---|---|
| G1 | `.env` is git-ignored (`.gitignore:18 *.env`) and untracked; no API keys or secrets in any tracked file (pattern scan); no absolute user paths in tracked code. | OK |
| G2 | **API key handling:** read once from the environment (`config/settings.py`); a placeholder is detected; a missing key degrades to retrieval-only. Error strings can include OpenAI exception text, shown to the user via the "System error" message. | LOW |
| G3 | **User input:** the query is validated (3–500 characters) and rendered with `st.markdown` (no `unsafe_allow_html`); follow-up history is the user's own session. Prompt injection via the question is inherent to RAG and mitigated by the grounding rules. | OK / LOW |
| G4 | **Unescaped HTML:** source-card HTML interpolates `text_en`, the video label and timestamps into `unsafe_allow_html` markup without escaping (`ui.py:463-474`). The corpus is first-party, but it is an HTML course, so transcript text containing tag-like strings can break or alter card rendering. | MEDIUM |
| G5 | **File upload:** none in the UI. | OK |
| G6 | **ffmpeg / ffprobe:** `subprocess.run` with argument lists, no `shell=True`, ingestion-only (offline). The archive script is the same. | OK |
| G7 | **Path traversal:** video paths are `VIDEOS_DIR / video_filename`, where `video_filename` comes from index metadata, not user input; existence is checked before `st.video`. | OK |
| G8 | **Database:** ChromaDB is local and embedded; no network DB, no credentials; anonymized telemetry is disabled in clients. | OK |
| G9 | **Cost and abuse:** the only limit is a per-session 10-question counter (`_QUESTION_LIMIT`), which resets on page reload. A public deployment spends the owner's OpenAI key with no global rate or budget limit. | MEDIUM |
| G10 | **Streamlit config:** `enableCORS=false` and headless mode in `app/.streamlit/config.toml`; no debug flags; usage stats off. Review `enableCORS` against the XSRF default for the target host. | LOW |
| G11 | **Local file permissions:** `.env` file mode is 644 locally. | LOW |

---

## H. Repository / documentation findings

| # | Finding | Severity |
|---|---|---|
| H1 | **README outdated** (see E2): architecture diagram, config table, corpus numbers, evaluation section, project status. | HIGH |
| H2 | **No top-level account of Stages 1–4.** Per-stage READMEs exist (`eval/benchmark/README.md`, `experiments/stage2_transcription/README.md`, `experiments/stage3_retrieval/README.md`, `experiments/stage4_faithfulness/README.md`), but nothing at top level links them or states the final metrics (TEST R@10 0.899, correct 81.5%, hallucination 10.3%). | HIGH |
| H3 | **What to commit:** decide between the evaluation code, tests, benchmark, reports and key result files (small) and the bulky per-run JSON in `eval/results/`, the Z2 sentence dump, and similar. Experiment artifacts are already git-ignored by their own `.gitignore`. | MEDIUM |
| H4 | **`.gitignore` coverage:** venv, `.env`, `videos/`, `data/transcripts`, `data/processed`, `__pycache__`, egg-info and generated eval JSON are covered. `data/vector_db` is tracked on purpose. | OK |
| H5 | **Large tracked files:** `data/vector_db/chroma.sqlite3` (9.5 MB) plus HNSW binaries (18 MB total). Acceptable size, but every index change commits new binaries. | LOW |
| H6 | **Two requirements files:** `requirements.txt` (full, includes `openai-whisper` and `tqdm`) and `app/requirements.txt` (runtime). Both pin versions verified on Python 3.14; the README says 3.10+. | MEDIUM |

---

## I. Deployment-readiness findings

| # | Finding | Severity |
|---|---|---|
| I1 | **Startup command:** `streamlit run app/ui.py` (the UI adds the project root to `sys.path`). | OK |
| I2 | **Environment variables:** `OPENAI_API_KEY` is required for answers; Search Evidence works without it. For Stage 2.6 the deployment must also set `SIMILARITY_THRESHOLD=0.44`, `ACTIVE_COLLECTION=lv2g_translate` and a `CHROMA_DB_PATH` that points at the shipped Stage 2.6 index, unless code defaults are changed. `OPENAI_MODEL`, `TEMPERATURE`, `MAX_TOKENS`, `RETRIEVAL_TOP_K`, `MAX_LLM_EVIDENCE` and `BGE_MODEL` are optional; their defaults match Stage 2.6. | **BLOCKER** (part of D1) |
| I3 | **Runtime dependencies:** `app/requirements.txt` (python-dotenv, torch, FlagEmbedding, numpy, chromadb, openai, streamlit). No ffmpeg or Whisper is needed at runtime. | OK |
| I4 | **Python version:** no pin (`runtime.txt` / `.python-version` absent); pinned wheels (torch 2.13.0, numpy 2.5.2, streamlit 1.62.0) were verified only on Python 3.14.2 locally. Their availability on the target host's Python must be confirmed. | MEDIUM |
| I5 | **Model download and memory:** BGE-M3 (`BAAI/bge-m3`, about 2.2 GB weights) downloads from Hugging Face on first start and loads in fp32. Memory and cold-start time on the target host must be verified. | MEDIUM |
| I6 | **Local-machine assumptions:** `videos/` exists only locally, and playback degrades gracefully (OK). `CHROMA_DB_PATH` is relative in `.env` (F3). The Stage 2.6 index is in a local-only ignored folder (D2). | see refs |
| I7 | **Writable index directory:** ChromaDB writes to its directory on open (E3), so a read-only filesystem on the host would fail; this must be verified. | MEDIUM |

---

## J. Test results

`venv/bin/python -m unittest discover -s tests` → **Ran 144 tests — OK (144 passed, 0 failed, 0 errors).**

| Suite | Tests | Tracked in git? |
|---|---:|---|
| `test_bench_data`, `test_bench_metrics`, `test_run_benchmark`, `test_compare_results`, `test_answer_eval` (Stages 1–2.5) | 83 | no (untracked) |
| `test_stage3_retrieval` (Stage 3) | 33 | no |
| `test_stage4_z1`, `test_stage4_z2`, `test_stage4_z3`, `test_stage4_z3_analysis` (Stage 4) | 28 | no |

- **Committed tests:** the repository has no committed unit tests. Every passing test comes from the untracked Stage 1–4 work, so "pre-existing failures" cannot exist; nothing fails.
- **Not run:** the tracked `eval/evaluate.py` and `eval/eval_followup.py` were not run, because they call the OpenAI API.
- **Gaps:** the tests cover evaluation and experiment code. There are no unit tests for the production modules themselves (`pipeline`, `generator`, `retriever`, the UI helpers), and none asserting that the runtime configuration equals Stage 2.6.

---

## K. Recommended finalization sequence

Each step should be small, reviewed and approved separately. No step requires TEST, new experiments or paid
API calls, except the optional, separately approved smoke test in step 6.

1. **Choose how to ship the Stage 2.6 index.** Copy `experiments/stage2_transcription/artifacts/lv2g_translate/vector_db` to a tracked, documented location (for example a new directory under `data/`). Verify fingerprint `2a3b259d…`, 1,391 chunks and collection `lv2g_translate`. Keep the old index available for rollback. (D2, E3)
2. **Make the runtime configuration Stage 2.6 by default** in `config/settings.py`: `ACTIVE_COLLECTION=lv2g_translate`, `CHROMA_DB_PATH` pointing to the shipped index (absolute via `ROOT_DIR`), `SIMILARITY_THRESHOLD=0.44`, `WHISPER_MODEL=large-v2`. Then update `.env`: drop `TOP_K_RESULTS`, set or remove overriding values, and use an absolute or unset DB path. Do the same for the deployment secrets. (D1, E4, F2, F3, I2)
3. **Add Rule 8 to `generation/generator.SYSTEM_PROMPT`** verbatim from `guard_prompt.txt`, and confirm the full prompt's SHA-256 equals `9b294ecd…`. (D3)
4. **Add a configuration consistency check.** For example, `health_check` or a small offline test asserts collection, count, threshold and prompt SHA against `locked_selection.json`, so a fallback to the baseline is detected. This uses offline unit tests only. (C, J)
5. **Reproducibility.** Either move the Stage 2.6 transcription options (translate, temperature 0, `condition_on_previous_text=False`) into `ingestion/transcriber.py` and make the collection name configurable in the indexer, or document `build_candidate.py` as the official rebuild path with exact commands. (D4)
6. **Verify without paid calls.** Run the 144 offline tests and a retrieval-only smoke check (Search Evidence, gate decisions on a few DEV queries) against the shipped index. An end-to-end answer smoke test needs a few API calls and separate approval. Stage 2.6 TEST results are already locked; do not re-run TEST.
7. **Small safety fixes:** escape HTML in source cards (G4), sanitize user-facing error text (G2), and decide on a global rate or cost limit for the public deployment (G9). Leave `not_found` detection unchanged unless it is re-evaluated (F1).
8. **Repository hygiene.** Decide what to commit: evaluation and experiment code, tests, benchmark, reports and selected small result files; ignore bulky per-run JSON. Mark the legacy `__main__` self-tests, v1 indexer code, the archive script and legacy eval scripts as legacy, or remove them, with approval. (E1, E3, F4–F8, H3)
9. **Documentation.** Rewrite the README for the final system: Stage 2.6 configuration, architecture, env var table, startup, the benchmark and its locked results, Stage 3 and 4 outcomes, and limitations (AI-checked gold labels, AI-assisted Z3, citation coverage). Link the per-stage READMEs. (E2, H1, H2)
10. **Deployment.** Pin the Python version, reconcile `requirements.txt` and `app/requirements.txt`, and verify BGE-M3 memory, cold start and a writable index directory on the target host. Deploy, then run the post-deploy smoke check. (H6, I4, I5, I7)
11. **Commit** in reviewed, logical commits once you approve. Nothing is committed in this audit.

---

## FINALIZATION VERDICT

**Needs fixes before finalization.** The selected Stage 2.6 configuration (index, 0.44 threshold, Rule 8) is not
wired into the production runtime or its deployable assets; steps 1–4 of section K are required before
EduVision can be called finalized.

### Audit bookkeeping

- **Files inspected:**
  - Production code: `README.md`, `requirements.txt`, `app/requirements.txt`, `setup.py`, `.gitignore`, `app/.streamlit/config.toml`, `config/settings.py`, `pipeline.py`, `app/ui.py`, `retrieval/retriever.py`, `generation/generator.py`, `ingestion/{video_processor,transcriber,cleaner,chunker,normalizer,embedder,indexer,indexer_v2}.py`, `archive/patch_t1_stage15.py`.
  - `.env`: variable names and non-secret values only; the API key was not read or printed.
  - Experiments: `experiments/stage2_6_abstention/{locked_selection.json,guard_prompt.txt}`, `experiments/stage2_transcription/build_candidate.py` (index build section).
  - Repo state: `git ls-files` and `git status` output, and directory sizes.
- **Files modified:** none. **Files created:** this report only, `experiments/finalization/FINALIZATION_AUDIT.md`.
- **Tests:** 144 passed, 0 failed (offline unit tests only).
- **Production behaviour changed:** no. 0 changes to tracked files; no API calls; no TEST runs; no commits.
