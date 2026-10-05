"""
eval/bench_data.py
───────────────────
Stage 1 benchmark: loading and validation of the benchmark and corpus manifest.

The benchmark file (eval/benchmark/eduvision_bench_v1.json) is the ground truth.
Its schema is documented in eval/benchmark/README.md. Validation is strict: the
runner refuses to score a benchmark that fails it, so a typo in a gold span cannot
silently turn into a metric.

No project imports — this module is unit-tested in isolation.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from bench_metrics import GoldSpan, is_valid_interval

BENCH_DIR: Path = Path(__file__).resolve().parent / "benchmark"
DEFAULT_BENCHMARK_PATH: Path = BENCH_DIR / "eduvision_bench_v1.json"
DEFAULT_MANIFEST_PATH: Path = BENCH_DIR / "corpus_manifest.json"

SPLITS = {"dev", "test"}
CATEGORIES = {"in_scope", "out_of_scope", "near_miss_oos"}
LANGUAGES = {"en", "hinglish"}
PHRASINGS = {"exact_term", "paraphrase", "rare_term"}
DIFFICULTIES = {"easy", "medium", "hard"}
# draft:            gold read from the production transcripts, not checked against the video
# video_checked:    gold checked by an AI annotator against the video's own audio
#                   (two independent Whisper re-transcriptions) and on-screen frames;
#                   not yet reviewed by a person
# verified:         gold checked against the video by a person
# needs_annotation: in-scope, but no gold span could be located confidently;
#                   excluded from ranking metrics (annotation.notes says why)
STATUSES = {"draft", "video_checked", "verified", "needs_annotation"}
SCORED_STATUSES = {"draft", "video_checked", "verified"}

# Gold spans may end slightly after the probed duration (Whisper segment ends are
# not clamped to the container duration).
DURATION_TOLERANCE_S = 1.0


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_json(path: Path) -> Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def manifest_index(manifest: dict) -> dict[str, dict]:
    """video_id → manifest entry."""
    return {v["video_id"]: v for v in manifest["videos"]}


def gold_spans(item: dict) -> list[GoldSpan]:
    return [GoldSpan(g["video_id"], float(g["start_time"]), float(g["end_time"])) for g in item["gold"]]


def is_scored(item: dict) -> bool:
    """True if the item contributes to the ranking metrics."""
    return item["category"] == "in_scope" and item["annotation"]["status"] in SCORED_STATUSES


def validate_benchmark(bench: dict, manifest: dict) -> list[str]:
    """Return a list of human-readable problems; empty means valid."""
    errors: list[str] = []
    videos = manifest_index(manifest)

    for key in ("benchmark_version", "queries"):
        if key not in bench:
            errors.append(f"benchmark is missing top-level key {key!r}")
    if errors:
        return errors

    seen: set[str] = set()
    for pos, item in enumerate(bench["queries"]):
        qid = item.get("id", f"<position {pos}>")
        def err(msg: str) -> None:
            errors.append(f"{qid}: {msg}")

        if qid in seen:
            err("duplicate id")
        seen.add(qid)

        query = item.get("query")
        if not isinstance(query, str) or not query.strip():
            err("query must be a non-empty string")
        if item.get("split") not in SPLITS:
            err(f"split must be one of {sorted(SPLITS)}")
        category = item.get("category")
        if category not in CATEGORIES:
            err(f"category must be one of {sorted(CATEGORIES)}")
        if item.get("language") not in LANGUAGES:
            err(f"language must be one of {sorted(LANGUAGES)}")
        if item.get("difficulty") not in DIFFICULTIES:
            err(f"difficulty must be one of {sorted(DIFFICULTIES)}")

        phrasing = item.get("phrasing")
        if category == "in_scope" and phrasing not in PHRASINGS:
            err(f"in-scope items need phrasing in {sorted(PHRASINGS)}")
        if category != "in_scope" and phrasing is not None:
            err("phrasing applies to in-scope items only; use null")

        history = item.get("history")
        if not isinstance(history, list):
            err("history must be a list (empty for single-turn queries)")
            history = []
        for turn in history:
            if not isinstance(turn, dict) or turn.get("role") not in ("user", "assistant") \
                    or not isinstance(turn.get("content"), str) or not turn["content"].strip():
                err("history turns need role user|assistant and non-empty content")
                break
        if item.get("followup") is not bool(history):
            err("followup must be true exactly when history is non-empty")

        annotation = item.get("annotation")
        if not isinstance(annotation, dict) or annotation.get("status") not in STATUSES:
            err(f"annotation.status must be one of {sorted(STATUSES)}")
            continue
        status = annotation["status"]

        gold = item.get("gold")
        if not isinstance(gold, list):
            err("gold must be a list")
            continue

        if category != "in_scope" and gold:
            err("out-of-scope items must not have gold spans")
        if status == "needs_annotation":
            if category != "in_scope":
                err("needs_annotation only applies to in-scope items")
            if gold:
                err("needs_annotation items must have empty gold (fill gold and change status)")
        elif category == "in_scope" and not gold:
            err(f"in-scope item with status {status!r} needs at least one gold span")

        spans: list[GoldSpan] = []
        for g in gold:
            vid, start, end = g.get("video_id"), g.get("start_time"), g.get("end_time")
            if not is_valid_interval(vid, start, end):
                err(f"invalid gold span {g!r} (need video_id and 0 <= start_time < end_time)")
                continue
            if vid not in videos:
                err(f"gold video_id {vid!r} is not in the corpus manifest")
                continue
            duration = videos[vid]["duration_s"]
            if end > duration + DURATION_TOLERANCE_S:
                err(f"gold span ends at {end}s, after the video's duration {duration:.1f}s")
            spans.append(GoldSpan(vid, float(start), float(end)))

        # Overlapping/touching spans in one video must be merged into one span;
        # otherwise a single chunk could be counted as covering two "different" spans.
        for i, a in enumerate(spans):
            for b in spans[i + 1:]:
                if a.video_id == b.video_id and min(a.end_time, b.end_time) >= max(a.start_time, b.start_time):
                    err(f"gold spans {a} and {b} overlap or touch; merge them")

    return errors
