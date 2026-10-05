"""
experiments/stage2_transcription/transcript_quality.py
───────────────────────────────────────────────────────
Retrieval-independent comparison of a candidate transcription with production.

Per video:
  * speech coverage: share of the video covered by cleaned segments;
  * indexable coverage: share covered by chunks the normalizer keeps (quality_ok);
  * looping: share of raw segments the cleaner drops as repetitive;
  * technical-term recall: occurrences of course vocabulary (Latin script) in the
    cleaned text, plus how many distinct terms appear at all.

  venv/bin/python experiments/stage2_transcription/transcript_quality.py --name lv2g_transcribe [--videos 3 4]
  venv/bin/python experiments/stage2_transcription/transcript_quality.py --name X --show 8 190 310
"""

import argparse
import json
import re
import sys
from pathlib import Path

EXP_DIR = Path(__file__).resolve().parent
ROOT = EXP_DIR.parent.parent
sys.path.insert(0, str(ROOT))

from config.settings import PROCESSED_DIR, TRANSCRIPTS_DIR  # noqa: E402

MANIFEST = json.loads((ROOT / "eval/benchmark/corpus_manifest.json").read_text())
VIDEOS = {v["tutorial_number"]: v for v in MANIFEST["videos"]}

# Course vocabulary that a listener would expect to be written in Latin script.
TERMS = [
    "html", "css", "javascript", "vs code", "browser", "server", "client", "chrome", "emmet", "extension",
    "live preview", "doctype", "head", "body", "title", "meta", "div", "span", "tag", "attribute", "anchor",
    "href", "paragraph", "lorem", "heading", "h1", "image", "src", "alt", "table", "rowspan", "colspan",
    "caption", "thead", "list", "ul", "ol", "li", "form", "input", "label", "radio", "checkbox", "select",
    "textarea", "placeholder", "required", "autofocus", "inline", "block", "display", "id", "class",
    "video", "audio", "controls", "autoplay", "loop", "muted", "poster", "svg", "iframe", "semantic",
    "header", "footer", "nav", "article", "aside", "figure", "figcaption", "entity", "nbsp", "pre", "code",
    "selector", "pseudo", "hover", "margin", "padding", "border", "box model", "box sizing", "lighthouse",
    "seo", "core web vitals", "github", "git", "flex", "style", "stylesheet", "internal", "external",
]


def _segments(path: Path) -> list[dict]:
    return json.loads(path.read_text())["segments"] if path.exists() else []


def _union(intervals) -> float:
    total, end = 0.0, -1.0
    for s, e in sorted(intervals):
        s = max(s, end)
        if e > s:
            total += e - s
            end = e
    return total


def _term_counts(texts: list[str]) -> dict[str, int]:
    blob = " ".join(texts).lower()
    return {t: len(re.findall(r"(?<![a-z0-9])" + re.escape(t) + r"(?![a-z0-9])", blob)) for t in TERMS}


def video_stats(vid: str, raw: list[dict], cleaned: list[dict], chunks: list[dict], duration: float) -> dict:
    from ingestion.chunker import Chunk
    from ingestion.normalizer import is_quality_ok
    kept = [c for c in chunks if is_quality_ok(Chunk(**c))]
    counts = _term_counts([s["text"] for s in cleaned])
    return {
        "speech_coverage": round(_union((s["start"], s["end"]) for s in cleaned) / duration, 3),
        "indexable_coverage": round(_union((c["start_time"], c["end_time"]) for c in kept) / duration, 3),
        "raw_segments": len(raw), "cleaned_segments": len(cleaned), "chunks": len(chunks),
        "dropped_as_repetitive": round(1 - len(cleaned) / len(raw), 3) if raw else None,
        "tech_term_occurrences": sum(counts.values()),
        "distinct_tech_terms": sum(1 for v in counts.values() if v),
    }


def production(vid: str):
    raw = _segments(TRANSCRIPTS_DIR / f"{vid}.json")
    cleaned = _segments(TRANSCRIPTS_DIR / f"{vid}_cleaned.json")
    chunks = json.loads((PROCESSED_DIR / "chunks" / f"{vid}_chunks.json").read_text())["chunks"]
    return raw, cleaned, chunks


def candidate(name: str, vid: str):
    art = EXP_DIR / "artifacts" / name
    raw = _segments(art / "transcripts" / f"{vid}.json")
    cleaned = _segments(art / "transcripts" / f"{vid}_cleaned.json")
    chunks = json.loads((art / "chunks" / f"{vid}_chunks.json").read_text())["chunks"]
    return raw, cleaned, chunks


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True)
    p.add_argument("--videos", type=int, nargs="*")
    p.add_argument("--show", type=float, nargs=3, metavar=("VIDEO", "START", "END"))
    p.add_argument("--out")
    args = p.parse_args()

    if args.show:
        t, lo, hi = int(args.show[0]), args.show[1], args.show[2]
        vid = VIDEOS[t]["video_id"]
        for label, (_, cleaned, _) in (("PRODUCTION", production(vid)), ("CANDIDATE", candidate(args.name, vid))):
            print(f"--- {label}")
            for s in cleaned:
                if s["end"] > lo and s["start"] < hi:
                    print(f"  {s['start']:7.1f} {s['text']}")
        return 0

    nums = args.videos or sorted(VIDEOS)
    report, totals = {}, {"production": {}, "candidate": {}}
    print(f"{'video':<6}{'speech cov':>18}{'indexable cov':>18}{'repetitive drop':>18}{'tech terms':>16}{'distinct':>12}")
    for n in nums:
        v = VIDEOS[n]
        vid = v["video_id"]
        art_chunks = EXP_DIR / "artifacts" / args.name / "chunks" / f"{vid}_chunks.json"
        if not art_chunks.exists():
            continue
        ps = video_stats(vid, *production(vid), v["duration_s"])
        cs = video_stats(vid, *candidate(args.name, vid), v["duration_s"])
        report[f"T{n:02d}"] = {"production": ps, "candidate": cs, "duration_s": v["duration_s"]}
        print(f"T{n:02d}  " + "".join(f"{ps[k]!s:>8} → {cs[k]!s:<7}" for k in
              ("speech_coverage", "indexable_coverage", "dropped_as_repetitive", "tech_term_occurrences",
               "distinct_tech_terms")))
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
