"""
eval/bench_tools.py
────────────────────
Stage 1 benchmark: annotation and maintenance CLI.

  python eval/bench_tools.py validate                 # schema + span checks, counts
  python eval/bench_tools.py manifest [--check]       # (re)build corpus_manifest.json
  python eval/bench_tools.py find "live preview"      # where is a term said? (all videos)
  python eval/bench_tools.py find "rowspan" --video 5
  python eval/bench_tools.py show 7 400 440           # transcript around a candidate span
  python eval/bench_tools.py todo                     # items still needing human work
  python eval/bench_tools.py asr --model medium       # independent annotation transcripts
  python eval/bench_tools.py frames 7 600 632         # screenshots of a span for review

`find` and `show` read the production Whisper transcripts, the indexed English text
and, if present, the independent annotation transcripts made by `asr`. They never
call the retriever, so annotating with them does not bias gold spans toward what the
current retrieval system ranks highly. They need the local data/ directory.

`asr` is an ANNOTATION AID ONLY. It re-transcribes each video's audio with a stronger
Whisper model (English translation, no conditioning on previous text) and writes to
eval/benchmark/annotation_asr/, which is git-ignored. Nothing in the production
pipeline reads these files; the production transcripts, chunks and index are untouched.

Annotation workflow is described in eval/benchmark/README.md.
"""

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

_EVAL_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_EVAL_DIR.parent))
sys.path.insert(0, str(_EVAL_DIR))

from bench_data import (  # noqa: E402
    BENCH_DIR,
    DEFAULT_BENCHMARK_PATH,
    DEFAULT_MANIFEST_PATH,
    gold_spans,
    is_scored,
    load_json,
    manifest_index,
    sha256_file,
    validate_benchmark,
)

ANNOTATION_ASR_DIR: Path = BENCH_DIR / "annotation_asr"
ASR_SETTINGS = {"language": "hi", "task": "translate", "temperature": 0.0,
                "condition_on_previous_text": False, "fp16": False}


def _fmt(seconds: float) -> str:
    m, s = divmod(seconds, 60)
    return f"{int(m):02d}:{s:04.1f}"


def _resolve_video(arg: str, manifest: dict) -> str:
    """Accept a tutorial number ('7', '07') or a full video_id."""
    videos = manifest_index(manifest)
    if arg in videos:
        return arg
    if arg.isdigit():
        for v in manifest["videos"]:
            if v["tutorial_number"] == int(arg):
                return v["video_id"]
    sys.exit(f"Unknown video {arg!r}. Use a tutorial number or a video_id from the manifest.")


def _load_transcript_rows(video_id: str) -> list[tuple[float, float, str, str]]:
    """(start, end, source, text) rows from raw + cleaned transcripts and indexed text."""
    from config.settings import ACTIVE_COLLECTION, TRANSCRIPTS_DIR
    rows: list[tuple[float, float, str, str]] = []
    raw_path = TRANSCRIPTS_DIR / f"{video_id}.json"
    if raw_path.exists():
        for s in load_json(raw_path)["segments"]:
            rows.append((s["start"], s["end"], "raw", s["text"]))
    from ingestion.indexer import get_chroma_client
    col = get_chroma_client().get_collection(ACTIVE_COLLECTION)
    got = col.get(where={"video_id": {"$eq": video_id}}, include=["metadatas", "documents"])
    for meta, doc in zip(got["metadatas"], got["documents"]):
        rows.append((meta["start_time"], meta["end_time"], "index_en", doc))
    for asr_path in sorted(ANNOTATION_ASR_DIR.glob(f"{video_id}.*.json")):
        asr = load_json(asr_path)
        for s in asr["segments"]:
            rows.append((s["start"], s["end"], f"asr_{asr['model']}", s["text"]))
    rows.sort(key=lambda r: (r[0], r[2]))
    return rows


# ── Subcommands ───────────────────────────────────────────────────────────────

def cmd_validate(args) -> int:
    bench = load_json(Path(args.benchmark))
    manifest = load_json(Path(args.manifest))
    errors = validate_benchmark(bench, manifest)
    for e in errors:
        print(f"  ERROR {e}")
    queries = bench["queries"]
    print(f"Benchmark {bench['benchmark_version']}: {len(queries)} queries, "
          f"{len(errors)} validation error(s)")
    for field in ("split", "category", "language", "phrasing", "difficulty"):
        print(f"  {field:<11}", dict(sorted(Counter(str(q[field]) for q in queries).items())))
    print(f"  {'followup':<11}", dict(Counter(q["followup"] for q in queries)))
    print(f"  {'status':<11}", dict(Counter(q["annotation"]["status"] for q in queries)))
    print(f"  {'scored':<11}", sum(is_scored(q) for q in queries),
          "(in-scope with draft, video_checked or verified gold → ranking metrics)")
    per_video = Counter(g.video_id for q in queries if is_scored(q) for g in gold_spans(q))
    na_video = Counter(q.get("annotation", {}).get("video_hint") for q in queries
                       if q["annotation"]["status"] == "needs_annotation")
    print("  gold spans per video (scored)  |  needs_annotation:")
    for v in manifest["videos"]:
        print(f"    T{v['tutorial_number']:02d}  {per_video.get(v['video_id'], 0):3d}  |  "
              f"{na_video.get(v['video_id'], 0)}")
    return 1 if errors else 0


def cmd_manifest(args) -> int:
    from config.settings import TRANSCRIPTS_DIR, VIDEOS_DIR
    from ingestion.video_processor import _make_video_id

    videos = []
    for mp4 in sorted(VIDEOS_DIR.glob("*.mp4")):
        video_id = _make_video_id(mp4.stem)
        transcript = TRANSCRIPTS_DIR / f"{video_id}.json"
        if not transcript.exists():
            sys.exit(f"Missing transcript for {mp4.name} ({transcript})")
        num = re.match(r"^(\d+)_", mp4.name)
        videos.append({
            "video_id":        video_id,
            "tutorial_number": int(num.group(1)) if num else None,
            "video_filename":  mp4.name,
            "duration_s":      round(float(load_json(transcript)["duration_seconds"]), 3),
            "sha256":          sha256_file(mp4),
        })
    manifest = {
        "description": (
            "Identity of the videos the benchmark's gold timestamps refer to. "
            "video_id is derived from the .mp4 filename by ingestion/video_processor.py "
            "and is what retrieval returns; sha256 pins the exact video file, so gold "
            "timestamps stay valid across re-transcription/re-chunking but must be "
            "re-checked if a video file is replaced or re-encoded. duration_s is the "
            "ffprobe duration recorded at ingestion."
        ),
        "generated_by": "python eval/bench_tools.py manifest",
        "videos": videos,
    }
    out = Path(args.manifest)
    text = json.dumps(manifest, indent=2, ensure_ascii=False) + "\n"
    if args.check:
        same = out.exists() and out.read_text(encoding="utf-8") == text
        print("manifest matches videos/ and transcripts" if same else "manifest DIFFERS from videos/")
        return 0 if same else 1
    out.write_text(text, encoding="utf-8")
    print(f"Wrote {out} ({len(videos)} videos)")
    return 0


def cmd_find(args) -> int:
    manifest = load_json(Path(args.manifest))
    targets = [_resolve_video(args.video, manifest)] if args.video else \
        [v["video_id"] for v in manifest["videos"]]
    needle = args.term.lower()
    hits = 0
    for vid in targets:
        for start, end, src, text in _load_transcript_rows(vid):
            if needle in text.lower():
                hits += 1
                print(f"{vid[:2]}  [{_fmt(start)}–{_fmt(end)}]  {start:7.1f}–{end:7.1f}s  "
                      f"{src:<8} {text[:140]}")
    print(f"{hits} hit(s). Raw transcripts are largely Urdu-script Hindi: absence of a hit "
          f"in Latin script is weak evidence.")
    return 0


def cmd_show(args) -> int:
    manifest = load_json(Path(args.manifest))
    vid = _resolve_video(args.video, manifest)
    lo, hi = args.start - args.margin, args.end + args.margin
    for start, end, src, text in _load_transcript_rows(vid):
        if end > lo and start < hi:
            inside = "*" if end > args.start and start < args.end else " "
            print(f"{inside} [{_fmt(start)}–{_fmt(end)}] {start:7.1f}–{end:7.1f}s {src:<8} {text}")
    return 0


def cmd_todo(args) -> int:
    bench = load_json(Path(args.benchmark))
    for q in bench["queries"]:
        status = q["annotation"]["status"]
        if status == "verified":
            continue
        if args.status and status != args.status:
            continue
        spans = ", ".join(f"T{g['video_id'][:2]} {g['start_time']}–{g['end_time']}s" for g in q["gold"])
        print(f"{q['id']}  [{status}]  {q['query']}")
        if spans:
            print(f"        gold: {spans}")
        if q["annotation"].get("notes"):
            print(f"        note: {q['annotation']['notes']}")
    return 0


def cmd_asr(args) -> int:
    import whisper
    from config.settings import PROCESSED_DIR

    manifest = load_json(Path(args.manifest))
    targets = [_resolve_video(args.video, manifest)] if args.video else \
        [v["video_id"] for v in manifest["videos"]]
    ANNOTATION_ASR_DIR.mkdir(exist_ok=True)
    gitignore = ANNOTATION_ASR_DIR / ".gitignore"
    if not gitignore.exists():
        gitignore.write_text("# Annotation-aid transcripts of course audio: regenerate with\n"
                             "# `python eval/bench_tools.py asr`; not committed.\n*\n!.gitignore\n")
    model = whisper.load_model(args.model, device=args.device)
    for vid in targets:
        out = ANNOTATION_ASR_DIR / f"{vid}.{args.model}.json"
        if out.exists() and not args.force:
            print(f"skip {out.name} (exists)")
            continue
        audio = PROCESSED_DIR / f"{vid}.wav"   # 16 kHz mono audio extracted at ingestion
        if not audio.exists():
            sys.exit(f"Missing {audio}; run ingestion/video_processor.py first.")
        result = model.transcribe(str(audio), **ASR_SETTINGS)
        payload = {
            "video_id": vid, "model": args.model, "settings": ASR_SETTINGS,
            "purpose": "benchmark annotation aid only; not used by the EduVision pipeline",
            "segments": [{"start": round(s["start"], 2), "end": round(s["end"], 2), "text": s["text"].strip()}
                         for s in result["segments"]],
        }
        out.write_text(json.dumps(payload, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"wrote {out.name} ({len(payload['segments'])} segments)", flush=True)
    return 0


def cmd_frames(args) -> int:
    import subprocess
    from config.settings import VIDEOS_DIR

    manifest = load_json(Path(args.manifest))
    vid = _resolve_video(args.video, manifest)
    filename = manifest_index(manifest)[vid]["video_filename"]
    out_dir = Path(args.out) / vid[:2]
    out_dir.mkdir(parents=True, exist_ok=True)
    t = args.start
    while t <= args.end:
        target = out_dir / f"{vid[:2]}_{t:07.1f}s.jpg"
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-ss", str(t), "-i", str(VIDEOS_DIR / filename),
                        "-frames:v", "1", "-vf", f"scale={args.width}:-2", "-q:v", "4", str(target)], check=True)
        print(target)
        t += args.every
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--benchmark", default=str(DEFAULT_BENCHMARK_PATH))
    p.add_argument("--manifest", default=str(DEFAULT_MANIFEST_PATH))
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("validate")
    m = sub.add_parser("manifest")
    m.add_argument("--check", action="store_true", help="compare instead of writing")
    f = sub.add_parser("find")
    f.add_argument("term")
    f.add_argument("--video")
    s = sub.add_parser("show")
    s.add_argument("video")
    s.add_argument("start", type=float)
    s.add_argument("end", type=float)
    s.add_argument("--margin", type=float, default=15.0)
    t = sub.add_parser("todo")
    t.add_argument("--status", choices=["draft", "needs_annotation"])
    a = sub.add_parser("asr")
    a.add_argument("--model", default="medium")
    a.add_argument("--video")
    a.add_argument("--force", action="store_true")
    a.add_argument("--device", default="cpu", help="cpu or mps")
    fr = sub.add_parser("frames")
    fr.add_argument("video")
    fr.add_argument("start", type=float)
    fr.add_argument("end", type=float)
    fr.add_argument("--every", type=float, default=10.0)
    fr.add_argument("--width", type=int, default=960)
    fr.add_argument("--out", default="frames")
    args = p.parse_args()
    return {"validate": cmd_validate, "manifest": cmd_manifest, "find": cmd_find, "show": cmd_show,
            "todo": cmd_todo, "asr": cmd_asr, "frames": cmd_frames}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
