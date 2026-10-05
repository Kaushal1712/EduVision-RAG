"""
eval/bench_metrics.py
──────────────────────
Stage 1 benchmark: pure metric functions for timestamp-anchored retrieval evaluation.

═══════════════════════════════════════════════════════════════
RELEVANCE IS DEFINED ON THE VIDEO TIMELINE, NOT ON CHUNK IDS
═══════════════════════════════════════════════════════════════

Ground truth for a query is a list of gold spans (video_id, start_time, end_time).
A retrieved chunk is relevant when it comes from the same video as a gold span and
its [start_time, end_time] interval overlaps that span by more than min_overlap_s
seconds (default 0.0 → any positive overlap; touching endpoints do not count).

Chunk IDs encode the chunk index ("<video_id>_chunk_0042"), so they change whenever
the corpus is re-transcribed or re-chunked. The location of the answer in the video
does not, so the benchmark stays valid across those changes.

═══════════════════════════════════════════════════════════════
METRIC DEFINITIONS
═══════════════════════════════════════════════════════════════

All ranking metrics are computed per query and then averaged over queries.
They are SPAN-based so they do not depend on chunk granularity:

  Recall@k   fraction of the query's gold spans overlapped by ≥1 of the top-k chunks.
             For single-span queries this equals Hit@k.
  MRR@k      1 / rank of the first relevant chunk within the top-k (0 if none).
  nDCG@k     binary gain, credited only when a chunk covers a gold span not already
             covered by a higher-ranked chunk. Several chunks hitting the same span
             earn credit once, so finer chunking cannot inflate the score.
             IDCG = one new span per position for min(n_gold, k) positions.
  Start error  |start_time of the first relevant chunk − start_time of the gold span
             it overlaps| in seconds (min over spans if it overlaps several).

Retrieved entries with invalid metadata (missing video_id, non-finite or inverted
times) and repeated chunk_ids never earn credit; they still occupy their rank.

Abstention (prepared for later stages; nothing here calibrates anything):
  The current system refuses without an LLM call when no retrieved chunk reaches
  SIMILARITY_THRESHOLD. "answered" below means that gate let the query through.
  False-answer rate   out-of-scope queries answered / out-of-scope queries
  False-refusal rate  in-scope queries refused     / in-scope queries
  AUROC               how well top-1 similarity separates in-scope from out-of-scope

No project imports — this module is unit-tested in isolation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence


# ── Data model ────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class GoldSpan:
    video_id:   str
    start_time: float   # seconds
    end_time:   float   # seconds


@dataclass(frozen=True)
class RetrievedChunk:
    rank:       int     # 1-based
    chunk_id:   str
    video_id:   str
    start_time: float   # seconds
    end_time:   float   # seconds
    similarity: float


# ── Span logic ────────────────────────────────────────────────────────────────

def is_valid_interval(video_id: object, start: object, end: object) -> bool:
    """True if the (video_id, start, end) triple describes a usable time span."""
    if not isinstance(video_id, str) or not video_id:
        return False
    if not isinstance(start, (int, float)) or not isinstance(end, (int, float)):
        return False
    if isinstance(start, bool) or isinstance(end, bool):
        return False
    if not (math.isfinite(start) and math.isfinite(end)):
        return False
    return 0.0 <= start < end


def overlap_seconds(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    """Length of the intersection of two intervals (0.0 if disjoint or touching)."""
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def judge_retrieved(
    retrieved: Sequence[RetrievedChunk],
    gold: Sequence[GoldSpan],
    min_overlap_s: float = 0.0,
) -> list[dict]:
    """
    Decide, for each retrieved chunk in rank order, which gold spans it overlaps.

    Returns one dict per retrieved chunk:
      matched_gold: sorted indices into `gold` that this chunk overlaps
      invalid:      True if the chunk's metadata is unusable
      duplicate:    True if the same chunk_id appeared at a higher rank
    """
    judged: list[dict] = []
    seen_ids: set[str] = set()
    for r in retrieved:
        invalid = not is_valid_interval(r.video_id, r.start_time, r.end_time)
        duplicate = r.chunk_id in seen_ids
        seen_ids.add(r.chunk_id)
        matched: list[int] = []
        if not invalid and not duplicate:
            matched = [
                i for i, g in enumerate(gold)
                if g.video_id == r.video_id
                and overlap_seconds(r.start_time, r.end_time, g.start_time, g.end_time) > min_overlap_s
            ]
        judged.append({"matched_gold": matched, "invalid": invalid, "duplicate": duplicate})
    return judged


def covered_fraction(gold: Sequence[GoldSpan], intervals: dict[str, Sequence[tuple[float, float]]]) -> float:
    """
    Fraction of the gold spans' total duration covered by the union of `intervals`
    (per video_id). Used to measure how much of a query's answer the index has
    usable text for, independently of any ranking.
    """
    total = sum(g.end_time - g.start_time for g in gold)
    if total <= 0:
        raise ValueError("covered_fraction requires gold spans with positive duration")
    covered = 0.0
    for g in gold:
        cursor = g.start_time
        for s, e in sorted(intervals.get(g.video_id, [])):
            s, e = max(s, cursor), min(e, g.end_time)
            if e > s:
                covered += e - s
                cursor = e
    return covered / total


# ── Ranking metrics (inputs: per-rank lists of matched gold indices) ──────────

def recall_at_k(matches: Sequence[Sequence[int]], n_gold: int, k: int) -> float:
    if n_gold <= 0:
        raise ValueError("recall_at_k requires at least one gold span")
    covered: set[int] = set()
    for m in matches[:k]:
        covered.update(m)
    return len(covered) / n_gold


def reciprocal_rank(matches: Sequence[Sequence[int]], k: int) -> float:
    for i, m in enumerate(matches[:k]):
        if m:
            return 1.0 / (i + 1)
    return 0.0


def ndcg_at_k(matches: Sequence[Sequence[int]], n_gold: int, k: int) -> float:
    if n_gold <= 0:
        raise ValueError("ndcg_at_k requires at least one gold span")
    covered: set[int] = set()
    dcg = 0.0
    for i, m in enumerate(matches[:k]):
        new = set(m) - covered
        if new:
            dcg += 1.0 / math.log2(i + 2)
            covered.update(m)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(min(n_gold, k)))
    return dcg / idcg


def first_relevant_start_error(
    retrieved: Sequence[RetrievedChunk],
    matches: Sequence[Sequence[int]],
    gold: Sequence[GoldSpan],
    k: int,
) -> Optional[float]:
    """Seconds between the first relevant chunk's start and its gold span's start."""
    for r, m in zip(retrieved[:k], matches[:k]):
        if m:
            return min(abs(r.start_time - gold[i].start_time) for i in m)
    return None


# ── Abstention metrics ────────────────────────────────────────────────────────

def auroc(positive_scores: Sequence[float], negative_scores: Sequence[float]) -> Optional[float]:
    """
    Area under the ROC curve via the Mann–Whitney U statistic (ties count 0.5).
    Returns None when either class is empty (AUROC is undefined).
    """
    if not positive_scores or not negative_scores:
        return None
    wins = 0.0
    for p in positive_scores:
        for n in negative_scores:
            if p > n:
                wins += 1.0
            elif p == n:
                wins += 0.5
    return wins / (len(positive_scores) * len(negative_scores))


def rate(numerator: int, denominator: int) -> Optional[float]:
    return numerator / denominator if denominator else None


# ── Aggregation helpers ───────────────────────────────────────────────────────

def mean(values: Sequence[float]) -> Optional[float]:
    return sum(values) / len(values) if values else None


def percentile(values: Sequence[float], q: float) -> Optional[float]:
    """Linear-interpolation percentile (q in [0, 100]), matching numpy's default."""
    if not values:
        return None
    xs = sorted(values)
    pos = (len(xs) - 1) * q / 100.0
    lo = math.floor(pos)
    hi = math.ceil(pos)
    return xs[lo] + (xs[hi] - xs[lo]) * (pos - lo)
