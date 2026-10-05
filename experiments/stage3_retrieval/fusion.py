"""
experiments/stage3_retrieval/fusion.py
───────────────────────────────────────
Reciprocal Rank Fusion (Cormack, Clarke & Büttcher, 2009). Stage 3 only.

  score(d) = Σ over input rankings of 1 / (k + rank(d)),  rank 1-based;
  a ranking that does not contain d contributes nothing.

Ties are broken by the first ranking's order (the dense ranking in Stage 3), then by item.
Only ranks are used, so the scales of the input scores (cosine, BM25) never meet.
"""

from __future__ import annotations

from typing import Sequence


def rrf_fuse(rankings: Sequence[Sequence[int]], k: int) -> list[tuple[int, float]]:
    """Fused (item, score) pairs, best first."""
    if k <= 0:
        raise ValueError("RRF k must be positive")
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking, start=1):
            scores[item] = scores.get(item, 0.0) + 1.0 / (k + rank)
    primary = {item: r for r, item in enumerate(rankings[0])} if rankings else {}
    unranked = len(primary)
    return sorted(scores.items(), key=lambda t: (-t[1], primary.get(t[0], unranked), t[0]))
