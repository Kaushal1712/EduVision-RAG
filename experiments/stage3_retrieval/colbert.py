"""
experiments/stage3_retrieval/colbert.py
────────────────────────────────────────
BGE-M3 multi-vector (ColBERT) rescoring. Stage 3 only.

BGE-M3's ColBERT head gives one unit-norm vector per input token (CLS excluded). The score of
a passage for a query is MaxSim averaged over query tokens:

  score(q, p) = (1 / |q|) · Σ_i max_j  q_i · p_j

— the same formula as BGEM3FlagModel.colbert_score, computed here in float64 NumPy for a whole
candidate list. Encoding uses the production query model (retrieval.retriever._get_model:
BAAI/bge-m3, fp32, cached weights, nothing downloaded) with the dense pipeline's lengths.
"""

from __future__ import annotations

import hashlib
from typing import Sequence

import numpy as np

PASSAGE_BATCH_SIZE = 16
MAX_LENGTH = 512


def encode_colbert(model, texts: Sequence[str], batch_size: int, max_length: int = MAX_LENGTH) -> list[np.ndarray]:
    out = model.encode(list(texts), batch_size=batch_size, max_length=max_length,
                       return_dense=False, return_sparse=False, return_colbert_vecs=True)
    return [np.asarray(v, dtype=np.float32) for v in out["colbert_vecs"]]


def vecs_sha256(vecs: Sequence[np.ndarray]) -> str:
    h = hashlib.sha256()
    for v in vecs:
        h.update(np.asarray(v.shape, dtype=np.int64).tobytes())
        h.update(np.ascontiguousarray(v, dtype=np.float32).tobytes())
    return h.hexdigest()


def colbert_score(q: np.ndarray, p: np.ndarray) -> float:
    sims = q.astype(np.float64) @ p.astype(np.float64).T
    return float(sims.max(axis=1).sum() / q.shape[0])


def colbert_rerank(q: np.ndarray, candidates: Sequence[int], passage_vecs: Sequence[np.ndarray]) -> list[tuple[int, float]]:
    """Candidates reordered by ColBERT score, best first; ties keep the input (dense) order."""
    scored = [(c, colbert_score(q, passage_vecs[c])) for c in candidates]
    order = sorted(range(len(scored)), key=lambda n: (-scored[n][1], n))
    return [scored[n] for n in order]
