"""
experiments/stage3_retrieval/m3sparse.py
─────────────────────────────────────────
BGE-M3 learned sparse ("lexical weight") retrieval. Stage 3 diagnostic only.

BGE-M3's sparse head gives each input token a non-negative weight (FlagEmbedding keeps the
maximum per token id and drops special tokens). A query-passage score is the sum, over token
ids present in both, of query weight × passage weight — what
BGEM3FlagModel.compute_lexical_matching_score computes for one pair. Here it is computed for
all passages at once from an inverted index.

Encoding uses the production query model (retrieval.retriever._get_model: BAAI/bge-m3, fp32,
cached weights, nothing downloaded) with the dense pipeline's lengths: passages max_length 512
in batches of 16, queries max_length 512 one at a time.
"""

from __future__ import annotations

import hashlib
import json
from typing import Sequence

import numpy as np

PASSAGE_BATCH_SIZE = 16
MAX_LENGTH = 512


def encode_sparse(model, texts: Sequence[str], batch_size: int, max_length: int = MAX_LENGTH) -> list[dict[str, float]]:
    """Lexical weights per text as {token_id: weight} (plain floats)."""
    out = model.encode(list(texts), batch_size=batch_size, max_length=max_length,
                       return_dense=False, return_sparse=True, return_colbert_vecs=False)
    return [{str(k): float(v) for k, v in w.items()} for w in out["lexical_weights"]]


def weights_sha256(weights: Sequence[dict[str, float]]) -> str:
    return hashlib.sha256(json.dumps(list(weights), sort_keys=True).encode()).hexdigest()


class SparseIndex:
    def __init__(self, passage_weights: Sequence[dict[str, float]]):
        self.n_docs = len(passage_weights)
        postings: dict[str, tuple[list[int], list[float]]] = {}
        for i, w in enumerate(passage_weights):
            for tok, weight in w.items():
                docs, ws = postings.setdefault(tok, ([], []))
                docs.append(i)
                ws.append(weight)
        self._postings = {t: (np.array(d), np.array(w, dtype=np.float64)) for t, (d, w) in postings.items()}

    def scores(self, query_weights: dict[str, float]) -> np.ndarray:
        s = np.zeros(self.n_docs, dtype=np.float64)
        for tok, qw in query_weights.items():
            if tok in self._postings:
                idx, pw = self._postings[tok]
                s[idx] += qw * pw
        return s
