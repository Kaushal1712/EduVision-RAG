"""
experiments/stage3_retrieval/bm25.py
─────────────────────────────────────
Okapi BM25 over the indexed chunk text (text_en), in NumPy. Stage 3 diagnostic only.

Choices, fixed in advance and not tuned on the benchmark:
  * tokens: lower-cased runs of [a-z0-9]. Markup and punctuation split terms
    ("<hr>" → "hr", "z-index" → "z", "index"). No stemming, no stop-word list: IDF already
    discounts words that occur in most chunks, and either would be an extra tuning choice.
  * k1 = 1.2, b = 0.75: the Lucene / Elasticsearch defaults.
  * idf = ln(1 + (N − df + 0.5) / (df + 0.5)), Lucene's variant, which never goes negative.
  * repeated query terms count once per occurrence (the textbook sum over query terms).
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Sequence

import numpy as np

TOKEN_RE = re.compile(r"[a-z0-9]+")
K1 = 1.2
B = 0.75


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


class BM25:
    def __init__(self, docs: Sequence[str], k1: float = K1, b: float = B):
        self.k1, self.b = k1, b
        toks = [tokenize(d) for d in docs]
        self.n_docs = len(toks)
        doc_len = np.array([len(t) for t in toks], dtype=np.float64)
        avgdl = float(doc_len.mean()) if self.n_docs else 0.0
        # Per-document length normalisation, precomputed: k1 * (1 - b + b * dl / avgdl).
        self._norm = k1 * (1.0 - b + b * doc_len / avgdl) if avgdl else np.full(self.n_docs, k1)
        postings: dict[str, tuple[list[int], list[int]]] = {}
        for i, t in enumerate(toks):
            for term, tf in Counter(t).items():
                docs_i, tfs = postings.setdefault(term, ([], []))
                docs_i.append(i)
                tfs.append(tf)
        self._postings = {term: (np.array(d), np.array(f, dtype=np.float64)) for term, (d, f) in postings.items()}
        self.idf = {term: math.log(1.0 + (self.n_docs - len(d) + 0.5) / (len(d) + 0.5))
                    for term, (d, _) in self._postings.items()}

    def scores(self, query: str) -> np.ndarray:
        s = np.zeros(self.n_docs, dtype=np.float64)
        for term in tokenize(query):
            if term not in self._postings:
                continue
            idx, tf = self._postings[term]
            s[idx] += self.idf[term] * tf * (self.k1 + 1.0) / (tf + self._norm[idx])
        return s
