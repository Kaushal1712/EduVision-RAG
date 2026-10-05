"""
experiments/stage3_retrieval/answer_eval_stage3.py
───────────────────────────────────────────────────
Stage 3: end-to-end answer quality (pipeline.ask → GPT-4o-mini) for a Stage 3 retrieval
variant, DEV only, with eval/answer_eval.py's own generate step and settings.

  venv/bin/python experiments/stage3_retrieval/answer_eval_stage3.py --retrieval control --label s3_control_dev
  venv/bin/python experiments/stage3_retrieval/answer_eval_stage3.py --retrieval colbert_dense20 --label s3_colbert20_dev
  venv/bin/python experiments/stage3_retrieval/answer_eval_stage3.py --retrieval colbert_dense20_denseorder \
      --label s3_colbert20_denseorder_dev
  then:  venv/bin/python eval/answer_eval.py judge --label <label>

Everything except retrieval is the locked Stage 2.6 configuration: lv2g_translate index, threshold
from locked_selection.json, the rule-8 guard prompt, GPT-4o-mini and its temperature, live
follow-up rewriting, MAX_LLM_EVIDENCE, 2 repeats.

  control         production retrieve() unchanged (as the Stage 2.6 answer runs)
  colbert_dense20 pipeline.retrieve is replaced, in this process only, by run_stage3's
                  colbert_dense20 retriever: exact dense top-20 reordered by ColBERT, top-10
                  returned, every chunk keeping its dense cosine as `similarity`, so the
                  generator's 0.44 gate and evidence filter are unchanged.
  colbert_dense20_denseorder
                  the same ColBERT list, except that the evidence the generator will select —
                  the first MAX_LLM_EVIDENCE chunks at or above the threshold, in ColBERT order —
                  is moved to the front and re-sorted by dense cosine (ties keep ColBERT order).
                  generator._select_evidence then passes exactly the ColBERT evidence SET, in
                  dense order. Only the order the LLM sees changes.

The production gate is "some returned chunk >= threshold". The Stage 3 rule is "exact dense
top-1 >= threshold". They can only disagree if reordering pushes every above-threshold chunk out
of the top-10; each call checks this and the count is written to the result (stage3.gate_mismatches).
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path
from typing import Sequence

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "eval"))
sys.path.insert(0, str(HERE))

import run_stage3 as rs  # noqa: E402

GUARD_PROMPT = ROOT / "experiments" / "stage2_6_abstention" / "guard_prompt.txt"
RETRIEVALS = ("control", "colbert_dense20", "colbert_dense20_denseorder")
REPEATS = 2


def dense_order_evidence(results: Sequence, max_evidence: int) -> list:
    """
    Move the evidence _select_evidence would take (first max_evidence above-threshold results)
    to the front, sorted by dense cosine (stable), keep the rest in order, renumber ranks.
    """
    selected = [r for r in results if not r.below_threshold][:max_evidence]
    chosen = {id(r) for r in selected}
    rest = [r for r in results if id(r) not in chosen]
    ordered = sorted(selected, key=lambda r: -r.similarity) + rest
    return [dataclasses.replace(r, rank=n) for n, r in enumerate(ordered, start=1)]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--retrieval", required=True, choices=RETRIEVALS)
    p.add_argument("--label", required=True)
    args = p.parse_args()

    locked = rs.load_json(rs.LOCKED_SELECTION)
    threshold = float(locked["similarity_threshold"])
    rs._bind_environment(threshold)

    import config.settings as s
    import pipeline
    from answer_eval import RESULTS_DIR, cmd_generate
    from run_benchmark import _index_fingerprint
    assert s.ACTIVE_COLLECTION == rs.COLLECTION and s.SIMILARITY_THRESHOLD == threshold
    index = _index_fingerprint(rs.COLLECTION)
    expected = rs.load_json(rs.INDEX_REFERENCE_RUN)["system"]["index"]["sha256"]
    if index["sha256"] != expected:
        raise SystemExit(f"index fingerprint {index['sha256']} != locked {expected}")

    stage3: dict = {"retrieval": args.retrieval, "index_sha256": index["sha256"],
                    "locked_selection_sha256": rs._sha256_text(rs.LOCKED_SELECTION),
                    "code_sha256": {str(f.relative_to(ROOT)): rs._sha256_text(f) for f in sorted(HERE.glob("*.py"))}}
    if args.retrieval.startswith("colbert_dense20"):
        retrieve_fn, _, extras = rs._colbert_dense_retriever(20)(
            rs._Corpus(rs.COLLECTION, threshold), s.RETRIEVAL_TOP_K, rs.POOL_SIZE)
        calls = {"n": 0, "gate_mismatches": []}

        def colbert_retrieve(query, top_k, similarity_threshold, video_id_filter=None, collection_name=None):
            if video_id_filter is not None or collection_name != rs.COLLECTION or top_k != s.RETRIEVAL_TOP_K \
                    or similarity_threshold != threshold:
                raise RuntimeError("Stage 3 retriever called outside the evaluated configuration")
            results = retrieve_fn(query)
            if args.retrieval == "colbert_dense20_denseorder":
                results = dense_order_evidence(results, s.MAX_LLM_EVIDENCE)
            calls["n"] += 1
            dense_gate = extras[query]["dense_top1"] >= threshold
            list_gate = any(not r.below_threshold for r in results)
            if dense_gate != list_gate:
                calls["gate_mismatches"].append(query)
            return results

        pipeline.retrieve = colbert_retrieve
        stage3.update({"retriever": rs.RETRIEVAL_DESCRIPTIONS["colbert_dense20"]
                       + ("; selected evidence re-sorted by dense cosine before generation"
                          if args.retrieval == "colbert_dense20_denseorder" else ""),
                       "index_build": {k: rs._r(v) if isinstance(v, float) else v
                                       for k, v in retrieve_fn.index_info.items()}})
    else:
        stage3["retriever"] = "production retrieve() (ChromaDB HNSW), unchanged"

    cmd_generate(argparse.Namespace(label=args.label, split=rs.SPLIT, repeats=REPEATS, threshold=threshold,
                                    system_prompt_file=str(GUARD_PROMPT.relative_to(ROOT))))
    if args.retrieval.startswith("colbert_dense20"):
        stage3.update({"retrieve_calls": calls["n"], "gate_mismatches": calls["gate_mismatches"]})
    path = RESULTS_DIR / f"answers_{args.label}.json"
    data = json.loads(path.read_text())
    data["stage3"] = stage3
    path.write_text(json.dumps(data, indent=1, ensure_ascii=False) + "\n")
    print(f"stage3 provenance added to {path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
