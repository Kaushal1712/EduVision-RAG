# Stage 4: citation integrity and claim-level faithfulness — offline closure

**Status: closed offline (2026-10-05). No Stage 4 faithfulness variant is promoted to production.**
DEV only; no TEST evaluation was performed for Stage 4. No OpenAI/API calls were made in Z1–Z3.
Production code, configuration and the index were not changed. The frozen Stage 2.6 configuration
(`experiments/stage2_6_abstention/locked_selection.json`, carried forward from Stage 3) is unchanged.

## 1. Objective

Measure whether EduVision's answers are supported by the evidence it retrieves, with a focus on
citation integrity (does a citation point at evidence the model was given, and at the evidence that
supports the claim) and claim-level faithfulness — without spending API credits, using existing DEV
answers from the locked Stage 2.6 configuration (frozen run, Stage 3 control re-run, and the no-rule-8
variant for comparison).

## 2. Z1 — citation integrity audit (offline, deterministic)

Tested: whether every `[Video: "…" @ MM:SS]` citation resolves to an evidence chunk actually given to the
model (±1 s), and how many factual sentences carry a citation (heuristic).
Report: `z1_citation_integrity_audit.md` · output `z1_citation_integrity_audit.json` · code `z1_citation_audit.py`.

Finding: **626 of 626 citations (100%) point at provided evidence** in all three runs; 0 fabricated
timestamps, 0 wrong videos, 0 unresolvable titles. Almost every cited time is the displayed start of a
provided chunk. Weakness: only about **58%** of factual sentences carry a citation (heuristic). Rule 8
makes no difference to either. Citation validity does not separate faithful from unfaithful answers.

## 3. Z2 — claim–citation alignment proxy (offline, cached BGE-M3)

Tested: for each factual sentence, BGE-M3 dense cosine and lexical overlap against every provided chunk;
cited-chunk rank and margin; a shuffled control (sentence vs another query's chunk).
Report: `z2_alignment_proxy.md` · output `z2_alignment_proxy.json` · code `z2_alignment_proxy.py`.

Finding: the proxy separates **grossly** misaligned citations (cited vs shuffled AUROC 0.97–0.99). Within
the five on-topic provided chunks, the cited chunk is the dense best in 66–69% of cited sentences; 15–19%
are ranked below the best by both signals. Its relation to the existing judge's flags is weak (AUROC
0.55–0.70). It is a similarity proxy, not a support judgement.

## 4. Z3 — verification set and analysis

Tested: 39 stratified DEV sentences (`z3/`), labelled for claim support (A) and citation status (B), then
compared with Z2 and the existing LLM judge.
Files: `z3/README.md`, `z3/z3_annotation_sheet.md`, `z3/z3_labels.csv`, `z3/z3_metadata.csv`,
`z3/z3_annotation_set.json`, `z3/z3_analysis.md`, `z3/z3_analysis.json`; code `z3_prepare_annotation.py`,
`z3/z3_analyze.py`. (`z3/README.md` describes the sheet before labelling; the labels were filled afterwards.)

> **Z3 labels are AI-assisted (one pass by Claude) and NOT human ground truth.**

Labels: A — Supported 24, Partially supported 11, Unsupported 3, Contradicted 1. B — correctly supports 21,
points to evidence but does not support 7, no citation 11, outside provided evidence 0.

Findings (agreement with the AI-assisted labels):
- **Z2 vs Z3, citation alignment:** "misaligned by both signals" found 6 of the 7 citation problems with 2
  false alarms (κ 0.73); it usually names the chunk that does support the claim.
- **Z2 vs Z3, claim support:** weak (AUROC 0.63–0.67; κ ≈ 0 on cited claims). Z2 cannot tell whether a
  claim's content is in the evidence (e.g. S12, a claim no chunk states, was ranked "aligned").
- **Judge vs Z3:** κ 0.07–0.20 at sentence and answer level; 5 judge flags on claims Z3 labels Supported
  (including a verbatim quote, EVB-054); 9 Z3 problem claims not flagged; none of the 7 citation problems
  flagged, and all 7 of those answers graded "fully supported".
- The 7 citation problems are of two kinds: the wrong chunk cited while another provided chunk supports the
  claim (5), and claims no chunk supports (2).

## 5–6. Limitations

- Z3 was labelled by **one AI annotator** in a single pass; **no human adjudication** and no second annotator.
- **Small, biased sample:** 39 sentences stratified to over-represent Z2-misaligned, judge-flagged and
  disagreement cases; rates describe this set, not DEV. 7 citation problems and 15 claim problems only.
- **Six non-blind cases:** S09/S26 (EVB-018), S24 (EVB-054), S23 (EVB-163), S14 (EVB-134), S13 (EVB-086)
  were discussed earlier in the session before labelling.
- **No API evaluation in Z3:** no claim-level LLM judge was run; the existing judge outputs were reused.
- Judge `unsupported_claims` are mapped to sentences heuristically (one miss found and corrected, S24).
- Z1 sentence coverage and Z2 are heuristics/proxies; Z2 uses one embedding model and a fixed stop list.
- Gold spans and reference transcripts come from AI-checked annotation (Stage 1.5), not human verification.

## 7. Conclusion

- **Citation syntax and integrity are strong:** every DEV citation resolves to provided evidence.
- **Z2 is useful for citation-alignment screening** (finding citations that point at a provided chunk other
  than the one that supports the claim), on this AI-labelled evidence.
- **Z2 is not reliable for claim-level faithfulness.**
- **The current LLM judge is not reliable enough for claim- or citation-level Stage 4 decisions** on this
  evidence; its answer-level correctness grading was not assessed here.
- These conclusions rest on AI-assisted labels and need human verification before any Stage 4 metric,
  threshold or intervention depends on them.

## 8–10. Decisions and integrity

- **No Stage 4 faithfulness variant is promoted to production.** No retrieval, reranking or citation
  mechanism was introduced.
- **No TEST evaluation was performed for Stage 4.**
- **No production code, configuration or index was changed.** Stage 4 work lives in
  `experiments/stage4_faithfulness/` and `tests/test_stage4_*.py`; inputs (answer files, benchmark, index)
  were only read.

## 11. Artifacts (preserved)

Z1: `z1_citation_audit.py`, `z1_citation_integrity_audit.json`, `z1_citation_integrity_audit.md`.
Z2: `z2_alignment_proxy.py`, `z2_alignment_proxy.json`, `z2_alignment_proxy.md`.
Z3: `z3_prepare_annotation.py`, `z3/README.md`, `z3/z3_annotation_sheet.md`, `z3/z3_labels.csv`,
`z3/z3_metadata.csv`, `z3/z3_annotation_set.json`, `z3/z3_analyze.py`, `z3/z3_analysis.json`,
`z3/z3_analysis.md`.
Tests: `tests/test_stage4_z1.py`, `tests/test_stage4_z2.py`, `tests/test_stage4_z3.py`,
`tests/test_stage4_z3_analysis.py`.

## 12. Test and integrity status at closure

- `venv/bin/python -m unittest discover -s tests`: **144 tests, all passing** (2026-10-05).
- Tracked files: 0 changes (`git status`); `data/vector_db` unchanged.
- Frozen Stage 2.6 / benchmark / candidate-index artifacts: SHA-256 of 63 files identical to the
  Stage 3 baseline snapshot.
- Nothing committed or pushed.
