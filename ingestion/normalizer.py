"""
ingestion/normalizer.py
───────────────────────
Stage 14 — English Normalisation of Chunk Text.
"""

import json
import logging
import time
from pathlib import Path
from typing import Optional

from config.settings import (
    OPENAI_API_KEY,
    OPENAI_MODEL,
    PROCESSED_DIR,
)
from ingestion.chunker import Chunk

logger = logging.getLogger(__name__)

# ── Quality filter constants ──────────────────────────────────────────────────
# ALL three conditions must hold simultaneously for a chunk to be excluded.
MIN_QUALITY_DURATION_S: float = 3.0   # seconds
MIN_QUALITY_CHARS: int = 25            # minimum meaningful character count
# (Isolated = only 1 Whisper segment — a gap-break orphan with no neighbours)

# ── Cache ─────────────────────────────────────────────────────────────────────
NORMALIZER_CACHE_PATH: Path = PROCESSED_DIR / "normalizer_cache.json"

# ── Sentinel ──────────────────────────────────────────────────────────────────
_PLACEHOLDER_KEY = "your-openai-api-key-here"


# ── Language detection ────────────────────────────────────────────────────────

def _has_urdu_script(text: str) -> bool:
    return any("\u0600" <= c <= "\u06FF" for c in text)


def _is_predominantly_english(text: str) -> bool:
    """
    True if text is already mostly English (no Urdu script, ≥70% ASCII tokens).
    """
    if _has_urdu_script(text):
        return False
    tokens = text.split()
    if not tokens:
        return False
    ascii_tokens = sum(1 for t in tokens if t.isascii())
    return (ascii_tokens / len(tokens)) >= 0.70


# ── Quality filter ────────────────────────────────────────────────────────────

def is_quality_ok(chunk: Chunk) -> bool:
    """
    Return True if this chunk is worth indexing.

    A chunk is low-quality only when ALL three conditions hold simultaneously:
      - duration < 3.0s
      - character count < 25
      - only 1 source segment (isolated gap-break orphan)

    If ANY condition is not met, the chunk is retained.
    """
    is_very_short = chunk.duration < MIN_QUALITY_DURATION_S
    is_few_chars  = len(chunk.text.strip()) < MIN_QUALITY_CHARS
    is_isolated   = len(chunk.source_segment_ids) == 1

    if is_very_short and is_few_chars and is_isolated:
        logger.debug(
            "Quality filter: SKIP %s  dur=%.1fs chars=%d segs=%d  text=%r",
            chunk.chunk_id, chunk.duration, len(chunk.text.strip()),
            len(chunk.source_segment_ids), chunk.text[:40],
        )
        return False
    return True


# ── Cache management ──────────────────────────────────────────────────────────

def _load_cache() -> dict:
    if NORMALIZER_CACHE_PATH.exists():
        try:
            with open(NORMALIZER_CACHE_PATH, encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Could not load normalizer cache: %s", e)
    return {}


def _save_cache(cache: dict) -> None:
    NORMALIZER_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(NORMALIZER_CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2, ensure_ascii=False)


# ── Translation ───────────────────────────────────────────────────────────────

_TRANSLATION_SYSTEM_PROMPT = """\
You are a precise translator for a web development tutorial.
Translate the given transcript excerpt into natural English.

Rules:
1. Translate the MEANING exactly — do not add, remove, or infer information.
2. Preserve all technical terms exactly as written: VS Code, HTML, CSS,
   JavaScript, Chrome, Live Preview, Emmet, npm, Node.js, Git, GitHub.
3. The speaker is a coding instructor talking to students. Keep that tone.
4. Output ONLY the English translation — no explanations, no quotes, no labels.
5. If the text is already English, output it unchanged.
6. If the text is noise or phonetic gibberish with no clear meaning,
   output exactly: [unclear audio]
7. CRITICAL: Your output must use ONLY ASCII characters (a-z, A-Z, 0-9,
   punctuation). Do NOT output any Urdu, Hindi, Arabic, or Devanagari script.
   Every character in your response must be a standard ASCII character.
"""


def _translate_chunk(text: str, client) -> str:
    """Translate chunk text to English.

    Retries once with a stricter instruction if the first response contains
    non-ASCII characters.  If the retry also fails, falls back to
    '[unclear audio]' so that text_en is always genuine English — never
    stored Urdu or other non-ASCII script.
    """
    _STRICT_RETRY_MESSAGES = [
        {"role": "system", "content": _TRANSLATION_SYSTEM_PROMPT},
        {"role": "user",   "content": text.strip()},
        {"role": "assistant", "content": "(previous response contained non-English characters)"},
        {"role": "user",   "content":
            "Your previous response contained Urdu/Arabic script. "
            "Respond ONLY in English using ASCII characters. "
            "Translate the original text to English now."},
    ]

    def _call(messages: list) -> str:
        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=messages,
            max_tokens=512,
            temperature=0.0,
        )
        return resp.choices[0].message.content.strip()

    try:
        result = _call([
            {"role": "system", "content": _TRANSLATION_SYSTEM_PROMPT},
            {"role": "user",   "content": text.strip()},
        ])

        if not result.isascii():
            logger.debug("Non-ASCII in first translation — retrying with strict instruction")
            result = _call(_STRICT_RETRY_MESSAGES)
            if not result.isascii():
                logger.warning(
                    "Retry still non-ASCII for chunk %r — storing [unclear audio]",
                    text[:60],
                )
                result = "[unclear audio]"

        return result
    except Exception as e:
        logger.warning("Translation failed for chunk text %r: %s", text[:60], e)
        return text


def _get_openai_client():
    key = OPENAI_API_KEY
    if not key or key.strip() == _PLACEHOLDER_KEY:
        raise RuntimeError(
            "OPENAI_API_KEY not set. Cannot translate chunks."
        )
    import openai
    return openai.OpenAI(api_key=key)


# ── NormalizedChunk data class ────────────────────────────────────────────────

class NormalizedChunk:
    """
    A chunk augmented with English text and quality metadata.

    Attributes:
        chunk:       Original Chunk (never modified).
        text_raw:    Original transcript text (same as chunk.text).
        text_en:     English-normalised text for retrieval and UI display.
        quality_ok:  False if chunk should be excluded from the v2 index.
        translated:  True if text_en differs meaningfully from text_raw.
    """
    def __init__(
        self,
        chunk: Chunk,
        text_en: str,
        quality_ok: bool,
        translated: bool,
    ):
        self.chunk = chunk
        self.text_raw = chunk.text
        self.text_en = text_en
        self.quality_ok = quality_ok
        self.translated = translated

    @property
    def chunk_id(self) -> str:
        return self.chunk.chunk_id


# ── Main normalisation function ───────────────────────────────────────────────

def normalize_chunks(
    chunks: list[Chunk],
    use_cache: bool = True,
    save_cache: bool = True,
) -> list["NormalizedChunk"]:
    """
    Normalise chunks: apply quality filter and translate to English.

    For each chunk:
      1. Apply quality filter → quality_ok flag
      2. If already English → text_en = text_raw (no API call)
      3. If non-English and quality_ok → translate via GPT-4o-mini
      4. Cache result to avoid re-calling the API on reruns

    Low-quality chunks are included in the returned list (quality_ok=False)
    for auditing. The v2 indexer skips them.

    Args:
        chunks:     Chunk objects from the chunker.
        use_cache:  Use cached translations from prior runs.
        save_cache: Write updated cache to disk after processing.

    Returns:
        List of NormalizedChunk objects, same order as input.
    """
    cache = _load_cache() if use_cache else {}

    needs_translation = [
        c for c in chunks
        if is_quality_ok(c)
        and not _is_predominantly_english(c.text)
        and c.chunk_id not in cache
    ]

    client = None
    if needs_translation:
        try:
            client = _get_openai_client()
            logger.info(
                "Normalizer: %d/%d chunks need translation",
                len(needs_translation), len(chunks),
            )
        except RuntimeError as e:
            logger.warning("Normalizer: %s — falling back to raw text for non-English chunks.", e)

    results: list[NormalizedChunk] = []
    translated_count = 0
    skipped_quality  = 0
    cache_hits       = 0

    for chunk in chunks:
        quality_ok = is_quality_ok(chunk)

        if not quality_ok:
            skipped_quality += 1
            results.append(NormalizedChunk(
                chunk=chunk,
                text_en=chunk.text,
                quality_ok=False,
                translated=False,
            ))
            continue

        if chunk.chunk_id in cache:
            cached = cache[chunk.chunk_id]
            cached_text_en = cached.get("text_en", "")
            # Guard: only trust the cache if text_en is ASCII (English).
            # Entries written before Stage 14 translation have text_en = text_raw
            # (still Urdu/Hinglish). Treat those as cache misses so they get translated.
            if cached_text_en.isascii():
                cache_hits += 1
                results.append(NormalizedChunk(
                    chunk=chunk,
                    text_en=cached_text_en,
                    quality_ok=True,
                    translated=cached.get("translated", False),
                ))
                continue
            # else: fall through to translate below

        if _is_predominantly_english(chunk.text) and chunk.text.isascii():
            text_en    = chunk.text
            translated = False
        elif client is not None:
            text_en = _translate_chunk(chunk.text, client)
            # Only fall back to raw text if the audio is unclear AND the raw text
            # is itself ASCII (English).  For non-ASCII raw text (Urdu script) that
            # couldn't be translated, keep "[unclear audio]" so text_en is always
            # English-safe and never stores Urdu in the index.
            if text_en == "[unclear audio]" and chunk.text.isascii():
                text_en = chunk.text
                translated = False
            elif text_en == "[unclear audio]":
                translated = False   # keep "[unclear audio]" as text_en
            else:
                translated = (text_en != chunk.text)
            translated_count += 1
            time.sleep(0.05)  # avoid rate limiting
        else:
            text_en    = chunk.text
            translated = False

        cache[chunk.chunk_id] = {
            "text_en":    text_en,
            "text_raw":   chunk.text,
            "translated": translated,
        }
        results.append(NormalizedChunk(
            chunk=chunk,
            text_en=text_en,
            quality_ok=True,
            translated=translated,
        ))

    if save_cache and cache:
        _save_cache(cache)
        logger.info("Normalizer cache saved: %d entries", len(cache))

    quality_ok_count = sum(1 for r in results if r.quality_ok)
    logger.info(
        "Normalizer: %d/%d passed quality | %d cache hits | %d translated | %d low-quality skipped",
        quality_ok_count, len(chunks),
        cache_hits, translated_count, skipped_quality,
    )

    return results


# ── Report helper ─────────────────────────────────────────────────────────────

def print_normalization_report(results: list["NormalizedChunk"]) -> None:
    """Print a summary of normalisation decisions to stdout."""
    total       = len(results)
    quality_ok  = [r for r in results if r.quality_ok]
    low_quality = [r for r in results if not r.quality_ok]
    translated  = [r for r in results if r.translated]

    print(f"\n=== NORMALIZER REPORT ===")
    print(f"  Total chunks:          {total}")
    print(f"  Quality OK (indexed):  {len(quality_ok)}")
    print(f"  Low-quality (skipped): {len(low_quality)}")
    print(f"  Translated:            {len(translated)}")
    print(f"  Already English:       {len(quality_ok) - len(translated)}")

    if low_quality:
        print(f"\n  Low-quality chunks excluded from v2 index:")
        for r in low_quality:
            print(
                f"    [{r.chunk.start_time_fmt}\u2192{r.chunk.end_time_fmt}] "
                f"dur={r.chunk.duration:.1f}s segs={len(r.chunk.source_segment_ids)} "
                f"chars={len(r.chunk.text)} | {r.chunk.text!r}"
            )

    if translated:
        print(f"\n  Translation samples (first 5):")
        for r in translated[:5]:
            print(f"    [{r.chunk.start_time_fmt}] RAW: {r.text_raw[:80]}")
            print(f"    [{r.chunk.start_time_fmt}]  EN: {r.text_en[:80]}")
            print()

    print(f"=========================\n")
