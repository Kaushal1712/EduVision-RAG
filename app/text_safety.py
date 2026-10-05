"""
app/text_safety.py
───────────────────
Small helpers for text the UI shows to users. Pure functions; no Streamlit import.

- escape(): for any value interpolated into HTML that is rendered with unsafe_allow_html=True
  (transcript text, video titles, timestamps, configuration names).
- user_facing_error(): pipeline/generator error strings can contain raw exception text
  (file paths, collection names, OpenAI error details). Users get a short, fixed message; the
  full error is logged on the server.
"""

from __future__ import annotations

import html
import logging

logger = logging.getLogger(__name__)

GENERIC_ERROR = "Something went wrong while answering. Please try again."

# Prefixes of the error strings built in pipeline.ask() and generation.generator.generate().
_ERROR_MESSAGES: tuple[tuple[str, str], ...] = (
    ("OPENAI_API_KEY is not set", "Answers are unavailable: no OpenAI API key is configured. "
                                  "Search Evidence still works."),
    ("Invalid API key", "The answer service is not configured correctly. Search Evidence still works."),
    ("OpenAI rate limit", "The answer service is busy right now. Please try again in a moment."),
    ("Could not connect to OpenAI", "Could not reach the answer service. Please try again."),
    ("OpenAI API error", "The answer service returned an error. Please try again."),
    ("Retrieval error", "The course index could not be searched. Please try again later."),
)


def escape(value: object) -> str:
    """HTML-escape a value for interpolation into unsafe_allow_html markup."""
    return html.escape(str(value), quote=True)


def user_facing_error(error: str) -> str:
    """Fixed message for the UI; the original error is logged, never shown."""
    logger.error("Pipeline error shown to user as a generic message: %s", error)
    for prefix, message in _ERROR_MESSAGES:
        if error.startswith(prefix):
            return message
    return GENERIC_ERROR
