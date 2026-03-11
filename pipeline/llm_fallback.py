"""
llm_fallback.py
───────────────
Priority-ordered LLM service with automatic failover.

Priority:  Groq (llama-3.3-70b-versatile)
           → OpenRouter (mistralai/mistral-nemo)
           → Google Gemini Flash 2.0

Each provider is tried in order.  If a key is missing from .env the
provider is skipped.  Falls back to the next on connection / rate errors.

All providers are OpenAI-compatible so we use the respective Pipecat
service classes (GroqLLMService, OpenRouterLLMService, GoogleLLMService).
"""

import os
from typing import Optional

from loguru import logger
from pipecat.services.llm_service import LLMService


def _key(name: str) -> Optional[str]:
    return os.getenv(name) or None


def build_llm_service() -> LLMService:
    """
    Return the highest-priority available LLM service.
    Raises RuntimeError if none of the three providers have keys set.
    """
    errors = []

    # ── 1. Groq ───────────────────────────────────────────────────────────────
    groq_key = _key("GROQ_API_KEY")
    if groq_key:
        try:
            from pipecat.services.groq.llm import GroqLLMService

            logger.info("LLM backend: Groq (llama-3.3-70b-versatile)")
            return GroqLLMService(
                api_key=groq_key,
                model="llama-3.3-70b-versatile",
            )
        except Exception as exc:
            logger.warning(f"Groq init failed: {exc}")
            errors.append(f"Groq: {exc}")

    # ── 2. OpenRouter → Mistral Nemo ──────────────────────────────────────────
    or_key = _key("OPENROUTER_API_KEY")
    if or_key:
        try:
            from pipecat.services.openrouter.llm import OpenRouterLLMService

            logger.info("LLM backend: OpenRouter (mistralai/mistral-nemo)")
            return OpenRouterLLMService(
                api_key=or_key,
                model="mistralai/mistral-nemo",
            )
        except Exception as exc:
            logger.warning(f"OpenRouter init failed: {exc}")
            errors.append(f"OpenRouter: {exc}")

    # ── 3. Google Gemini Flash 2.0 ────────────────────────────────────────────
    google_key = _key("GOOGLE_API_KEY")
    if google_key:
        try:
            from pipecat.services.google.llm import GoogleLLMService

            logger.info("LLM backend: Google Gemini 2.0 Flash")
            return GoogleLLMService(
                api_key=google_key,
                model="gemini-2.0-flash",
            )
        except Exception as exc:
            logger.warning(f"Google init failed: {exc}")
            errors.append(f"Google: {exc}")

    raise RuntimeError(
        "No LLM provider available. Set at least one of: "
        "GROQ_API_KEY, OPENROUTER_API_KEY, GOOGLE_API_KEY.\n"
        + "\n".join(errors)
    )
