"""
core/llm_client.py
Thin wrapper around the Groq SDK.
Provides a single async helper `chat_completion()` with:
  - Configurable retry logic (up to 3 attempts with exponential back-off)
  - Structured JSON output enforcement via response_format
  - Timeout handling
  - Graceful error propagation
"""

import asyncio
import json
import logging
from typing import Any

import groq

from core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

# Module-level Groq client (re-used across requests)
_client: groq.AsyncGroq | None = None


def _get_client() -> groq.AsyncGroq:
    global _client
    if _client is None:
        if not settings.groq_api_key:
            raise ValueError(
                "GROQ_API_KEY is not set. Add it to your .env file. "
                "Get a free key at https://console.groq.com/"
            )
        _client = groq.AsyncGroq(api_key=settings.groq_api_key)
    return _client


async def chat_completion(
    system_prompt: str,
    user_message: str,
    *,
    max_retries: int = 3,
    json_mode: bool = False,
    temperature: float = 0.2,
    max_tokens: int = 1024,
) -> str:
    """
    Call the Groq LLM and return the assistant's response text.

    Args:
        system_prompt: The system-level instruction for the model.
        user_message:  The user-turn message.
        max_retries:   Number of retry attempts on transient failures.
        json_mode:     If True, instructs the model to return valid JSON.
        temperature:   Sampling temperature (0.0 = deterministic).
        max_tokens:    Maximum output tokens.

    Returns:
        The raw assistant text response.

    Raises:
        RuntimeError: After all retries are exhausted.
    """
    client = _get_client()
    model = settings.groq_model

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_message},
    ]

    kwargs: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}

    last_error: Exception | None = None

    for attempt in range(1, max_retries + 1):
        try:
            logger.debug("Groq call attempt %d/%d (model=%s)", attempt, max_retries, model)
            response = await client.chat.completions.create(**kwargs)
            content = response.choices[0].message.content or ""
            logger.debug("Groq responded with %d chars", len(content))
            return content

        except groq.RateLimitError as exc:
            wait = 2 ** attempt
            logger.warning("Groq rate-limit hit (attempt %d). Retrying in %ds…", attempt, wait)
            last_error = exc
            await asyncio.sleep(wait)

        except groq.APITimeoutError as exc:
            wait = 2 ** attempt
            logger.warning("Groq timeout (attempt %d). Retrying in %ds…", attempt, wait)
            last_error = exc
            await asyncio.sleep(wait)

        except groq.APIConnectionError as exc:
            wait = 2 ** attempt
            logger.warning("Groq connection error (attempt %d). Retrying in %ds…", attempt, wait)
            last_error = exc
            await asyncio.sleep(wait)

        except Exception as exc:
            logger.error("Groq unexpected error: %s", exc)
            raise RuntimeError(f"Groq API error: {exc}") from exc

    raise RuntimeError(
        f"Groq API failed after {max_retries} attempts. Last error: {last_error}"
    )


async def chat_completion_json(
    system_prompt: str,
    user_message: str,
    *,
    max_retries: int = 3,
    temperature: float = 0.1,
    max_tokens: int = 1024,
) -> dict:
    """
    Like chat_completion() but automatically parses and returns a dict.
    Raises ValueError if the response is not valid JSON.
    """
    raw = await chat_completion(
        system_prompt,
        user_message,
        max_retries=max_retries,
        json_mode=True,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        logger.error("Groq returned non-JSON: %s", raw[:200])
        raise ValueError(f"LLM response is not valid JSON: {exc}") from exc
