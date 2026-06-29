"""
Groq API LLM client — drop-in replacement for Ollama.
Uses Groq's free tier to serve Llama 3 / Mixtral at high speed.
"""
import logging
import json
import time
import asyncio
from typing import AsyncGenerator
import httpx
import requests

from core.config import settings

logger = logging.getLogger(__name__)


def _query_groq(prompt: str, system_prompt: str | None = None) -> str | None:
    """Send a prompt to Groq API and return the response text."""
    if not settings.llm_api_key:
        logger.error("[GROQ] No API key configured. Set LLM_API_KEY env var.")
        return None

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    try:
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {settings.llm_api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": settings.llm_model,
                "messages": messages,
                "temperature": settings.llm_temperature,
                "max_tokens": settings.llm_max_tokens,
                "top_p": 0.9,
                "seed": settings.llm_seed,
                "stream": False,
            },
            timeout=settings.llm_timeout,
        )
        response.raise_for_status()
        result = response.json()
        content = result.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
        return content

    except requests.exceptions.ConnectionError:
        logger.error("[GROQ] Cannot connect to Groq API. Check your internet connection.")
        return None
    except requests.exceptions.Timeout:
        logger.error(f"[GROQ] Groq API request timed out after {settings.llm_timeout}s")
        return None
    except requests.exceptions.HTTPError as e:
        logger.error(f"[GROQ] HTTP error: {e}")
        if hasattr(e, 'response') and e.response is not None:
            logger.error(f"[GROQ] Response body: {e.response.text}")
        return None
    except Exception as e:
        logger.error(f"[GROQ] Unexpected error: {e}")
        return None


async def _query_groq_stream(
    prompt: str,
    system_prompt: str | None = None,
) -> AsyncGenerator[str, None]:
    """Stream tokens from Groq API."""
    if not settings.llm_api_key:
        logger.error("[GROQ] No API key configured.")
        yield "NOT_FOUND"
        return

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    try:
        async with httpx.AsyncClient(timeout=150.0) as client:
            async with client.stream(
                "POST",
                "https://api.groq.com/openai/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {settings.llm_api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": settings.llm_model,
                    "messages": messages,
                    "temperature": settings.llm_temperature,
                    "max_tokens": settings.llm_max_tokens,
                    "top_p": 0.9,
                    "seed": settings.llm_seed,
                    "stream": True,
                },
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        data_str = line[6:].strip()
                        if data_str == "[DONE]":
                            break
                        try:
                            chunk = json.loads(data_str)
                            delta = chunk.get("choices", [{}])[0].get("delta", {})
                            token = delta.get("content", "")
                            if token:
                                yield token
                        except json.JSONDecodeError:
                            continue
    except Exception as e:
        logger.error(f"[GROQ] Streaming error: {e}")
        yield "NOT_FOUND"


async def _query_groq_stream_with_retry(
    prompt: str,
    system_prompt: str | None = None,
    max_retries: int = 3,
    retry_delay: float = 5.0,
) -> AsyncGenerator[str, None]:
    """Stream from Groq with retry logic."""
    for attempt in range(1, max_retries + 1):
        try:
            async for token in _query_groq_stream(prompt, system_prompt):
                yield token
            return
        except Exception as e:
            logger.warning(f"[GROQ] Stream attempt {attempt}/{max_retries} failed: {e}")
            if attempt < max_retries:
                await asyncio.sleep(retry_delay)
            else:
                logger.error("[GROQ] All streaming attempts failed.")
                yield "NOT_FOUND"
