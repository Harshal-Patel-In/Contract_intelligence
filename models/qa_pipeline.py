import logging
import re
import requests
import asyncio
from typing import AsyncGenerator

from core.config import settings
from models.groq_llm import _query_groq, _query_groq_stream_with_retry

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are an expert legal contract analyst. Your job is to extract precise information from the contract text.

RULES:
1. Provide a direct, factual answer based ONLY on the provided text.
2. If the information is genuinely absent or not specified in the text, respond with exactly: NOT_FOUND
3. Do not assume or extrapolate. If it's not in the text, say NOT_FOUND.
4. Keep the answer concise and to the point. Summarize clauses in 1-3 sentences.
5. If a clause exists but uses different terminology than the question (e.g. "hold harmless" for indemnification), still extract it."""

CATEGORY_HINTS = {
    "Document Name": (
        "Look for the title at the very top of the document, often in ALL CAPS or bold. "
        "Check recitals and the first paragraph. Common formats: 'COOPERATION AGREEMENT', "
        "'LICENSE, DEVELOPMENT AND COMMERCIALIZATION AGREEMENT', 'MASTER SERVICES AGREEMENT'."
    ),
    "Parties": (
        "Look for: 'by and between', 'entered into by', 'among', signature blocks at the end, "
        "and recitals ('WHEREAS'). Include ALL parties with their full legal names including "
        "suffixes (Inc., Corp., LLC, L.P., Ltd.). Check both the preamble AND signature pages."
    ),
    "Effective Date": (
        "Look for: 'as of', 'dated', 'effective date', 'entered into on', 'made and entered into as of'. "
        "Also check signature dates at the end of the document and recital paragraphs."
    ),
    "Expiration Date": (
        "Look for: 'term', 'expires', 'end date', 'initial term of X years', 'termination date', "
        "'standstill period', 'cooperation period'. If a fixed term is stated from the effective date, "
        "state the term duration. Also look for 'until' clauses."
    ),
    "Governing Law": (
        "Look for: 'governed by the laws of', 'jurisdiction', 'venue', 'forum selection', "
        "'applicable law', 'construed in accordance with'. Include both the governing law state/country "
        "AND any specific court or arbitration venue if mentioned."
    ),
    "Assignment": (
        "Look for: 'assign', 'transfer', 'delegate', 'change of control', 'successor', "
        "'binding upon successors'. State whether assignment requires consent and any exceptions "
        "like mergers or affiliates."
    ),
    "Renewal Term": (
        "Look for: 'auto-renew', 'automatic renewal', 'successive periods', 'extend', "
        "'evergreen', 'unless terminated', 'notice of non-renewal'. Include the renewal "
        "period length and how to opt out."
    ),
    "Payment Terms": (
        "Look for: dollar amounts, fee schedules, royalties, milestones, 'net 30', 'net 60', "
        "'within X days', payment timing, invoicing requirements, late payment penalties, "
        "interest rates. Include ALL financial terms."
    ),
    "Limitation of Liability": (
        "Look for: 'limitation of liability', 'aggregate liability shall not exceed', 'cap', "
        "'consequential damages', 'direct damages', 'special damages', 'IN NO EVENT SHALL'. "
        "State the cap amount and what types of damages are excluded."
    ),
    "Indemnification": (
        "Look for: 'indemnify', 'hold harmless', 'defend', 'indemnification', 'third party claims', "
        "'losses', 'damages'. Describe EACH party's indemnification obligations separately. "
        "Note if indemnification is mutual or one-sided."
    ),
    "Non-Compete": (
        "Look for: 'non-compete', 'covenant not to compete', 'restricted business', "
        "'competitive activity', 'shall not engage in'. Include geographic scope, time period, "
        "and what activities are restricted."
    ),
    "Confidentiality": (
        "Look for: 'confidential information', 'non-disclosure', 'proprietary information', "
        "'trade secret', 'NDA', 'confidentiality agreement'. Include the definition of "
        "confidential information, duration, and any carve-outs/exceptions."
    ),
    "Non-Solicitation": (
        "Look for: 'non-solicitation', 'shall not solicit', 'shall not hire', 'shall not recruit', "
        "'employees of the other party', 'customers'. Include time period and scope."
    ),
    "Termination for Convenience": (
        "Look for: 'without cause', 'for convenience', 'at any time', 'upon X days notice', "
        "'either party may terminate', 'for any reason or no reason'. Include the notice period "
        "and any post-termination obligations."
    ),
    "Termination for Cause": (
        "Look for: 'material breach', 'cure period', 'default', 'for cause', 'failure to perform', "
        "'insolvency', 'bankruptcy'. List ALL termination triggers and their cure periods. "
        "Also check for cross-default provisions."
    ),
    "Intellectual Property Ownership": (
        "Look for: 'work product', 'intellectual property', 'inventions', 'work for hire', "
        "'license grant', 'ownership', 'background IP', 'foreground IP', 'improvements'. "
        "State who owns what and any license-back provisions."
    ),
}


def _query_llm(prompt: str) -> str | None:
    """
    Route the query to the configured LLM provider.
    Supports 'groq' (free API) and 'ollama' (local).
    """
    if settings.llm_provider == "groq":
        return _query_groq(prompt, SYSTEM_PROMPT)
    return _query_ollama(prompt)


async def _query_llm_stream(
    prompt: str,
    max_retries: int = 3,
    retry_delay: float = 5.0,
) -> AsyncGenerator[str, None]:
    """Stream the query from the configured LLM provider."""
    if settings.llm_provider == "groq":
        async for token in _query_groq_stream_with_retry(
            prompt, SYSTEM_PROMPT, max_retries, retry_delay
        ):
            yield token
    else:
        # For non-streaming ollama fallback, yield the full response
        result = _query_ollama(prompt)
        if result:
            yield result
        else:
            yield "NOT_FOUND"


def _query_ollama(prompt: str) -> str:
    """Send a prompt to Ollama and return the response text."""
    try:
        response = requests.post(
            settings.ollama_url,
            json={
                "model": settings.ollama_model,
                "prompt": prompt,
                "system": SYSTEM_PROMPT,
                "stream": False,
                "options": {
                    "temperature": settings.ollama_temperature,
                    "top_p": 0.9,
                    "num_predict": settings.ollama_max_tokens,
                    "num_ctx": settings.ollama_num_ctx,
                    "seed": settings.ollama_seed,
                }
            },
            timeout=settings.ollama_timeout,
        )
        response.raise_for_status()
        result = response.json()
        return result.get("response", "").strip()
    except requests.exceptions.ConnectionError:
        logger.error("[QA] Cannot connect to Ollama. Is it running? (ollama serve)")
        return None
    except requests.exceptions.Timeout:
        logger.error(f"[QA] Ollama request timed out after {settings.ollama_timeout}s")
        return None
    except Exception as e:
        logger.error(f"[QA] Ollama error: {e}")
        return None


def _clean_answer(answer: str) -> str:
    """Clean up LLM response artifacts."""
    if not answer:
        return answer

    for prefix in ["Answer:", "The answer is:", "Based on the text,",
                    "According to the contract,", "Based on the contract text,",
                    "Based on the provided text,", "From the contract text,"]:
        if answer.lower().startswith(prefix.lower()):
            answer = answer[len(prefix):].strip()

    answer = re.sub(r'\s*[Nn][Oo][Tt]_?[Ff][Oo][Uu][Nn][Dd]\b[^.\n]*[.\n]?', '', answer).strip()
    answer = re.sub(r'\s*NOT_FOUND\s*', '', answer, flags=re.IGNORECASE).strip()

    lines = answer.split('\n')
    cleaned_lines = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if re.match(r'^\s*(not_?found|n/a|none|no\s+information|not\s+specified|not\s+mentioned)\s*\.?\s*$', stripped, re.IGNORECASE):
            continue
        cleaned_lines.append(line)
    answer = '\n'.join(cleaned_lines).strip()

    if answer:
        lines = answer.rstrip().split('\n')
        last_line = lines[-1].strip()
        if last_line and last_line[-1] not in '.!?)"\'':
            last_period = answer.rfind('.')
            last_question = answer.rfind('?')
            last_end = max(last_period, last_question)
            if last_end > len(answer) * 0.4:
                answer = answer[:last_end + 1]

    return answer.strip()


def _compute_confidence(answer: str, query: str, category: str) -> float:
    """Compute a meaningful confidence score based on answer quality signals."""
    if not answer:
        return 0.0

    score = 7.0

    legal_terms = [
        "shall", "hereby", "pursuant", "notwithstanding", "thereof", "herein",
        "agrees to", "obligation", "covenant", "warranty", "represents",
    ]
    has_legal_language = any(term in answer.lower() for term in legal_terms)
    if has_legal_language:
        score += 0.5

    has_specifics = bool(re.search(r'\b\d+\b', answer))  # numbers
    has_dates = bool(re.search(r'\b\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4}\b|\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b', answer, re.IGNORECASE))
    has_money = bool(re.search(r'\$[\d,]+|\b\d+\s*(?:dollars|USD|EUR|GBP)\b', answer, re.IGNORECASE))
    has_entities = bool(re.search(r'\b(?:Inc\.|Corp\.|LLC|L\.P\.|Ltd\.|Company|Corporation)\b', answer))

    if has_dates:
        score += 0.5
    if has_money:
        score += 0.5
    if has_entities:
        score += 0.5
    if has_specifics:
        score += 0.3

    if 20 < len(answer) < 500:
        score += 0.2

    if len(answer) < 5:
        score = max(3.0, score - 3.0)
    elif len(answer) > 1500:
        score = max(5.0, score - 1.0)

    query_words = set(query.lower().split())
    answer_words = set(answer.lower().split())
    if len(answer_words) > 0:
        overlap = len(query_words & answer_words) / len(answer_words)
        if overlap > 0.6:
            score = max(4.0, score - 2.0)

    return min(10.0, round(score, 1))


def answer_question(query: str, chunks: list, category: str = "") -> dict:
    """
    Uses configured LLM (Groq or Ollama) to extract answers from contract chunks.

    Builds a category-aware prompt with extraction hints and chain-of-thought
    reasoning for precise clause extraction.
    """
    if not chunks:
        logger.warning(f"[QA] No chunks for query: {query[:60]}")
        return {"answer": None, "score": 0.0, "error": "No context chunks provided"}

    context_parts = []
    for chunk in chunks:
        if isinstance(chunk, dict):
            section = chunk.get("section_title", "")
            text = chunk.get("text", "")
            if section:
                context_parts.append(f"[Section: {section}]\n{text}")
            else:
                context_parts.append(text)
        else:
            context_parts.append(str(chunk))

    context = "\n\n---\n\n".join(context_parts)
    max_chars = settings.max_context_chars
    if len(context) > max_chars:
        context = context[:max_chars]

    hint = CATEGORY_HINTS.get(category, "")
    hint_block = f"\nEXTRACTION HINTS: {hint}\n" if hint else ""

    prompt = f"""Extract the answer to this question from the contract text below.
If the information is genuinely not present in the text, respond with exactly: NOT_FOUND
{hint_block}
Think step by step: first identify which section is relevant, then extract the specific answer.

QUESTION: {query}

CONTRACT TEXT:
{context}

ANSWER:"""

    logger.info(f"[QA] Querying {settings.llm_provider}/{settings.llm_model} for: {query[:60]}")

    raw_answer = _query_llm(prompt)

    if raw_answer is None:
        return {"answer": None, "score": 0.0, "error": f"{settings.llm_provider} not available"}

    answer = raw_answer.strip()

    # Smart refusal detection
    is_refusal = False
    cleaned_upper = answer.strip().upper()
    if cleaned_upper in ("NOT_FOUND", "NOT FOUND", "N/A", "NONE"):
        is_refusal = True
    elif len(answer) < 80:
        lower_ans = answer.lower()
        refusal_phrases = [
            "not found", "not mentioned", "not specified", "not provided",
            "no information", "does not contain", "no such clause", "not addressed"
        ]
        if any(phrase in lower_ans for phrase in refusal_phrases):
            is_refusal = True

    if is_refusal:
        logger.info(f"[QA] Not found for: {query[:60]}")
        return {"answer": None, "score": 0.0}

    answer = _clean_answer(answer)

    if not answer:
        return {"answer": None, "score": 0.0}

    score = _compute_confidence(answer, query, category)

    logger.info(f"[QA] Answer (score={score:.1f}, len={len(answer)}): {answer[:120]}")

    return {
        "answer": answer,
        "score": score,
    }