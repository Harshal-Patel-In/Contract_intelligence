import logging
import re
import requests

logger = logging.getLogger(__name__)

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "qwen2.5:7b"

SYSTEM_PROMPT = """You are a legal contract analysis expert. Extract specific information from contract text.

STRICT RULES:
1. Provide ONLY the extracted answer — no introductory phrases, no "According to...".
2. If the exact information is genuinely absent from the text, respond with exactly one word: NOT_FOUND
3. NEVER combine an answer with NOT_FOUND. Either give the answer OR say NOT_FOUND.
4. For party names: list full legal entity names separated by " and ".
5. For dates: use the exact format from the document.
6. For clauses: summarize the key terms in 2-3 concise sentences.
7. Do NOT fabricate information. Only extract what is explicitly stated.
8. Do NOT truncate your answer mid-sentence. Complete your thought.
9. After the answer, include a confidence score on a separate line in the exact format: Confidence: <integer 1-10>.
"""


def _query_ollama(prompt: str, timeout: int = 90) -> str:
    """Send a prompt to Ollama and return the response text."""
    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": MODEL_NAME,
                "prompt": prompt,
                "system": SYSTEM_PROMPT,
                "stream": False,
                "options": {
                    "temperature": 0.0,
                    "top_p": 0.9,
                    "num_predict": 512,
                    "num_ctx": 8192,
                    "seed": 42,
                }
            },
            timeout=timeout,
        )
        response.raise_for_status()
        result = response.json()
        return result.get("response", "").strip()
    except requests.exceptions.ConnectionError:
        logger.error("[QA] Cannot connect to Ollama. Is it running? (ollama serve)")
        return None
    except requests.exceptions.Timeout:
        logger.error(f"[QA] Ollama request timed out after {timeout}s")
        return None
    except Exception as e:
        logger.error(f"[QA] Ollama error: {e}")
        return None


def _clean_answer(answer: str) -> str:
    """Clean up LLM response artifacts."""
    if not answer:
        return answer
    
    for prefix in ["Answer:", "The answer is:", "Based on the text,",
                    "According to the contract,", "Based on the contract text,"]:
        if answer.lower().startswith(prefix.lower()):
            answer = answer[len(prefix):].strip()
    
    answer = re.sub(r'\s*[Nn][Oo][Tt]_?[Ff][Oo][Uu][Nn][Dd]\b[^.\n]*[.\n]?', '', answer).strip()
    answer = re.sub(r'\s*NOT_FOUND\s*', '', answer, flags=re.IGNORECASE).strip()
    
    lines = answer.split('\n')
    cleaned_lines = []
    for line in lines:
        stripped = line.strip()
        # Skip empty lines and pure not-found lines
        if not stripped:
            continue
        if re.match(r'^\s*(not_?found|n/a|none|no\s+information)\s*\.?\s*$', stripped, re.IGNORECASE):
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


def answer_question(query: str, chunks: list) -> dict:
    """
    Uses Ollama (Qwen 2.5 7B) to extract answers from contract chunks.
    
    Generates a precise answer from combined context — handles multi-paragraph
    clauses, party names, and date extraction.
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
    if len(context) > 6000:
        context = context[:6000]
    
    prompt = f"""Extract the answer to this question from the contract text below.
If the information is not present, respond with exactly: NOT_FOUND

QUESTION: {query}

CONTRACT TEXT:
{context}

RESPONSE FORMAT:
Answer: <extracted answer or NOT_FOUND>
Confidence: <integer 1-10>
"""
    
    logger.warning(f"[QA] Querying Ollama ({MODEL_NAME}) for: {query[:60]}")
    
    raw_answer = _query_ollama(prompt)
    
    if raw_answer is None:
        return {"answer": None, "score": 0.0, "error": "Ollama not available"}
    
    answer = raw_answer.strip()
    confidence = None
    confidence_match = re.search(r'(?mi)^confidence\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)(?:\s*/\s*10)?\s*$', answer)
    if confidence_match:
        confidence = float(confidence_match.group(1))
        answer = re.sub(r'(?mi)^confidence\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)(?:\s*/\s*10)?\s*$', '', answer).strip()

    answer = re.sub(r'(?mi)^answer\s*[:=]\s*', '', answer).strip()
    
    not_found_signals = ["NOT_FOUND", "not found in the", "not provided in the", 
                          "no information available", "cannot determine from",
                          "does not contain", "no such clause", "no mention of",
                          "the contract does not specify", "the text does not"]
    
    is_refusal = (
        len(answer) < 100 and 
        any(signal.lower() in answer.lower() for signal in not_found_signals)
    ) or answer.strip().upper() == "NOT_FOUND"
    
    if is_refusal:
        logger.warning(f"[QA] Not found for: {query[:60]}")
        return {"answer": None, "score": 0.0}
    
    answer = _clean_answer(answer)
    
    if not answer:
        return {"answer": None, "score": 0.0}
    
    if confidence is None:
        # Fallback heuristic if the model did not return a confidence line.
        if len(answer) < 20:
            score = 4.0
        elif len(answer) < 120:
            score = 6.5
        else:
            score = 8.0
    else:
        score = max(1.0, min(10.0, confidence))
    
    logger.warning(f"[QA] Answer (score={score:.1f}, len={len(answer)}): {answer[:100]}")
    
    return {
        "answer": answer,
        "score": score,
    }
