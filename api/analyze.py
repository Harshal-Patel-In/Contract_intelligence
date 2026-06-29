import os
import re
import logging
import asyncio
from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from core.config import settings
from core.vector_db import search_chunks, get_chunks_by_order, get_all_chunks, keyword_search_chunks
from models.qa_pipeline import answer_question, _query_llm_stream, _clean_answer, _compute_confidence, CATEGORY_HINTS

logger = logging.getLogger(__name__)

router = APIRouter()

CUAD_QUESTIONS = {
    # --- Core Identifiers ---
    "Document Name": "What is the name or title of this contract or agreement?",
    "Parties": "Who are the parties to this agreement? List the full legal names of all companies or individuals.",
    "Effective Date": "What is the effective date or execution date of this contract?",
    "Expiration Date": "What is the expiration date or end date of this contract?",

    # --- Key Terms ---
    "Governing Law": "What is the governing law or jurisdiction for this agreement?",
    "Assignment": "Does the contract restrict the right to assign or transfer the agreement? What are the restrictions?",
    "Renewal Term": "Is there an automatic renewal clause? What are the renewal terms and conditions?",

    # --- Financial Terms ---
    "Payment Terms": "What are the payment terms, fees, or compensation specified in this agreement?",
    "Limitation of Liability": "What is the limitation of liability? What is the maximum liability cap?",
    "Indemnification": "What are the indemnification obligations of each party?",

    # --- Restrictive Covenants ---
    "Non-Compete": "Is there a non-compete clause? What are the specific restrictions on competition?",
    "Confidentiality": "What are the confidentiality or non-disclosure obligations?",
    "Non-Solicitation": "Is there a non-solicitation clause? What does it restrict?",

    # --- Termination ---
    "Termination for Convenience": "Under what circumstances can this agreement be terminated for convenience (without cause)?",
    "Termination for Cause": "Under what circumstances can this agreement be terminated for cause or breach?",

    # --- IP ---
    "Intellectual Property Ownership": "Who owns the intellectual property created under this agreement?",
}

HIGH_CONFIDENCE = settings.confidence_high
LOW_CONFIDENCE = settings.confidence_low

CRITICAL_CATEGORIES = {
    "Limitation of Liability": ("HIGH", "Missing Limitation of Liability clause", 5),
    "Indemnification": ("HIGH", "Missing Indemnification clause", 4),
    "Governing Law": ("MEDIUM", "Missing Governing Law clause", 3),
    "Termination for Convenience": ("MEDIUM", "No Termination for Convenience clause found", 2),
    "Confidentiality": ("MEDIUM", "Missing Confidentiality clause", 2),
}

PREAMBLE_CATEGORIES = {"Parties", "Effective Date", "Document Name", "Expiration Date"}

SECTION_KEYWORDS = {
    "Termination for Convenience": ["termination", "terminate", "convenience", "without cause"],
    "Termination for Cause": ["termination", "terminate", "cause", "breach", "default", "cure"],
    "Governing Law": ["governing law", "jurisdiction", "applicable law", "venue", "forum"],
    "Limitation of Liability": ["liability", "limitation", "damages", "indemnif", "cap", "aggregate"],
    "Indemnification": ["indemnif", "hold harmless", "defend", "losses", "third party claim"],
    "Confidentiality": ["confidential", "non-disclosure", "nda", "proprietary", "trade secret"],
    "Non-Compete": ["non-compete", "noncompete", "compete", "competition", "restrictive", "covenant not to compete"],
    "Non-Solicitation": ["non-solicitation", "nonsolicitation", "solicit", "recruit", "hire"],
    "Assignment": ["assign", "transfer", "delegate", "change of control", "successor"],
    "Renewal Term": ["renewal", "renew", "auto-renew", "extend", "evergreen", "successive"],
    "Payment Terms": ["payment", "fee", "compensation", "commission", "price", "royalt", "milestone"],
    "Intellectual Property Ownership": ["intellectual property", "ip", "patent", "copyright", "trademark", "ownership", "work product", "work for hire"],
    "Expiration Date": ["term", "expir", "end date", "initial term", "standstill", "period"],
}

RISK_PATTERNS = {
    "Termination for Convenience": [
        ("immediately", "HIGH", "Allows immediate termination without notice period", 5),
        ("without notice", "HIGH", "No notice period required for termination", 5),
        ("sole discretion", "MEDIUM", "Termination at sole discretion of one party", 3),
        ("for any reason", "MEDIUM", "Broad termination right for any reason", 3),
        ("no cure period", "HIGH", "No opportunity to cure before termination", 4),
    ],
    "Termination for Cause": [
        ("immediately", "HIGH", "Allows immediate termination without notice period", 5),
        ("without notice", "HIGH", "No notice period required for termination", 5),
        ("sole discretion", "MEDIUM", "Termination at sole discretion of one party", 3),
    ],
    "Limitation of Liability": [
        ("unlimited", "HIGH", "No cap on liability exposure", 5),
        ("no limit", "HIGH", "No cap on liability exposure", 5),
        ("consequential damages", "MEDIUM", "Consequential damages clause present — review scope", 2),
        ("exclusive remedy", "HIGH", "Limited to a single exclusive remedy", 4),
        ("in no event", "LOW", "Liability limitation clause present", 0),
    ],
    "Indemnification": [
        ("sole expense", "MEDIUM", "Potentially one-sided indemnification cost", 3),
        ("unlimited indemnification", "HIGH", "No cap on indemnification exposure", 5),
        ("broad indemnity", "MEDIUM", "Broadly scoped indemnification obligation", 3),
        ("exclusive", "MEDIUM", "Exclusive indemnification may limit remedies", 2),
    ],
    "Non-Compete": [
        ("worldwide", "HIGH", "Global geographic restriction on competition", 4),
        ("perpetual", "HIGH", "No time limit on non-compete restriction", 5),
        ("indefinite", "HIGH", "Unreasonable indefinite duration", 5),
        ("any business", "HIGH", "Overly broad scope of restricted activities", 4),
    ],
    "Confidentiality": [
        ("perpetual", "MEDIUM", "Indefinite confidentiality obligation", 3),
        ("survive termination indefinitely", "MEDIUM", "Obligation survives without time limit", 3),
        ("no exceptions", "MEDIUM", "No carve-outs for standard exceptions", 2),
    ],
    "Assignment": [
        ("without consent", "MEDIUM", "Assignment allowed without other party's approval", 3),
        ("freely assignable", "MEDIUM", "No restrictions on assignment", 3),
    ],
    "Renewal Term": [
        ("auto-renew", "MEDIUM", "Automatic renewal — watch for opt-out deadlines", 2),
        ("evergreen", "MEDIUM", "Evergreen clause with automatic renewal", 2),
        ("without notice", "HIGH", "Auto-renews without notification", 4),
    ],
    "Payment Terms": [
        ("net 90", "MEDIUM", "Extended payment terms (90+ days)", 2),
        ("net 120", "HIGH", "Very long payment terms (120+ days)", 3),
        ("sole satisfaction", "HIGH", "Payment contingent on sole satisfaction", 4),
        ("non-refundable", "MEDIUM", "Non-refundable fees or deposits", 2),
    ],
    "Non-Solicitation": [
        ("perpetual", "HIGH", "Indefinite non-solicitation restriction", 4),
        ("worldwide", "HIGH", "Global scope non-solicitation", 4),
    ],
}


def _extract_document_name_from_filename(filename: str) -> str | None:
    """Derive a cleaner document title from a noisy SEC-style filename."""
    if not filename:
        return None

    name_part = filename.rsplit(".", 1)[0]
    name_part = re.sub(r"^[0-9a-fA-F-]{36}_", "", name_part)
    name_part = re.sub(r"^\s*[^A-Za-z0-9]+", "", name_part)

    exhibit_split = re.split(r"-EX-[\d.]+-", name_part, flags=re.IGNORECASE, maxsplit=1)
    if len(exhibit_split) > 1 and exhibit_split[1].strip():
        candidate = exhibit_split[1]
    else:
        candidate = name_part

    candidate = candidate.replace("_", " ")
    candidate = re.sub(r"^\s*[-:]+\s*", "", candidate)
    candidate = re.sub(r"\s+", " ", candidate).strip()

    return candidate or None


def _assess_risk(category: str, answer: str, score: float) -> tuple:
    """Assess risk using expanded pattern matching across all categories."""
    risk_level = "LOW"
    risk_flag = None
    risk_points = 0

    if category in CRITICAL_CATEGORIES and (not answer or score < LOW_CONFIDENCE):
        level, flag, points = CRITICAL_CATEGORIES[category]
        return level, flag, points

    if answer and category in RISK_PATTERNS:
        text = answer.lower()
        for pattern, level, flag, points in RISK_PATTERNS[category]:
            if pattern.lower() in text:
                return level, flag, points

    if answer and score < LOW_CONFIDENCE:
        return "MEDIUM", "Low confidence — needs human review", 1

    return risk_level, risk_flag, risk_points


def _get_document_metadata(job_id: str) -> dict:
    """Extract metadata about the uploaded document from Redis."""
    from core.redis_client import redis_client

    metadata = {"filename": None, "total_files": None}

    try:
        job_data = redis_client.hgetall(f"job:{job_id}")
        if job_data:
            filenames = job_data.get("filenames", "") or job_data.get(b"filenames", b"").decode()
            total = job_data.get("total_files", "") or job_data.get(b"total_files", b"").decode()
            metadata["filename"] = filenames if filenames else None
            metadata["total_files"] = int(total) if total else None
    except Exception as e:
        logger.warning(f"Could not retrieve document metadata: {e}")

    return metadata


def _get_best_chunks(job_id: str, category: str, question: str) -> list:
    """
    Hybrid retrieval: semantic search + keyword search + preamble retrieval.
    Merges and deduplicates results for maximum recall.
    """
    chunks = search_chunks(job_id=job_id, query=question)

    # Keyword search for this category's known terms
    if category in SECTION_KEYWORDS:
        keywords = SECTION_KEYWORDS[category]
        kw_chunks = keyword_search_chunks(job_id=job_id, keywords=keywords, limit=10)
        existing_texts = {c["text"] for c in chunks}
        for c in kw_chunks:
            if c["text"] not in existing_texts:
                chunks.append(c)
                existing_texts.add(c["text"])

    # Preamble retrieval for identity categories
    if category in PREAMBLE_CATEGORIES:
        preamble_limit = settings.preamble_chunks
        early_chunks = get_chunks_by_order(job_id, limit=preamble_limit)
        existing_texts = {c["text"] for c in chunks}
        for c in early_chunks:
            if c["text"] not in existing_texts:
                chunks.insert(0, c)

        alt_queries = {
            "Parties": [
                "This agreement is entered into by and between",
                "The parties to this agreement",
                "WHEREAS",
            ],
            "Effective Date": [
                "This agreement is made and entered into as of",
                "effective as of the date",
                "dated as of",
            ],
            "Document Name": [
                "This agreement titled",
                "agreement contract exhibit",
            ],
            "Expiration Date": [
                "initial term of this agreement",
                "term shall expire",
                "standstill period",
            ],
        }

        existing_texts = {c["text"] for c in chunks}
        for alt_q in alt_queries.get(category, []):
            alt_chunks = search_chunks(job_id=job_id, query=alt_q, top_k=5)
            for c in alt_chunks:
                if c["text"] not in existing_texts:
                    chunks.append(c)
                    existing_texts.add(c["text"])

    # Boost chunks from sections matching category keywords
    if category in SECTION_KEYWORDS:
        keywords = SECTION_KEYWORDS[category]

        def section_relevance(chunk):
            title = chunk.get("section_title", "").lower()
            text_start = chunk.get("text", "")[:200].lower()
            for kw in keywords:
                if kw in title or kw in text_start:
                    return 0
            return 1

        chunks.sort(key=section_relevance)

    return chunks


def _extract_parties(answer, score, chunks, job_id, question):
    """
    Comprehensive multi-step party extraction:
    1. Regex entity scan across ALL document text
    2. Ask LLM to select actual signatories from entity list
    3. Follow-up for missing second party
    """
    needs_improvement = (not answer or " and " not in answer)

    if not needs_improvement:
        return answer, score

    all_chunks = get_all_chunks(job_id)
    full_text = " ".join(c["text"] for c in all_chunks)

    entity_pattern = r'([A-Z][A-Za-z\s&,\.\'-]+(?:Inc\.|Corp\.|LLC|L\.P\.|Ltd\.|L\.L\.C\.|Co\.|N\.A\.|Limited|Corporation|Company))'
    entities = re.findall(entity_pattern, full_text)

    seen = set()
    unique_entities = []
    for e in entities:
        cleaned = e.strip().rstrip(',').strip()
        normalized = cleaned.lower()
        if normalized not in seen and len(cleaned) > 5:
            seen.add(normalized)
            unique_entities.append(cleaned)

    if len(unique_entities) >= 2:
        entity_list = ", ".join(unique_entities[:10])
        party_q = (
            f"From this list of entities found in the contract, "
            f"which ones are the actual PARTIES (signatories) to this agreement? "
            f"List only the party names separated by ' and '.\n\n"
            f"Entities found: {entity_list}\n\n"
            f"Contract context: {full_text[:3000]}"
        )
        # FIX: Pass actual chunks, not empty list
        party_data = answer_question(query=party_q, chunks=all_chunks[:10], category="Parties")
        party_answer = party_data.get("answer")
        if party_answer and len(party_answer) > 5:
            answer = party_answer
            score = 8.0
    elif len(unique_entities) == 1 and not answer:
        answer = unique_entities[0]
        score = 6.0

    # If still only one party, try to find the second
    if answer and " and " not in answer and unique_entities:
        other_entities = [
            e for e in unique_entities
            if e.lower() not in answer.lower() and answer.lower() not in e.lower()
        ]
        if other_entities:
            followup_q = (
                f"This contract involves {answer}. "
                f"The other entity mentioned is '{other_entities[0]}'. "
                f"Is '{other_entities[0]}' the other party to this agreement? "
                f"Answer with just the full legal name, or NOT_FOUND."
            )
            followup_data = answer_question(query=followup_q, chunks=chunks[:5], category="Parties")
            other_party = followup_data.get("answer")
            if other_party and other_party.lower() != answer.lower():
                answer = f"{answer} and {other_party}"
                score = 8.0

    return answer, score


GROUPS = {
    "Group 1 (Identifiers)": [
        "Document Name", "Parties", "Effective Date", "Expiration Date"
    ],
    "Group 2 (Key Terms)": [
        "Governing Law", "Assignment", "Renewal Term"
    ],
    "Group 3 (Financials)": [
        "Payment Terms", "Limitation of Liability", "Indemnification"
    ],
    "Group 4 (Covenants)": [
        "Non-Compete", "Confidentiality", "Non-Solicitation"
    ],
    "Group 5 (Ops & IP)": [
        "Termination for Convenience", "Termination for Cause", 
        "Intellectual Property Ownership"
    ]
}


def _get_best_chunks_focused_optimized(job_id: str, category: str, question: str) -> list:
    """
    Optimized hybrid retrieval for speed and precision.
    Retrieves a higher limit of chunks since we are querying categories individually.
    """
    # Get top 5 semantic chunks
    chunks = search_chunks(job_id=job_id, query=question, top_k=5)
    
    # Get top 3 keyword chunks
    if category in SECTION_KEYWORDS:
        keywords = SECTION_KEYWORDS[category]
        kw_chunks = keyword_search_chunks(job_id=job_id, keywords=keywords, limit=3)
        existing_texts = {c["text"] for c in chunks}
        for c in kw_chunks:
            if c["text"] not in existing_texts:
                chunks.append(c)
                existing_texts.add(c["text"])
                
    # Add preamble chunks for identity categories
    if category in PREAMBLE_CATEGORIES:
        early_chunks = get_chunks_by_order(job_id, limit=8)
        existing_texts = {c["text"] for c in chunks}
        for c in early_chunks:
            if c["text"] not in existing_texts:
                chunks.insert(0, c)
                
    return chunks


def _query_llm_individual(question: str, context: str, category: str, max_retries: int = 3, retry_delay: float = 5.0) -> str:
    """
    Call the configured LLM (Groq or Ollama) with retry logic for a single category.
    Uses the provider-agnostic prompt builder.
    """
    import time
    from models.qa_pipeline import _query_llm

    prompt = _build_streaming_prompt(question, context, category)

    for attempt in range(1, max_retries + 1):
        try:
            result = _query_llm(prompt)
            if result is not None:
                return result
            logger.warning(f"LLM query returned None for {category} (attempt {attempt}/{max_retries})")
        except Exception as e:
            logger.warning(f"LLM query error for {category} (attempt {attempt}/{max_retries}): {e}")
        
        if attempt < max_retries:
            logger.info(f"Retrying in {retry_delay}s...")
            time.sleep(retry_delay)
        else:
            logger.error(f"LLM query failed after {max_retries} attempts for {category}.")
            return "NOT_FOUND"
    
    return "NOT_FOUND"


def _build_streaming_prompt(question: str, context: str, category: str, history: list = None) -> str:
    """Build the prompt for streaming LLM calls, supporting both category extraction and chat."""
    from models.qa_pipeline import CATEGORY_HINTS

    hint = CATEGORY_HINTS.get(category, "")
    hint_block = f"\nEXTRACTION HINTS: {hint}\n" if hint else ""

    extra_guidelines = ""
    if category == "Termination for Convenience":
        extra_guidelines = (
            "\n5. If a party has a right to terminate the contract upon notice or under certain conditions (even if "
            "heavily redacted with [**], like eBix terminating under Section 13.1), you MUST extract this as the "
            "Termination for Convenience clause. Do not return NOT_FOUND if such a termination clause is present."
        )

    if category == "Chat":
        history_text = ""
        if history:
            history_text = "\nPREVIOUS CONVERSATION HISTORY:\n"
            for msg in history:
                role_name = "User" if msg.get("role") == "user" else "AI"
                history_text += f"{role_name}: {msg.get('text', '')}\n"
            history_text += "\n"

        prompt = f"""You are a helpful, professional legal contract analyst assisting a user. 

If the user is just saying hello or greeting you, respond politely and ask how you can help them analyze the contract. Do not mention the contract text for simple greetings.
Otherwise, answer the user's question based ONLY on the provided contract text. 
If the information is not present in the text, clearly state that you could not find the information in the contract. Do NOT make up an answer.
{history_text}
QUESTION: {question}

CONTRACT TEXT:
{context}
"""
    else:
        prompt = f"""You are a professional legal contract analyst. Extract the answer to the question below from the contract text.
If the information is genuinely not present in the text, respond with exactly: NOT_FOUND.
{hint_block}
Think step by step: first identify the relevant clauses or sections, then extract the specific answer.

IMPORTANT GUIDELINES:
1. Provide a direct quote or the exact text from the contract. Do not summarize unless necessary.
2. If a clause is present but contains redacted placeholders (like [**]), you must still extract the clause.
3. If a clause uses different terminology (e.g. 'assign' or 'transfer' for assignment, 'exclusivity' or 'proprietary' or 'ownership of data' for IP ownership, 'terminate... without cause' or 'terminate... upon notice' for termination for convenience), you must still extract it.{extra_guidelines}
4. Do not return NOT_FOUND if there is a clause that addresses the general topic of the question.

QUESTION: {question}

CONTRACT TEXT:
{context}

ANSWER:"""
    return prompt


@router.get("/analyze/stream/{job_id}")
async def analyze_contract_stream(job_id: str):
    """
    Runs the sequential individual extraction pipeline for maximum accuracy and focus,
    streaming the results in real-time via Server-Sent Events (SSE).
    """
    from models.qa_pipeline import _clean_answer, _compute_confidence
    import json

    async def event_generator():
        try:
            doc_metadata = _get_document_metadata(job_id)
            filename = doc_metadata.get("filename", "") or ""

            results = {}
            total_risk_score = 0
            high_risk_count = 0
            medium_risk_count = 0

            # Process each category individually
            for cat, question in CUAD_QUESTIONS.items():
                logger.info(f"[STREAM] Processing category: {cat}")
                
                # Yield START event
                yield f"data: {json.dumps({'status': 'start', 'category': cat, 'question': question})}\n\n"
                await asyncio.sleep(0.01)

                # Gather context chunks for the category
                cat_chunks = await asyncio.to_thread(
                    _get_best_chunks_focused_optimized, job_id, cat, question
                )
                
                # Sort chunks by document order
                cat_chunks.sort(key=lambda x: x.get("chunk_index", 0))
                
                context_parts = []
                for c in cat_chunks:
                    sec = c.get("section_title", "")
                    txt = c.get("text", "")
                    if sec:
                        context_parts.append(f"[Section: {sec}]\n{txt}")
                    else:
                        context_parts.append(txt)
                        
                context = "\n\n---\n\n".join(context_parts)
                
                # Cap context per category to settings.max_context_chars
                if len(context) > settings.max_context_chars:
                    scored_chunks = sorted(cat_chunks, key=lambda x: x.get("score", 0), reverse=True)
                    kept_chunks = []
                    total_chars = 0
                    for c in scored_chunks:
                        chunk_text = c.get("text", "")
                        if total_chars + len(chunk_text) <= settings.max_context_chars:
                            kept_chunks.append(c)
                            total_chars += len(chunk_text)
                        else:
                            break
                    kept_chunks.sort(key=lambda x: x.get("chunk_index", 0))
                    context_parts = []
                    for c in kept_chunks:
                        sec = c.get("section_title", "")
                        txt = c.get("text", "")
                        if sec:
                            context_parts.append(f"[Section: {sec}]\n{txt}")
                        else:
                            context_parts.append(txt)
                    context = "\n\n---\n\n".join(context_parts)

                # Query LLM with streaming (Groq or Ollama)
                raw_res_parts = []
                stream_prompt = _build_streaming_prompt(question, context, cat)
                async for text_chunk in _query_llm_stream(stream_prompt, max_retries=3, retry_delay=5.0):
                    raw_res_parts.append(text_chunk)
                    yield f"data: {json.dumps({'status': 'chunk', 'category': cat, 'text': text_chunk})}\n\n"
                    await asyncio.sleep(0.001)

                raw_res = "".join(raw_res_parts)
                
                # Smart refusal detection
                is_refusal = False
                cleaned_upper = raw_res.strip().upper()
                if cleaned_upper in ("NOT_FOUND", "NOT FOUND", "N/A", "NONE"):
                    is_refusal = True
                elif len(raw_res) < 80:
                    lower_ans = raw_res.lower()
                    refusal_phrases = [
                        "not found", "not mentioned", "not specified", "not provided",
                        "no information", "does not contain", "no such clause", "not addressed"
                    ]
                    if any(phrase in lower_ans for phrase in refusal_phrases):
                        is_refusal = True
                
                if is_refusal:
                    answer = None
                    score = 0.0
                else:
                    answer = _clean_answer(raw_res)
                    if answer:
                        answer = re.sub(r'<a\s[^>]*>', '', answer)
                        answer = re.sub(r'</a>', '', answer)
                        answer = re.sub(r'<[^>]+>', '', answer)
                        answer = re.sub(r'a\s+href=["\'][^"\']*["\']', '', answer)
                        answer = re.sub(r'https?://\S+', '', answer)
                        answer = re.sub(r'^[\s<>]+', '', answer).strip()
                        if len(answer) < 15:
                            answer = None
                    if not answer:
                        score = 0.0
                    else:
                        score = _compute_confidence(answer, question, cat)
                
                # Special category fallback logic
                if cat == "Document Name":
                    if not answer and filename:
                        answer = _extract_document_name_from_filename(filename)
                        if answer:
                            score = 7.0
                    if answer:
                        answer = re.sub(r'^\d+\s+', '', answer)
                        answer = re.sub(r'^EX-[\d.]+\s*', '', answer, flags=re.IGNORECASE)

                if cat == "Parties":
                    answer, score = _extract_parties(answer, score, cat_chunks, job_id, question)

                # Risk Assessment
                risk_level, risk_flag, risk_points = _assess_risk(cat, answer, score)
                total_risk_score += risk_points
                
                if risk_level == "HIGH":
                    high_risk_count += 1
                elif risk_level == "MEDIUM":
                    medium_risk_count += 1
                    
                if not answer:
                    confidence_label = "NOT_FOUND"
                elif score >= settings.confidence_high:
                    confidence_label = "HIGH"
                elif score >= settings.confidence_low:
                    confidence_label = "MEDIUM"
                else:
                    confidence_label = "LOW"
                    
                results[cat] = {
                    "question": question,
                    "extracted_answer": answer,
                    "confidence_score": round(score, 4),
                    "confidence_label": confidence_label,
                    "risk_level": risk_level,
                    "risk_flag": risk_flag,
                }

                # Yield DONE event for this category
                yield f"data: {json.dumps({
                    'status': 'done',
                    'category': cat,
                    'extracted_answer': answer,
                    'confidence_score': round(score, 4),
                    'confidence_label': confidence_label,
                    'risk_level': risk_level,
                    'risk_flag': risk_flag,
                    'risk_points': risk_points
                })}\n\n"
                await asyncio.sleep(0.01)

            # Post-process overall risk summary
            if total_risk_score >= 15:
                overall_risk = "CRITICAL"
            elif total_risk_score >= 8:
                overall_risk = "HIGH"
            elif total_risk_score >= 3:
                overall_risk = "MEDIUM"
            else:
                overall_risk = "LOW"

            final_summary = {
                "overall_risk": overall_risk,
                "total_risk_score": total_risk_score,
                "high_risk_flags": high_risk_count,
                "medium_risk_flags": medium_risk_count,
                "categories_analyzed": len(CUAD_QUESTIONS),
            }

            # Yield FINAL event
            yield f"data: {json.dumps({
                'status': 'final_summary',
                'risk_summary': final_summary
            })}\n\n"
        except Exception as e:
            logger.error(f"[STREAM] Error during contract streaming analysis: {e}", exc_info=True)
            yield f"data: {json.dumps({'status': 'error', 'message': str(e)})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@router.get("/analyze/{job_id}")
async def analyze_contract(job_id: str):
    """
    Runs the sequential individual extraction pipeline for maximum accuracy and focus.
    Returns extracted entities/clauses, risk assessment, and document metadata.
    """
    from models.qa_pipeline import _clean_answer, _compute_confidence

    doc_metadata = _get_document_metadata(job_id)
    filename = doc_metadata.get("filename", "") or ""

    results = {}
    total_risk_score = 0
    high_risk_count = 0
    medium_risk_count = 0

    # Process each category individually
    for cat, question in CUAD_QUESTIONS.items():
        logger.info(f"Processing category: {cat}")
        
        # Gather context chunks for the category
        cat_chunks = await asyncio.to_thread(
            _get_best_chunks_focused_optimized, job_id, cat, question
        )
        
        # Sort chunks by document order
        cat_chunks.sort(key=lambda x: x.get("chunk_index", 0))
        
        context_parts = []
        for c in cat_chunks:
            sec = c.get("section_title", "")
            txt = c.get("text", "")
            if sec:
                context_parts.append(f"[Section: {sec}]\n{txt}")
            else:
                context_parts.append(txt)
                
        context = "\n\n---\n\n".join(context_parts)
        
        # Cap context per category to settings.max_context_chars
        if len(context) > settings.max_context_chars:
            # Sort by relevance to keep the best chunks
            scored_chunks = sorted(cat_chunks, key=lambda x: x.get("score", 0), reverse=True)
            kept_chunks = []
            total_chars = 0
            for c in scored_chunks:
                chunk_text = c.get("text", "")
                if total_chars + len(chunk_text) <= settings.max_context_chars:
                    kept_chunks.append(c)
                    total_chars += len(chunk_text)
                else:
                    break
            kept_chunks.sort(key=lambda x: x.get("chunk_index", 0))
            context_parts = []
            for c in kept_chunks:
                sec = c.get("section_title", "")
                txt = c.get("text", "")
                if sec:
                    context_parts.append(f"[Section: {sec}]\n{txt}")
                else:
                    context_parts.append(txt)
            context = "\n\n---\n\n".join(context_parts)

        # Call LLM for individual category (Groq or Ollama)
        raw_res = await asyncio.to_thread(_query_llm_individual, question, context, cat)
        
        # Smart refusal detection
        is_refusal = False
        cleaned_upper = raw_res.strip().upper()
        if cleaned_upper in ("NOT_FOUND", "NOT FOUND", "N/A", "NONE"):
            is_refusal = True
        elif len(raw_res) < 80:
            lower_ans = raw_res.lower()
            refusal_phrases = [
                "not found", "not mentioned", "not specified", "not provided",
                "no information", "does not contain", "no such clause", "not addressed"
            ]
            if any(phrase in lower_ans for phrase in refusal_phrases):
                is_refusal = True
        
        if is_refusal:
            answer = None
            score = 0.0
        else:
            answer = _clean_answer(raw_res)
            # Strip HTML tags and stray angle brackets (from PDF cross-references)
            if answer:
                answer = re.sub(r'<a\s[^>]*>', '', answer)        # remove <a href=...>
                answer = re.sub(r'</a>', '', answer)               # remove </a>
                answer = re.sub(r'<[^>]+>', '', answer)            # remove any other HTML tags
                answer = re.sub(r'a\s+href=["\'][^"\']*["\']', '', answer)  # remove leftover href attrs
                answer = re.sub(r'https?://\S+', '', answer)       # remove bare URLs
                answer = re.sub(r'^[\s<>]+', '', answer).strip()   # strip leading < or >
                if len(answer) < 15:
                    answer = None
            if not answer:
                score = 0.0
            else:
                score = _compute_confidence(answer, question, cat)
        
        # Special category handling
        # Document Name fallback
        if cat == "Document Name":
            if not answer and filename:
                answer = _extract_document_name_from_filename(filename)
                if answer:
                    score = 7.0
            if answer:
                answer = re.sub(r'^\d+\s+', '', answer)
                answer = re.sub(r'^EX-[\d.]+\s*', '', answer, flags=re.IGNORECASE)

        # Parties multi-step fallback
        if cat == "Parties":
            answer, score = _extract_parties(answer, score, cat_chunks, job_id, question)

        # Risk Assessment
        risk_level, risk_flag, risk_points = _assess_risk(cat, answer, score)
        total_risk_score += risk_points
        
        if risk_level == "HIGH":
            high_risk_count += 1
        elif risk_level == "MEDIUM":
            medium_risk_count += 1
            
        # Determine confidence label
        if not answer:
            confidence_label = "NOT_FOUND"
        elif score >= settings.confidence_high:
            confidence_label = "HIGH"
        elif score >= settings.confidence_low:
            confidence_label = "MEDIUM"
        else:
            confidence_label = "LOW"
            
        results[cat] = {
            "question": question,
            "extracted_answer": answer,
            "confidence_score": round(score, 4),
            "confidence_label": confidence_label,
            "risk_level": risk_level,
            "risk_flag": risk_flag,
        }

    if total_risk_score >= 15:
        overall_risk = "CRITICAL"
    elif total_risk_score >= 8:
        overall_risk = "HIGH"
    elif total_risk_score >= 3:
        overall_risk = "MEDIUM"
    else:
        overall_risk = "LOW"

    return {
        "job_id": job_id,
        "document": doc_metadata,
        "risk_summary": {
            "overall_risk": overall_risk,
            "total_risk_score": total_risk_score,
            "high_risk_flags": high_risk_count,
            "medium_risk_flags": medium_risk_count,
            "categories_analyzed": len(CUAD_QUESTIONS),
        },
        "extraction_results": results,
    }


@router.get("/debug/chunks/{job_id}")
def debug_chunks(job_id: str):
    """Debug endpoint: view all stored text chunks with section metadata."""
    from core.vector_db import client, COLLECTION_NAME
    from qdrant_client.models import Filter, FieldCondition, MatchValue

    query_filter = Filter(
        must=[FieldCondition(key="job_id", match=MatchValue(value=job_id))]
    )

    results = client.scroll(
        collection_name=COLLECTION_NAME,
        scroll_filter=query_filter,
        limit=200,
    )

    chunks = []
    for point in results[0]:
        chunks.append({
            "chunk_index": point.payload.get("chunk_index", 0),
            "section_title": point.payload.get("section_title", ""),
            "text": point.payload.get("text", ""),
        })

    chunks.sort(key=lambda x: x["chunk_index"])

    return {
        "job_id": job_id,
        "total_chunks": len(chunks),
        "sections": list(set(c["section_title"] for c in chunks)),
        "chunks": chunks,
    }


from api.models import ChatRequest

@router.post("/analyze/chat/{job_id}")
async def chat_contract_stream(job_id: str, request: ChatRequest):
    """
    Endpoint for custom 'Chat with the Contract' feature.
    Uses SSE to stream the AI response token-by-token.
    """
    import json
    import api.models
    from fastapi.responses import StreamingResponse
    
    async def event_generator():
        try:
            question = request.message
            logger.info(f"[CHAT STREAM] User asked: {question}")
            
            chunks = search_chunks(job_id=job_id, query=question, top_k=8)
            chunks.sort(key=lambda x: x.get("chunk_index", 0))
            
            context_parts = []
            for c in chunks:
                sec = c.get("section_title", "")
                txt = c.get("text", "")
                if sec:
                    context_parts.append(f"[Section: {sec}]\n{txt}")
                else:
                    context_parts.append(txt)
                    
            context = "\n\n---\n\n".join(context_parts)
            
            if len(context) > settings.max_context_chars:
                context = context[:settings.max_context_chars]

            yield f"data: {json.dumps({'status': 'start'})}\n\n"
            
            stream_prompt = _build_streaming_prompt(question, context, "Chat", history=request.history)
            async for text_chunk in _query_llm_stream(stream_prompt, max_retries=3, retry_delay=5.0):
                yield f"data: {json.dumps({'status': 'chunk', 'text': text_chunk})}\n\n"
                
            yield f"data: {json.dumps({'status': 'done'})}\n\n"
            
        except Exception as e:
            logger.error(f"[CHAT STREAM] Error: {e}", exc_info=True)
            yield f"data: {json.dumps({'status': 'error', 'message': str(e)})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")
