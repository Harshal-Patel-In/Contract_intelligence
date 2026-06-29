import os
import logging
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance

logger = logging.getLogger(__name__)

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_LOCAL_PATH = os.getenv("QDRANT_LOCAL_PATH", "")
COLLECTION_NAME = "contracts"
VECTOR_SIZE = 384  # all-MiniLM-L6-v2

if QDRANT_URL:
    logger.info(f"Connecting to Qdrant at {QDRANT_URL}")
    client = QdrantClient(url=QDRANT_URL)
elif QDRANT_LOCAL_PATH:
    logger.info(f"Using local Qdrant storage at {QDRANT_LOCAL_PATH}")
    os.makedirs(QDRANT_LOCAL_PATH, exist_ok=True)
    client = QdrantClient(path=QDRANT_LOCAL_PATH)
else:
    logger.info("Using in-memory Qdrant (data lost on restart)")
    client = QdrantClient(":memory:")

def _get_sbert_device() -> str:
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def init_db():
    try:
        if not client.collection_exists(collection_name=COLLECTION_NAME):
            client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
            )
            logger.info(f"Created Qdrant collection: {COLLECTION_NAME}")
        else:
            logger.info(f"Qdrant collection {COLLECTION_NAME} already exists.")
    except Exception as e:
        logger.error(f"Failed to initialize Qdrant collection: {e}")
        raise

try:
    init_db()
except Exception as e:
    logger.critical(f"Could not initialize vector database. Exiting: {e}")
    raise SystemExit(1)


def insert_chunks(chunks_with_embeddings: list[dict], metadata: dict):
    """
    Insert chunks with embeddings and metadata into Qdrant.
    
    chunks_with_embeddings: List of dicts like:
        {"text": "...", "vector": [...], "section_title": "...", "chunk_index": 0}
    metadata: Dict with global metadata e.g. {"job_id": "...", "filename": "..."}
    """
    from qdrant_client.models import PointStruct
    import uuid

    points = []
    for chunk in chunks_with_embeddings:
        payload = metadata.copy()
        payload["text"] = chunk["text"]
        payload["section_title"] = chunk.get("section_title", "")
        payload["chunk_index"] = chunk.get("chunk_index", 0)
        
        point = PointStruct(
            id=str(uuid.uuid4()),
            vector=chunk["vector"],
            payload=payload
        )
        points.append(point)
    
    if points:
        batch_size = 100
        for i in range(0, len(points), batch_size):
            batch = points[i:i + batch_size]
            client.upsert(collection_name=COLLECTION_NAME, points=batch)
        logger.info(f"Inserted {len(points)} chunks into Qdrant for job {metadata.get('job_id')}")
        
    return True


def _log_error_to_redis(err_msg: str):
    try:
        from core.redis_client import redis_client
        if redis_client:
            redis_client.set("hf_api_error", err_msg)
    except Exception:
        pass


class HFInferenceEmbeddingModel:
    def __init__(self):
        self.local_model = None
        self.use_api = True
        self.hf_token = os.getenv("HF_TOKEN")
        self.fallback_to_local = not os.getenv("RENDER")

    def encode(self, texts, show_progress_bar=False):
        import numpy as np
        import time
        if isinstance(texts, str):
            single = True
            texts = [texts]
        else:
            single = False

        if self.use_api:
            try:
                import requests
                headers = {}
                if self.hf_token and self.hf_token.startswith("hf_"):
                    headers["Authorization"] = f"Bearer {self.hf_token}"
                
                logger.info(f"Generating embedding via HF Inference API for {len(texts)} texts...")
                
                max_hf_retries = 5
                response = None
                for hf_attempt in range(max_hf_retries):
                    response = requests.post(
                        "https://api-inference.huggingface.co/models/sentence-transformers/all-MiniLM-L6-v2",
                        json={"inputs": texts},
                        headers=headers,
                        timeout=15
                    )
                    if response.status_code == 200:
                        embeddings = response.json()
                        if isinstance(embeddings, list) and len(embeddings) > 0:
                            if isinstance(embeddings[0], float):
                                embeddings = [embeddings]
                            logger.info("Successfully fetched embedding from HF Inference API.")
                            if single:
                                return np.array(embeddings[0])
                            return np.array(embeddings)
                    elif response.status_code == 503:
                        try:
                            err_data = response.json()
                            est_time = err_data.get("estimated_time", 5.0)
                        except Exception:
                            est_time = 5.0
                        logger.info(f"HF model is loading. Waiting {est_time}s before retry (attempt {hf_attempt+1}/{max_hf_retries})...")
                        time.sleep(min(est_time, 10.0))
                    else:
                        break
                        
                if response:
                    err_msg = f"HF Inference API returned status {response.status_code}: {response.text}"
                    logger.warning(err_msg)
                    _log_error_to_redis(err_msg)
            except Exception as e:
                err_msg = f"HF Inference API failed: {e}"
                logger.warning(err_msg)
                _log_error_to_redis(err_msg)
            
        if self.local_model is None:
            if not self.fallback_to_local:
                logger.error("HF Inference API failed and local fallback is disabled on Render to prevent OOM. Returning None.")
                return None
            logger.info("Loading local SentenceTransformer model...")
            from sentence_transformers import SentenceTransformer
            device = _get_sbert_device()
            self.local_model = SentenceTransformer("all-MiniLM-L6-v2", device=device)
            
        embeddings = self.local_model.encode(texts, show_progress_bar=show_progress_bar)
        if single:
            return embeddings[0] if hasattr(embeddings, '__len__') else embeddings
        return embeddings

_search_model = None

def _get_search_model():
    global _search_model
    if _search_model is None:
        _search_model = HFInferenceEmbeddingModel()
    return _search_model


def search_chunks(job_id: str, query: str, top_k: int = 10) -> list[dict]:
    """
    Embeds the query and searches Qdrant for the most relevant chunks.
    Returns list of dicts with text, section_title, chunk_index, and score.
    """
    from qdrant_client.models import Filter, FieldCondition, MatchValue
    
    model = _get_search_model()
    if model is None:
        logger.error("Embedding model not available for search")
        return []
        
    try:
        query_vector = model.encode(query)
        if query_vector is None:
            logger.warning("Embedding failed, falling back to keyword search")
            return keyword_search_chunks(job_id=job_id, keywords=query.split(), limit=top_k)
        query_vector = query_vector.tolist()
    except Exception as e:
        logger.error(f"Embedding encoding failed: {e}, falling back to keyword search")
        return keyword_search_chunks(job_id=job_id, keywords=query.split(), limit=top_k)
    
    query_filter = Filter(
        must=[
            FieldCondition(
                key="job_id",
                match=MatchValue(value=job_id)
            )
        ]
    )
    
    try:
        results = client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vector,
            query_filter=query_filter,
            limit=top_k,
            with_payload=True,
        )
        
        chunks = []
        for hit in results.points:
            chunks.append({
                "text": hit.payload.get("text", ""),
                "section_title": hit.payload.get("section_title", ""),
                "chunk_index": hit.payload.get("chunk_index", 0),
                "score": hit.score if hasattr(hit, 'score') else 0,
            })
        
        logger.debug(f"[SEARCH] job_id={job_id} | query='{query[:50]}...' | found {len(chunks)} chunks")
        if chunks:
            logger.debug(f"[SEARCH] Top chunk (section='{chunks[0]['section_title']}'): {chunks[0]['text'][:100]}...")
        else:
            logger.info(f"[SEARCH] NO CHUNKS FOUND for job_id={job_id}")
        
        return chunks
    except Exception as e:
        logger.error(f"Error searching Qdrant: {e}")
        return []


def get_chunks_by_order(job_id: str, limit: int = 10) -> list[dict]:
    """
    Retrieve the first N chunks of a document sorted by chunk_index.
    
    IMPORTANT: Qdrant scroll returns points in arbitrary UUID order,
    so we must fetch ALL chunks and sort by chunk_index to get the 
    actual document-order preamble.
    """
    all_chunks = get_all_chunks(job_id)
    return all_chunks[:limit]


def get_all_chunks(job_id: str) -> list[dict]:
    """
    Retrieve ALL chunks for a document, sorted by chunk_index (document order).
    Used for comprehensive text analysis and preamble extraction.
    """
    from qdrant_client.models import Filter, FieldCondition, MatchValue
    
    try:
        query_filter = Filter(
            must=[FieldCondition(key="job_id", match=MatchValue(value=job_id))]
        )
        
        all_points = []
        offset = None
        
        while True:
            results, next_offset = client.scroll(
                collection_name=COLLECTION_NAME,
                scroll_filter=query_filter,
                limit=100,
                offset=offset,
            )
            all_points.extend(results)
            
            if next_offset is None:
                break
            offset = next_offset
        
        chunks = []
        for point in all_points:
            chunks.append({
                "text": point.payload.get("text", ""),
                "section_title": point.payload.get("section_title", ""),
                "chunk_index": point.payload.get("chunk_index", 0),
            })
        
        # Sort by chunk_index to get true document order
        chunks.sort(key=lambda x: x["chunk_index"])
        
        logger.info(f"Retrieved {len(chunks)} total chunks for job {job_id}")
        return chunks
    except Exception as e:
        logger.warning(f"Could not retrieve ordered chunks: {e}")
        return []


def keyword_search_chunks(job_id: str, keywords: list[str], limit: int = 10) -> list[dict]:
    """
    Retrieve chunks for a given job_id that match any of the provided keywords.
    Uses Qdrant scroll + in-memory keyword matching (since Qdrant's full-text filter
    requires a payload index, this approach is simpler and more portable).
    
    Returns up to `limit` chunks sorted by keyword match density (most matches first).
    """
    from qdrant_client.models import Filter, FieldCondition, MatchValue
    
    try:
        query_filter = Filter(
            must=[FieldCondition(key="job_id", match=MatchValue(value=job_id))]
        )
        
        all_points = []
        offset = None
        
        while True:
            results, next_offset = client.scroll(
                collection_name=COLLECTION_NAME,
                scroll_filter=query_filter,
                limit=100,
                offset=offset,
            )
            all_points.extend(results)
            
            if next_offset is None:
                break
            offset = next_offset
        
        scored_chunks = []
        for point in all_points:
            text = (point.payload.get("text", "") or "").lower()
            score = 0
            for kw in keywords:
                kw_lower = kw.lower()
                count = text.count(kw_lower)
                score += count
            
            if score > 0:
                scored_chunks.append({
                    "text": point.payload.get("text", ""),
                    "section_title": point.payload.get("section_title", ""),
                    "chunk_index": point.payload.get("chunk_index", 0),
                    "score": score,
                })
        
        # Sort by keyword match count descending, then by chunk_index ascending
        scored_chunks.sort(key=lambda x: (-x["score"], x["chunk_index"]))
        
        # Return only top `limit` results without the internal score
        result = scored_chunks[:limit]
        for chunk in result:
            chunk.pop("score", None)
        
        logger.info(f"Keyword search: job_id={job_id} | keywords={keywords} | found {len(result)} matching chunks")
        return result
    except Exception as e:
        logger.warning(f"Could not perform keyword search: {e}")
        return []