import os
import logging
from extraction.extractor import extract_text
from processing.cleaner import clean_text
from processing.chunker import chunk_text
from core.vector_db import insert_chunks

logger = logging.getLogger(__name__)

_embedding_model = None

def _get_sbert_device() -> str:
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def get_embedding_model():
    from core.vector_db import _get_search_model
    return _get_search_model()

def run_ocr_pipeline(filepath, job_id=None):
    """
    Full document processing pipeline:
    1. Extract text from PDF/DOCX/Image
    2. Clean text (preserve structure)
    3. Section-aware chunking
    4. Embed and store in Qdrant
    """
    raw = extract_text(filepath)
    clean = clean_text(raw)
    chunks = chunk_text(clean)  # Returns list of {"text": ..., "section_title": ..., "chunk_index": ...}
    
    filename = os.path.basename(filepath)
    model = get_embedding_model()
    
    if model and chunks:
        logger.info(f"Embedding {len(chunks)} chunks for {filename}...")
        
        # Extract just the text for embedding
        chunk_texts = [c["text"] for c in chunks]
        embeddings = model.encode(chunk_texts, show_progress_bar=False)
        
        # Combine embeddings with chunk metadata
        chunks_with_embeddings = []
        for chunk, emb in zip(chunks, embeddings):
            chunks_with_embeddings.append({
                "text": chunk["text"],
                "section_title": chunk["section_title"],
                "chunk_index": chunk["chunk_index"],
                "vector": emb.tolist(),
            })
        
        metadata = {
            "job_id": job_id or "unknown",
            "filename": filename
        }
        
        insert_chunks(chunks_with_embeddings, metadata)
        
        # Log section breakdown
        sections = set(c["section_title"] for c in chunks)
        logger.info(f"Sections found: {sections}")
        
        return {
            "status": "success",
            "message": f"Extracted and stored {len(chunks)} chunks from {len(sections)} sections in Vector DB.",
            "num_chunks": len(chunks),
            "num_sections": len(sections),
        }
    else:
        return {
            "clean_text": clean,
            "chunks": chunks
        }