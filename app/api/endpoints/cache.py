from fastapi import APIRouter
from app.models.schemas import HealthResponse, MemoryStatusResponse
from app.config.settings import (
    SUPPORTED_FORMATS, UNSUPPORTED_FORMATS, GEMINI_API_KEYS, COHERE_API_KEYS,
    current_gemini_key_index, gemini_request_count, current_cohere_key_index,
    qa_storage, MAX_FILE_SIZE, MAX_WORKERS, PARALLEL_CHUNK_SIZE,
    PARALLEL_OCR_BATCH, PARALLEL_EMBEDDING_CONCURRENT
)
from app.utils.memory_utils import get_memory_usage, clear_memory
from app.services.faiss_search import FAISSSearchService
from app.config.settings import document_cache

router = APIRouter()



@router.get("/cache-status")
async def cache_status():
    """Get cache statistics and status"""
    stats = document_cache.get_cache_stats()
    return {
        "cache_statistics": stats,
        "memory_savings": {
            "description": "Cached documents skip processing entirely",
            "benefits": [
                "No document download/parsing",
                "No chunking computation", 
                "No embedding generation",
                "Instant result retrieval"
            ]
        }
    }

@router.post("/clear-cache")
async def clear_cache(confirm: bool = False):
    """Clear all cached documents"""
    result = document_cache.clear_cache(confirm=confirm)
    return result

@router.post("/optimize-cache") 
async def optimize_cache():
    """Optimize cache by removing expired entries"""
    result = document_cache.optimize_cache()
    return result