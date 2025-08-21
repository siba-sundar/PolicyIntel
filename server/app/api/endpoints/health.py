# app/api/endpoints/health.py
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

router = APIRouter()

@router.get("/", summary="Root endpoint")
@router.get("/health", summary="Health check endpoint", response_model=dict)
async def health_check():
    """Health check endpoint for deployment and monitoring"""
    try:
        from app.services.faiss_search import FAISSSearchService
        faiss_service = FAISSSearchService()
        faiss_stats = faiss_service.get_stats()
    except:
        faiss_stats = {"status": "not_initialized"}
    
    memory_info = get_memory_usage()
    
    return {
        "status": "healthy", 
        "memory_usage": memory_info, 
        "version": "enhanced_memory_optimized_v2.1",
        "supported_formats": list(SUPPORTED_FORMATS),
        "unsupported_formats": list(UNSUPPORTED_FORMATS),
        "gemini_keys_available": len(GEMINI_API_KEYS),
        "current_gemini_key": current_gemini_key_index + 1,
        "gemini_requests_on_current_key": gemini_request_count,
        "cohere_keys_available": len(COHERE_API_KEYS),
        "current_cohere_key": current_cohere_key_index + 1,
        "qa_storage_size": len(qa_storage),
        "max_file_size_mb": MAX_FILE_SIZE / (1024 * 1024),
        "faiss_index": faiss_stats,
        "enhanced_features": {
            "semantic_chunking": True,
            "topic_based_chunking": True,
            "hybrid_search": True,
            "faiss_indexing": True,
            "quality_scoring": True,
            "parallel_chunking": True,
            "parallel_embedding": True,
            "parallel_ocr": True,
            "intelligent_reasoning": True,
            "context_deduction": True,
            "fallback_intelligence": True
        },
        "parallel_processing": {
            "max_workers": MAX_WORKERS,
            "parallel_chunk_size": PARALLEL_CHUNK_SIZE,
            "parallel_ocr_batch": PARALLEL_OCR_BATCH,
            "parallel_embedding_concurrent": PARALLEL_EMBEDDING_CONCURRENT
        }
    }

@router.get("/memory-status", response_model=MemoryStatusResponse)
async def memory_status():
    """Endpoint to check current memory usage and perform cleanup"""
    # Get memory before cleanup
    memory_before = get_memory_usage()
    
    # Force cleanup
    collected = clear_memory("Memory status check")
    
    # Get memory after cleanup
    memory_after = get_memory_usage()
    
    memory_freed = 0
    if memory_before and memory_after:
        memory_freed = round(memory_before.get('process_memory_mb', 0) - memory_after.get('process_memory_mb', 0), 2)
    
    return MemoryStatusResponse(
        memory_before_cleanup=memory_before,
        memory_after_cleanup=memory_after,
        objects_collected=collected,
        cleanup_performed=True,
        memory_freed_mb=memory_freed,
        render_free_tier_optimized=True
    )