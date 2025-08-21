# app/api/endpoints/status.py
from fastapi import APIRouter
from app.models.schemas import APIStatusResponse
from app.config.settings import (
    GEMINI_API_KEYS, current_gemini_key_index, gemini_request_count, 
    REQUESTS_PER_KEY, qa_storage, SUPPORTED_FORMATS, MAX_FILE_SIZE
)

router = APIRouter()

@router.get("/api-status", response_model=APIStatusResponse)
async def api_status():
    """Endpoint to check API key rotation status"""
    return APIStatusResponse(
        total_api_keys=len(GEMINI_API_KEYS),
        current_key_index=current_gemini_key_index + 1,
        requests_on_current_key=gemini_request_count,
        requests_per_key_limit=REQUESTS_PER_KEY,
        qa_storage_current_size=len(qa_storage),
        supported_formats=list(SUPPORTED_FORMATS),
        memory_limit_mb=MAX_FILE_SIZE / (1024 * 1024)
    )

@router.get("/faiss-stats")
async def faiss_statistics():
    """Endpoint to check FAISS index statistics"""
    try:
        from app.services.faiss_search import FAISSSearchService
        faiss_service = FAISSSearchService()
        stats = faiss_service.get_stats()
    except Exception as e:
        stats = {"error": str(e), "status": "not_available"}
    
    return {
        "faiss_index_info": stats,
        "search_capabilities": {
            "semantic_search": True,
            "keyword_search": stats.get("tfidf_available", False),
            "hybrid_search": True,
            "domain_specific_boosting": True,
            "numeric_matching": True,
            "phrase_matching": True
        },
        "chunking_strategies": {
            "sliding_window": True,
            "semantic_chunking": True,
            "topic_based_chunking": True,
            "structured_chunking": True,
            "quality_ranking": True
        }
    }

@router.get("/supported-formats")
async def supported_formats():
    """Endpoint to check supported file formats"""
    from app.config.settings import UNSUPPORTED_FORMATS
    
    return {
        "supported_formats": {
            "documents": ["pdf", "docx", "doc", "txt"],
            "spreadsheets": ["xlsx", "xls"],
            "presentations": ["pptx", "ppt"],
            "images": ["jpg", "jpeg", "png", "gif", "bmp", "tiff", "webp"],
            "web": ["html", "htm"],
            "archives": ["zip"]
        },
        "unsupported_formats": list(UNSUPPORTED_FORMATS),
        "max_file_size_mb": MAX_FILE_SIZE / (1024 * 1024),
        "features": {
            "excel_header_mapping": True,
            "nested_zip_extraction": True,
            "image_ocr": True,
            "memory_optimized": True,
            "format_validation": True,
            "faiss_search": True,
            "hybrid_retrieval": True,
            "semantic_chunking": True,
            "topic_based_chunking": True,
            "quality_scoring": True,
            "enhanced_precision": True
        }
    }