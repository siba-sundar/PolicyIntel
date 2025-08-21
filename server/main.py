# main.py
import logging
from fastapi import FastAPI
from app.api.endpoints import health, status, query
from app.config.settings import SUPPORTED_FORMATS, GEMINI_API_KEYS, COHERE_API_KEYS
from app.utils.memory_utils import get_memory_usage

# Setup Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add HTML format support
SUPPORTED_FORMATS.add('html')
SUPPORTED_FORMATS.add('htm')

# Initialize FastAPI app
app = FastAPI(
    title="PolicyIntel API", 
    description="Document processing and Q&A API with intelligent reasoning", 
    version="1.0.0"
)

# Include routers
app.include_router(health.router, tags=["Health"])
app.include_router(status.router, tags=["Status"])
app.include_router(query.router, tags=["Query"])

@app.get("/")
async def root():
    """Root endpoint with service information"""
    memory_info = get_memory_usage()
    return {
        "message": "Intelligent Multi-Format Document Processor with Parallel Processing & Reasoning", 
        "status": "intelligent_reasoning_v3.1",
        "supported_formats": list(SUPPORTED_FORMATS),
        "api_keys": {
            "gemini_keys_count": len(GEMINI_API_KEYS),
            "cohere_keys_count": len(COHERE_API_KEYS),
        },
        "current_memory_usage_mb": memory_info.get('process_memory_mb', 0),
        "memory_optimization": "render_free_tier_ready",
        "endpoints": {
            "main": "/hackrx/run",
            "health": "/health", 
            "memory_status": "/memory-status",
            "api_status": "/api-status",
            "faiss_stats": "/faiss-stats",
            "supported_formats": "/supported-formats"
        },
        "features": {
            "multi_format_processing": True,
            "intelligent_reasoning": True,
            "memory_optimized": True,
            "parallel_processing": True,
            "semantic_search": True,
            "hybrid_retrieval": True,
            "api_key_rotation": True,
            "nested_zip_support": True,
            "ocr_processing": True,
            "html_processing": True
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")