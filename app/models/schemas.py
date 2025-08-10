# app/models/schemas.py
from pydantic import BaseModel
from typing import List, Dict, Optional, Any

class QueryRequest(BaseModel):
    documents: str
    questions: List[str]

class QueryResponse(BaseModel):
    answers: List[str]

class HealthResponse(BaseModel):
    status: str
    memory_usage: Dict[str, Any]
    version: str
    supported_formats: List[str]
    unsupported_formats: List[str]
    gemini_keys_available: int
    current_gemini_key: int
    gemini_requests_on_current_key: int
    cohere_keys_available: int
    current_cohere_key: int
    qa_storage_size: int
    max_file_size_mb: float
    faiss_index: Dict[str, Any]
    enhanced_features: Dict[str, bool]
    parallel_processing: Dict[str, int]

class MemoryStatusResponse(BaseModel):
    memory_before_cleanup: Dict[str, Any]
    memory_after_cleanup: Dict[str, Any]
    objects_collected: int
    cleanup_performed: bool
    memory_freed_mb: float
    render_free_tier_optimized: bool

class APIStatusResponse(BaseModel):
    total_api_keys: int
    current_key_index: int
    requests_on_current_key: int
    requests_per_key_limit: int
    qa_storage_current_size: int
    supported_formats: List[str]
    memory_limit_mb: float