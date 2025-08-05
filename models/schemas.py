from pydantic import BaseModel
from typing import List

class QueryRequest(BaseModel):
    documents: str
    questions: List[str]

class QueryResponse(BaseModel):
    answers: List[str]

class HealthResponse(BaseModel):
    status: str
    memory_usage: str
    version: str
    api_keys_available: int
    current_key: int
    requests_on_current_key: int
    qa_storage_size: int

class APIStatusResponse(BaseModel):
    total_api_keys: int
    current_key_index: int
    requests_on_current_key: int
    requests_per_key_limit: int
    qa_storage_current_size: int