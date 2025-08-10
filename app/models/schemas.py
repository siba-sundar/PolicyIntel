# app/models/schemas.py
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
from datetime import datetime

# Existing schemas (unchanged for backward compatibility)
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

# New security-enhanced schemas
class EncryptedData(BaseModel):
    """Model for encrypted data packages"""
    encrypted_data: str = Field(..., description="Base64 encoded encrypted data")
    salt: str = Field(..., description="Base64 encoded salt")
    nonce: str = Field(..., description="Base64 encoded nonce")
    tag: str = Field(..., description="Base64 encoded authentication tag")
    integrity_hash: str = Field(..., description="HMAC-SHA256 integrity hash")
    context: str = Field(..., description="Encryption context")

class SecureQueryRequest(BaseModel):
    """Enhanced query request with optional encryption"""
    documents: str
    questions: List[str]
    encrypt_response: bool = Field(False, description="Whether to return encrypted response")
    request_id: Optional[str] = Field(None, description="Optional request identifier")

class SecureQueryResponse(BaseModel):
    """Enhanced query response with security metadata"""
    answers: List[str]
    request_id: Optional[str] = Field(None, description="Request identifier")
    integrity_hash: str = Field(..., description="Response integrity hash")
    timestamp: datetime = Field(default_factory=datetime.now)
    processing_time: float = Field(..., description="Processing time in seconds")

class EncryptedQueryResponse(BaseModel):
    """Fully encrypted query response"""
    encrypted_answers: EncryptedData = Field(..., description="Encrypted answers package")
    request_id: Optional[str] = Field(None, description="Request identifier")
    integrity_verified: bool = Field(..., description="Whether integrity was verified")
    timestamp: datetime = Field(default_factory=datetime.now)
    processing_time: float = Field(..., description="Processing time in seconds")

class SecurityStatusResponse(BaseModel):
    """Security system status response"""
    encryption_algorithm: str = Field(..., description="Encryption algorithm in use")
    integrity_algorithm: str = Field(..., description="Integrity verification algorithm")
    secure_cache_status: str = Field(..., description="Secure cache status")
    audit_logging: str = Field(..., description="Audit logging status")
    system_status: str = Field(..., description="Overall security system status")
    last_key_rotation: Optional[datetime] = Field(None, description="Last key rotation time")

class AuditLogEntry(BaseModel):
    """Audit log entry model"""
    request_id: str = Field(..., description="Request identifier")
    timestamp: float = Field(..., description="Unix timestamp")
    action: str = Field(..., description="Action performed")
    user_id: Optional[str] = Field(None, description="User identifier if available")
    integrity_hash: str = Field(..., description="Log entry integrity hash")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

class SecureHealthResponse(HealthResponse):
    """Enhanced health response with security information"""
    security_status: SecurityStatusResponse = Field(..., description="Security system status")
    encrypted_cache_entries: int = Field(0, description="Number of encrypted cache entries")
    integrity_checks_passed: int = Field(0, description="Number of integrity checks passed")
    integrity_checks_failed: int = Field(0, description="Number of integrity checks failed")