# app/utils/auth_utils.py
import logging
from app.config.settings import (
    GEMINI_API_KEYS, COHERE_API_KEYS, 
    current_gemini_key_index, gemini_request_count,
    current_cohere_key_index, REQUESTS_PER_KEY,
    TEAM_TOKEN
)

logger = logging.getLogger(__name__)

def get_current_gemini_key() -> str:
    """Get current Gemini API key and handle rotation"""
    global current_gemini_key_index, gemini_request_count
    
    current_key = GEMINI_API_KEYS[current_gemini_key_index]
    gemini_request_count += 1
    
    if gemini_request_count >= REQUESTS_PER_KEY:
        gemini_request_count = 0
        current_gemini_key_index = (current_gemini_key_index + 1) % len(GEMINI_API_KEYS)
        logger.info(f"🔄 SWITCHED to Gemini API key #{current_gemini_key_index + 1}")
    
    return current_key

def get_current_cohere_key() -> str:
    """Get current Cohere API key for this request"""
    return COHERE_API_KEYS[current_cohere_key_index]

def rotate_cohere_key():
    """Rotate to the next Cohere API key after a complete request"""
    global current_cohere_key_index
    current_cohere_key_index = (current_cohere_key_index + 1) % len(COHERE_API_KEYS)
    logger.info(f"🔄 SWITCHED to Cohere API key #{current_cohere_key_index + 1}")

def verify_token(authorization: str) -> bool:
    """Verify the authorization token"""
    if not authorization or not authorization.startswith("Bearer "):
        return False
    
    token = authorization.split(" ")[1]
    return token == TEAM_TOKEN