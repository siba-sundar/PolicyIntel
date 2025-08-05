import os
from typing import List, Optional
from dotenv import load_dotenv
import logging

# Load environment variables
load_dotenv()

class Settings:
    # API Keys
    COHERE_API_KEY: Optional[str] = os.getenv("COHERE_API_KEY")
    GEMINI_API_KEY_1: Optional[str] = os.getenv("GEMINI_API_KEY_1")
    GEMINI_API_KEY_2: Optional[str] = os.getenv("GEMINI_API_KEY_2")
    GEMINI_API_KEY_3: Optional[str] = os.getenv("GEMINI_API_KEY_3")
    
    # Authentication
    TEAM_TOKEN: str = "833695cad1c0d2600066bf2b08aab7614d0dec93b4b6f0ae3acd37ef7d6fcb1c"
    
    # API Configuration
    REQUESTS_PER_KEY: int = 12
    
    # Text Processing
    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 100
    MAX_PDF_PAGES: int = 50
    SIMILARITY_THRESHOLD: float = 0.25
    
    # Embedding Configuration
    EMBEDDING_BATCH_SIZE: int = 96
    
    # Search Configuration
    TOP_K_CHUNKS: int = 15
    MAX_CONTEXT_CHUNKS: int = 12
    
    def __init__(self):
        self.GEMINI_API_KEYS = [
            key for key in [self.GEMINI_API_KEY_1, self.GEMINI_API_KEY_2, self.GEMINI_API_KEY_3] 
            if key is not None
        ]
        
        if not self.COHERE_API_KEY or not self.GEMINI_API_KEYS:
            logging.error("Missing API keys. Check COHERE_API_KEY and GEMINI_API_KEY environment variables.")
            raise ValueError("Missing required API keys")

settings = Settings()