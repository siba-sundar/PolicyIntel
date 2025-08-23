import os
from dotenv import load_dotenv
from typing import List

from app.utils.document_cache import create_document_cache

# Load environment variables
load_dotenv()

# API Keys
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
COHERE_API_KEY_2 = os.getenv("COHERE_API_KEY_2")
GEMINI_API_KEY_1 = os.getenv("GEMINI_API_KEY_1")
GEMINI_API_KEY_2 = os.getenv("GEMINI_API_KEY_2")
GEMINI_API_KEY_3 = os.getenv("GEMINI_API_KEY_3")
OCR_SPACE_API_KEY = os.getenv("OCR_SPACE_API_KEY")

# API Key Management
COHERE_API_KEYS = [key for key in [COHERE_API_KEY, COHERE_API_KEY_2] if key is not None]
GEMINI_API_KEYS = [key for key in [GEMINI_API_KEY_1, GEMINI_API_KEY_2, GEMINI_API_KEY_3] if key is not None]
REQUESTS_PER_KEY = 12


# Global Variables for API Key Rotation
current_gemini_key_index = 0
gemini_request_count = 0
current_cohere_key_index = 0

# Global Q&A Storage
qa_storage = []


PG_DATABASE = os.getenv("PG_DATABASE")
PG_USER =  os.getenv("PG_USER")
PG_PASSWORD = os.getenv("PG_PASSWORD")
PG_HOST = "localhost"
PG_PORT = "5432"

# Authentication
TEAM_TOKEN = os.getenv("TEAM_TOKEN")

# File Processing Settings
MAX_FILE_SIZE = 500 * 1024 * 1024  # 500MB
MAX_IMAGE_SIZE = (2048, 2048)
OCR_BATCH_SIZE = 5

# Supported File Formats
SUPPORTED_FORMATS = {
    'pdf', 'jpg', 'jpeg', 'png', 'gif', 'bmp', 'tiff', 'webp',
    'xlsx', 'xls', 'pptx', 'ppt', 'docx', 'doc', 'txt', 'zip'
}

UNSUPPORTED_FORMATS = {
    'exe', 'bat', 'sh', 'dll', 'sys', 'bin', 'iso', 'dmg',
    'mp4', 'avi', 'mov', 'mp3', 'wav', 'flac', 'mkv'
}

# FAISS Settings
FAISS_INDEX_TYPE = "L2"  # L2 (Euclidean) distance
FAISS_NPROBE = 10  # Number of clusters to search
FAISS_K_SEARCH = 20  # Balanced search for quality results

# Enhanced Chunking Settings
CHUNK_SIZE = 450  # Balanced for precision and coverage
CHUNK_OVERLAP = 90  # Increased overlap for better context
MIN_CHUNK_SIZE = 50  # Minimum chunk size in characters
MAX_CHUNK_SIZE = 650  # Increased for better coverage

# Semantic Chunking Parameters
SENTENCE_MIN_LENGTH = 15  # Minimum sentence length
PARAGRAPH_MIN_LENGTH = 50  # Minimum paragraph length
SEMANTIC_THRESHOLD = 0.6  # Lowered for better coverage

# Q&A Enhancement Settings
CONTEXT_WINDOW_SIZE = 15  # Focused context for reasoning
KEYWORD_BOOST_FACTOR = 0.3  # Reduced keyword emphasis
SEMANTIC_SIMILARITY_THRESHOLD = 0.20  # Reasonable threshold for quality

# LLM Settings
GEMINI_MODEL = "gemini-2.5-flash-lite"
LLM_TEMPERATURE = 0.25  # Higher for better reasoning and creativity
LLM_MAX_TOKENS = 300  # Increased for more detailed intelligent responses
LLM_TOP_P = 0.9
LLM_TOP_K = 30

# Embedding Settings
COHERE_MODEL = "embed-english-v3.0"
EMBEDDING_BATCH_SIZE = 96
EMBEDDING_MAX_LENGTH = 2000

# Parallel Processing Settings
MAX_WORKERS = min(4, os.cpu_count())  # Conservative worker count for memory management
PARALLEL_CHUNK_SIZE = 5  # Number of chunks to process in parallel
PARALLEL_OCR_BATCH = 3  # Number of images to process simultaneously
PARALLEL_EMBEDDING_CONCURRENT = 2  # Number of concurrent embedding requests

# Optional Large Document Mode (can be enabled via environment variable)
# Set LARGE_DOC_MODE=true in environment to enable
LARGE_DOC_MODE = os.getenv("LARGE_DOC_MODE", "false").lower() == "true"

# Large document overrides (only used when LARGE_DOC_MODE=true)
if LARGE_DOC_MODE:
    FAISS_K_SEARCH = 30  # More search results
    CONTEXT_WINDOW_SIZE = 20  # More context
    SEMANTIC_SIMILARITY_THRESHOLD = 0.15  # Lower threshold
    CHUNK_SIZE = 600  # Slightly larger chunks
    CHUNK_OVERLAP = 120  # More overlap
    MAX_CHUNK_SIZE = 800  # Larger max chunks
    LLM_MAX_TOKENS = 300  # More tokens for longer answers
    print("🔍 LARGE DOCUMENT MODE ENABLED - Using enhanced settings for large documents")

# Validation
if not COHERE_API_KEYS or not GEMINI_API_KEYS:
    raise ValueError("Missing required API keys. Check COHERE_API_KEY and GEMINI_API_KEY environment variables.")






# Initialize shared document cache
document_cache = create_document_cache(
    cache_dir="document_cache",
    min_cache_threshold=50000,
    max_cache_size_mb=500,
    cache_expiry_hours=1,
    enable_compression=True
)