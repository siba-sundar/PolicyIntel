# app/utils/file_utils.py
from pathlib import Path
from typing import Tuple, Optional
from app.config.settings import SUPPORTED_FORMATS, UNSUPPORTED_FORMATS

def get_file_extension(filename: str) -> str:
    """Extract file extension safely"""
    return Path(filename).suffix.lower().lstrip('.')

def is_supported_format(filename: str) -> Tuple[bool, Optional[str]]:
    """Check if file format is supported"""
    ext = get_file_extension(filename)
    
    if ext in UNSUPPORTED_FORMATS:
        return False, f"File format '{ext}' is not allowed for security reasons"
    elif ext in SUPPORTED_FORMATS:
        return True, None
    else:
        return False, f"File format '{ext}' is not supported. Supported formats: {', '.join(sorted(SUPPORTED_FORMATS))}"