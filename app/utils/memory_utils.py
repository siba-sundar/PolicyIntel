# app/utils/memory_utils.py
import gc
import psutil
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

def get_memory_usage() -> Dict[str, Any]:
    """Get current memory usage information"""
    try:
        process = psutil.Process()
        memory_info = process.memory_info()
        memory_percent = process.memory_percent()
        system_memory = psutil.virtual_memory()
        
        return {
            "process_memory_mb": round(memory_info.rss / 1024 / 1024, 2),
            "process_memory_percent": round(memory_percent, 2),
            "system_memory_total_gb": round(system_memory.total / 1024 / 1024 / 1024, 2),
            "system_memory_available_gb": round(system_memory.available / 1024 / 1024 / 1024, 2),
            "system_memory_used_percent": round(system_memory.percent, 2)
        }
    except Exception as e:
        logger.warning(f"Failed to get memory usage: {e}")
        return {}

def log_memory_usage(stage: str = "") -> Dict[str, Any]:
    """Log current memory usage with optional stage description"""
    try:
        memory_info = get_memory_usage()
        stage_text = f" ({stage})" if stage else ""
        logger.info(f"📊 Memory Usage{stage_text}: {memory_info['process_memory_mb']}MB ({memory_info['process_memory_percent']}% of system)")
        return memory_info
    except Exception as e:
        logger.warning(f"Failed to get memory usage: {e}")
        return {}

def clear_memory(stage: str = "") -> int:
    """Force garbage collection and log memory usage"""
    # Log memory before cleanup
    memory_before = get_memory_usage()
    
    # Force garbage collection
    collected = gc.collect()
    
    # Log memory after cleanup
    memory_after = get_memory_usage()
    
    # Calculate savings
    if memory_before and memory_after:
        memory_saved = memory_before['process_memory_mb'] - memory_after['process_memory_mb']
        stage_text = f" ({stage})" if stage else ""
        if memory_saved > 0:
            logger.info(f"🧹 Memory Cleanup{stage_text}: Freed {memory_saved:.2f}MB, collected {collected} objects")
        else:
            logger.info(f"🧹 Memory Cleanup{stage_text}: {memory_after['process_memory_mb']}MB used, collected {collected} objects")
    
    return collected

def cleanup_variables(*variables) -> int:
    """Cleanup specific variables and force garbage collection"""
    for var in variables:
        try:
            del var
        except:
            pass
    return clear_memory("variables cleanup")