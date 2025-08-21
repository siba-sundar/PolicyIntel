# app/services/embedding_service.py
import httpx
import asyncio
import numpy as np
import re
import time
import logging
from typing import List
from app.config.settings import (
    COHERE_MODEL, EMBEDDING_BATCH_SIZE, EMBEDDING_MAX_LENGTH,
    PARALLEL_EMBEDDING_CONCURRENT
)
from app.utils.auth_utils import get_current_cohere_key
from app.utils.text_utils import words_to_numbers

logger = logging.getLogger(__name__)

async def get_embeddings_batch(client: httpx.AsyncClient, batch: List[str], input_type: str, batch_num: int) -> List[List[float]]:
    """Process a single batch of embeddings"""
    url = "https://api.cohere.com/v1/embed"
    current_cohere_key = get_current_cohere_key()
    headers = {
        "Authorization": f"Bearer {current_cohere_key}",
        "Content-Type": "application/json",
        "Accept": "application/json"
    }
    
    data = {
        "model": COHERE_MODEL,
        "texts": batch,
        "input_type": input_type,
        "truncate": "END"
    }
    
    try:
        response = await client.post(url, headers=headers, json=data)
        
        if response.status_code == 200:
            response_data = response.json()
            embeddings_data = response_data.get("embeddings", [])
            
            batch_embeddings = []
            for embedding in embeddings_data:
                vec = np.array(embedding, dtype=np.float32)
                norm = np.linalg.norm(vec)
                if norm > 0:
                    vec = vec / norm
                batch_embeddings.append(vec.tolist())
            
            logger.info(f"Completed embedding batch {batch_num}")
            return batch_embeddings
        else:
            logger.error(f"Cohere API error for batch {batch_num}: {response.status_code} - {response.text}")
            response.raise_for_status()
            
    except Exception as e:
        logger.error(f"Embedding batch {batch_num} failed: {str(e)}")
        raise e

async def get_embeddings(texts: List[str], input_type: str = "search_document") -> List[List[float]]:
    """Get embeddings for a list of texts with parallel processing"""
    start_time = time.time()
    logger.info(f"🤖 Getting embeddings for {len(texts)} texts with parallel processing")
    
    clean_texts = []
    for text in texts:
        if text.strip():
            converted_text = words_to_numbers(text)
            clean_text = re.sub(r'\s+', ' ', converted_text.strip())[:EMBEDDING_MAX_LENGTH]
            clean_texts.append(clean_text)
    
    if not clean_texts:
        raise ValueError("No valid texts provided for embedding")
    
    # Create batches
    batches = [clean_texts[i:i+EMBEDDING_BATCH_SIZE] for i in range(0, len(clean_texts), EMBEDDING_BATCH_SIZE)]
    
    # Process batches with limited concurrency to avoid overwhelming the API
    timeout = httpx.Timeout(45.0)
    limits = httpx.Limits(max_keepalive_connections=5, max_connections=10)
    
    all_embeddings = []
    
    async with httpx.AsyncClient(timeout=timeout, limits=limits) as client:
        # Process batches in groups to control concurrency
        for i in range(0, len(batches), PARALLEL_EMBEDDING_CONCURRENT):
            concurrent_batches = batches[i:i+PARALLEL_EMBEDDING_CONCURRENT]
            
            # Create tasks for concurrent batch processing
            tasks = [
                get_embeddings_batch(client, batch, input_type, i + j + 1)
                for j, batch in enumerate(concurrent_batches)
            ]
            
            # Wait for all batches in this group to complete
            batch_results = await asyncio.gather(*tasks)
            
            # Collect results
            for batch_embeddings in batch_results:
                all_embeddings.extend(batch_embeddings)
    
    logger.info(f"Got {len(all_embeddings)} embeddings in {time.time() - start_time:.2f}s using parallel processing")
    return all_embeddings