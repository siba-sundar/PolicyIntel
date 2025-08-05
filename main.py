from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel
import httpx
import asyncio
import fitz
from google import genai
import os
from dotenv import load_dotenv
from typing import List, Dict
import numpy as np
import json
import re
import hashlib
import logging
import time

# ---- Setup Logging ----
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---- Load API Keys ----
load_dotenv()
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
GEMINI_API_KEY_1 = os.getenv("GEMINI_API_KEY_1")
GEMINI_API_KEY_2 = os.getenv("GEMINI_API_KEY_2")
GEMINI_API_KEY_3 = os.getenv("GEMINI_API_KEY_3")

# ---- API Key Rotation Setup ----
GEMINI_API_KEYS = [GEMINI_API_KEY_1, GEMINI_API_KEY_2, GEMINI_API_KEY_3]
# Remove None keys if any are missing
GEMINI_API_KEYS = [key for key in GEMINI_API_KEYS if key is not None]

if not COHERE_API_KEY or not GEMINI_API_KEYS:
    logger.error("Missing API keys. Check COHERE_API_KEY and GEMINI_API_KEY environment variables.")
    raise ValueError("Missing required API keys")

# Global variables for API key rotation
current_key_index = 0
request_count = 0
REQUESTS_PER_KEY = 12

# ---- Global array to store Q&A pairs ----
qa_storage = []

def get_current_gemini_key():
    """Get current Gemini API key and handle rotation"""
    global current_key_index, request_count
    
    # Get current key
    current_key = GEMINI_API_KEYS[current_key_index]
    
    # Increment request count
    request_count += 1
    
    # Check if we need to rotate
    if request_count >= REQUESTS_PER_KEY:
        request_count = 0  # Reset counter
        current_key_index = (current_key_index + 1) % len(GEMINI_API_KEYS)  # Move to next key
        logger.info(f"🔄 SWITCHED to Gemini API key #{current_key_index + 1} - Now using KEY_{current_key_index + 1}")
    
    return current_key

app = FastAPI()

# ---- Auth Token ----
TEAM_TOKEN = "833695cad1c0d2600066bf2b08aab7614d0dec93b4b6f0ae3acd37ef7d6fcb1c"

# ---- Data Models ----
class QueryRequest(BaseModel):
    documents: str
    questions: List[str]

class QueryResponse(BaseModel):
    answers: List[str]

# ---- Word to Number Conversion ----
def words_to_numbers(text):
    """Convert written numbers to digits for better matching"""
    word_to_num = {
        'zero': '0', 'one': '1', 'two': '2', 'three': '3', 'four': '4', 'five': '5',
        'six': '6', 'seven': '7', 'eight': '8', 'nine': '9', 'ten': '10',
        'eleven': '11', 'twelve': '12', 'thirteen': '13', 'fourteen': '14', 'fifteen': '15',
        'sixteen': '16', 'seventeen': '17', 'eighteen': '18', 'nineteen': '19', 'twenty': '20',
        'thirty': '30', 'forty': '40', 'fifty': '50', 'sixty': '60', 'seventy': '70',
        'eighty': '80', 'ninety': '90', 'hundred': '100', 'thousand': '1000', 'million': '1000000'
    }
    
    for word, num in word_to_num.items():
        text = re.sub(r'\b' + word + r'\b', num, text, flags=re.IGNORECASE)
    
    return text

# ---- Optimized PDF Downloader ----
async def download_pdf_text(url: str) -> str:
    start_time = time.time()
    try:
        logger.info(f"Starting PDF download from: {url}")
        
        # Faster timeout and connection settings
        timeout = httpx.Timeout(20.0, connect=5.0)
        limits = httpx.Limits(max_keepalive_connections=5, max_connections=10)
        
        async with httpx.AsyncClient(timeout=timeout, limits=limits, follow_redirects=True) as client:
            response = await client.get(url)
        
        response.raise_for_status()
        logger.info(f"PDF downloaded successfully. Size: {len(response.content)} bytes")
        
        # Use memory buffer instead of file
        doc = fitz.open(stream=response.content, filetype="pdf")
        text_parts = []
        
        # Process only first 50 pages for speed
        max_pages = min(50, doc.page_count)
        logger.info(f"Processing {max_pages} pages from PDF")
        
        for page_num in range(max_pages):
            page = doc[page_num]
            page_text = page.get_text()
            if page_text.strip():
                # Convert word numbers to digits for better matching
                page_text = words_to_numbers(page_text)
                # Basic cleaning during extraction
                cleaned_text = re.sub(r'\s+', ' ', page_text.strip())
                text_parts.append(cleaned_text)
        
        doc.close()
        full_text = "\n\n".join(text_parts)
        
        logger.info(f"PDF processing completed in {time.time() - start_time:.2f}s. Text length: {len(full_text)}")
        return full_text
        
    except Exception as e:
        logger.error(f"PDF download/extraction failed: {str(e)}")
        raise Exception(f"Download or extraction failed: {str(e)}")

# ---- Optimized Chunking ----
def split_text_into_chunks(text: str, chunk_size: int = 500, overlap: int = 100) -> List[str]:
    start_time = time.time()
    logger.info("Starting enhanced text chunking")
    
    # Pre-clean text once
    text = re.sub(r'\s+', ' ', text.strip())
    
    # Better sentence boundary detection
    sentences = []
    # Split on sentence boundaries but keep some context
    sentence_pattern = r'(?<=[.!?])\s+(?=[A-Z])'
    raw_sentences = re.split(sentence_pattern, text)
    
    for sentence in raw_sentences:
        sentence = sentence.strip()
        if len(sentence) > 15:  # Filter very short fragments
            sentences.append(sentence)
    
    logger.info(f"Split into {len(sentences)} sentences")
    
    # Build chunks with better overlap management - INCREASED OVERLAP
    chunks = []
    words = text.split()
    
    for i in range(0, len(words), chunk_size - overlap):
        chunk_words = words[i:i + chunk_size]
        if len(chunk_words) >= 20:  # Ensure meaningful chunks
            chunk_text = " ".join(chunk_words)
            # Ensure chunk ends at reasonable boundary if possible
            if len(chunk_text) < len(text) and not chunk_text.endswith(('.', '!', '?')):
                # Try to extend to next sentence boundary
                next_words = words[i + chunk_size:i + chunk_size + 20]
                if next_words:
                    extended = " ".join(chunk_words + next_words)
                    next_boundary = extended.find('.', len(chunk_text))
                    if next_boundary != -1 and next_boundary < len(chunk_text) + 100:
                        chunk_text = extended[:next_boundary + 1]
            
            chunks.append(chunk_text)
    
    logger.info(f"Created {len(chunks)} enhanced chunks in {time.time() - start_time:.2f}s")
    return chunks


# ---- Optimized Embeddings ----
async def get_embeddings(texts: List[str], input_type: str = "search_document") -> List[List[float]]:
    start_time = time.time()
    logger.info(f"Getting embeddings for {len(texts)} texts")
    
    url = "https://api.cohere.com/v1/embed"
    headers = {
        "Authorization": f"Bearer {COHERE_API_KEY}",
        "Content-Type": "application/json",
        "Accept": "application/json"
    }
    
    # Pre-clean texts and convert word numbers
    clean_texts = []
    for text in texts:
        if text.strip():
            converted_text = words_to_numbers(text)
            clean_text = re.sub(r'\s+', ' ', converted_text.strip())[:2000]
            clean_texts.append(clean_text)
    
    if not clean_texts:
        raise ValueError("No valid texts provided for embedding")

    # Larger batches for speed
    BATCH_SIZE = 96  # Cohere's limit
    all_embeddings = []
    
    # Use connection pooling
    timeout = httpx.Timeout(45.0)
    limits = httpx.Limits(max_keepalive_connections=5, max_connections=10)
    
    async with httpx.AsyncClient(timeout=timeout, limits=limits) as client:
        for i in range(0, len(clean_texts), BATCH_SIZE):
            batch = clean_texts[i:i+BATCH_SIZE]
            
            data = {
                "model": "embed-english-v3.0",
                "texts": batch,
                "input_type": input_type,
                "truncate": "END"  # Handle long texts
            }
            
            try:
                response = await client.post(url, headers=headers, json=data)
                
                if response.status_code == 200:
                    response_data = response.json()
                    embeddings_data = response_data.get("embeddings", [])
                    
                    # Fast normalization
                    for embedding in embeddings_data:
                        vec = np.array(embedding, dtype=np.float32)
                        norm = np.linalg.norm(vec)
                        if norm > 0:
                            vec = vec / norm
                        all_embeddings.append(vec.tolist())
                else:
                    logger.error(f"Cohere API error: {response.status_code} - {response.text}")
                    response.raise_for_status()
                    
            except Exception as e:
                logger.error(f"Embedding batch {i//BATCH_SIZE + 1} failed: {str(e)}")
                raise e
    
    logger.info(f"Got {len(all_embeddings)} embeddings in {time.time() - start_time:.2f}s")
    return all_embeddings

# ---- Optimized Similarity Search ----
def cosine_similarity_batch(query_vec: List[float], chunk_embeddings: List[List[float]]) -> List[float]:
    """Vectorized cosine similarity for speed"""
    query_arr = np.array(query_vec, dtype=np.float32)
    chunk_arr = np.array(chunk_embeddings, dtype=np.float32)
    
    # Batch dot product
    similarities = np.dot(chunk_arr, query_arr)
    return similarities.tolist()

async def search_similar_chunks(query_embedding: List[float], chunk_embeddings: List[List[float]], 
                               chunks: List[str], question: str, k=15) -> List[str]:
    start_time = time.time()
    
    # Fast vectorized similarity
    similarities = cosine_similarity_batch(query_embedding, chunk_embeddings)
    max_similarity = max(similarities) if similarities else 0
    
    # Extract key terms for keyword matching
    key_terms = extract_key_terms(question.lower())
    question_lower = question.lower()
    
    # Score all chunks (don't break early)
    scored_chunks = []
    
    for i, sim_score in enumerate(similarities):
        chunk = chunks[i]
        chunk_lower = chunk.lower()
        
        # Enhanced keyword scoring with NUMBER MATCHING
        keyword_score = 0
        if key_terms:
            # Exact matches get higher score
            exact_matches = sum(1 for term in key_terms if f" {term} " in f" {chunk_lower} ")
            partial_matches = sum(1 for term in key_terms if term in chunk_lower and f" {term} " not in f" {chunk_lower} ")
            
            # BOOST for numeric matches
            numeric_matches = 0
            for term in key_terms:
                if re.search(r'\d', term):  # If term contains digits
                    if term in chunk_lower:
                        numeric_matches += 2  # Double weight for numbers
            
            keyword_score = (exact_matches * 0.25 + partial_matches * 0.1 + numeric_matches * 0.15) / len(key_terms)
        
        # Question word overlap bonus
        question_words = set(re.findall(r'\b\w+\b', question_lower))
        chunk_words = set(re.findall(r'\b\w+\b', chunk_lower))
        overlap_score = len(question_words.intersection(chunk_words)) / len(question_words) * 0.15
        
        final_score = sim_score + keyword_score + overlap_score
        scored_chunks.append((i, chunk, final_score, sim_score))
    
    # Sort by final score
    scored_chunks.sort(key=lambda x: x[2], reverse=True)
    
    # LOWERED threshold for better recall
    if max_similarity < 0.25:
        logger.warning(f"🔍 Low semantic similarity ({max_similarity:.3f}) - enhancing with keyword search")
        # Boost chunks with strong keyword matches
        keyword_boosted = []
        for i, chunk, final_score, sem_score in scored_chunks:
            if any(term in chunk.lower() for term in key_terms):
                keyword_boosted.append((i, chunk, final_score + 0.4, sem_score))  # Higher boost
            else:
                keyword_boosted.append((i, chunk, final_score, sem_score))
        
        keyword_boosted.sort(key=lambda x: x[2], reverse=True)
        scored_chunks = keyword_boosted
    
    # Return top k chunks with metadata for prompt
    result_chunks = []
    for idx, (chunk_idx, chunk, final_score, sem_score) in enumerate(scored_chunks[:k]):
        # Add chunk index for better LLM traceability
        indexed_chunk = f"[CONTEXT {idx+1}] {chunk}"
        result_chunks.append(indexed_chunk)
    
    logger.info(f"Similarity search completed in {time.time() - start_time:.2f}s. Max sim: {max_similarity:.3f}")
    return result_chunks

# ---- Enhanced Key Terms ----
def extract_key_terms(question: str) -> List[str]:
    # Enhanced insurance-specific terms
    important_terms = {
        'coverage', 'limit', 'deductible', 'premium', 'claim', 'benefit', 'exclusion',
        'copay', 'coinsurance', 'maximum', 'minimum', 'annual', 'lifetime', 'policy',
        'insured', 'covered', 'eligible', 'amount', 'percentage', 'network', 'provider',
        'emergency', 'prescription', 'medical', 'dental', 'vision', 'mental', 'health',
        'hospital', 'outpatient', 'inpatient', 'surgery', 'diagnostic', 'preventive',
        'waiting', 'period', 'authorization', 'existing', 'condition'
    }
    
    # Convert question word numbers to digits first
    question_converted = words_to_numbers(question.lower())
    
    # Fast extraction with better pattern matching
    question_words = set(re.findall(r'\b\w+\b', question_converted))
    key_terms = list(important_terms.intersection(question_words))
    
    # Add numbers, percentages, and dollar amounts (IMPROVED)
    numbers = re.findall(r'\b\d+(?:\.\d+)?%?\b', question_converted)
    dollar_amounts = re.findall(r'\$[\d,]+(?:\.\d{2})?', question_converted)
    key_terms.extend(numbers + dollar_amounts)
    
    # Add important phrases (2-3 words)
    phrases = re.findall(r'\b(?:out of pocket|prior authorization|pre existing|waiting period|per year|per day)\b', question_converted)
    key_terms.extend(phrases)
    
    return key_terms

def is_yes_no_question(question: str) -> bool:
    question_lower = question.lower().strip()
    return question_lower.startswith(('is ', 'are ', 'do ', 'does ', 'did ', 'will ', 'would ', 'can ', 'could '))

def build_prompt(question: str, context_chunks: List[str]) -> str:
    # More context for better accuracy
    context = "\n\n---\n\n".join(context_chunks[:12])
    
    is_yes_no = is_yes_no_question(question)
    key_terms = extract_key_terms(question.lower())
    
    # Create hints for better context matching
    hints = ""
    if key_terms:
        hints = f"""
**KEY TERMS TO LOOK FOR:** {", ".join(key_terms[:10])}
(The above terms from your question should help identify relevant information in the context below)
"""
    
    if is_yes_no:
        response_instruction = """This is a yes/no question. Start your response with "Yes" or "No", then provide a brief explanation with specific details from the context in 25-40 words total."""
    else:
        response_instruction = """Answer in ONE concise, direct paragraph (30-60 words maximum). Include specific numbers, amounts, percentages, and conditions from the context. Use figures (1, 2, 3) instead of words (one, two, three) for all numbers."""

    return f"""You are an expert insurance policy analyst. Analyze the provided policy context carefully and provide precise answers.

**CRITICAL INSTRUCTIONS:**
1. {response_instruction}
2. ALWAYS quote specific numbers, percentages, dollar amounts, limits, and conditions from the context
3. Look for information across ALL context sections - scan every section thoroughly
4. If you find partial information, state what you found and note what's missing
5. Include exceptions, sub-limits, waiting periods, or special conditions when mentioned
6. Use simple, clear language without markdown formatting
7. ALWAYS use numeric figures (1, 2, 50, 100) never spell out numbers as words
8. Only say "The information is not available in the provided context" if NONE of the context sections contain any relevant information about the topic

**ANALYSIS APPROACH:**
- Scan each [CONTEXT X] section thoroughly for relevant information
- Look for the key terms mentioned above AND related concepts
- Cross-reference information between sections
- Pay special attention to numbers, amounts, percentages, and conditions
- Consider partial matches and related information

{hints}

**POLICY CONTEXT:**
{context}

**QUESTION:** {question}

**ANSWER:**"""

# ---- Optimized LLM Call with API Key Rotation ----
async def ask_llm(prompt: str, retry_count: int = 0) -> str:
    try:
        # Get current API key with rotation
        current_api_key = get_current_gemini_key()
        logger.info(f"🔑 Using Gemini API KEY_{GEMINI_API_KEYS.index(current_api_key) + 1} (Request #{request_count}/12)")
        
        start_time = time.time()
        
        client = genai.Client(api_key=current_api_key)
        from google.genai import types
        
        response = client.models.generate_content(
            model="gemini-2.5-flash-lite", 
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.1,  # Slightly higher for less rigid responses
                max_output_tokens=180,  # More tokens for complete answers
                top_p=0.9,
                top_k=25,
            )
        )
        
        logger.info(f"Gemini API call completed in {time.time() - start_time:.2f}s")
        
        result_text = ""
        if hasattr(response, 'text') and response.text:
            result_text = response.text.strip()
        elif hasattr(response, 'candidates') and response.candidates:
            candidate = response.candidates[0]
            if hasattr(candidate, 'content') and candidate.content.parts:
                result_text = candidate.content.parts[0].text.strip()
        
        # Enhanced response cleaning with number conversion
        if result_text:
            # Convert any word numbers to digits in the response
            result_text = words_to_numbers(result_text)
            # Remove common prefixes
            result_text = re.sub(r'^(Answer:|ANSWER:|Response:|Based on the context:)\s*', '', result_text, flags=re.IGNORECASE)
            # Clean up extra whitespace
            result_text = re.sub(r'\s+', ' ', result_text).strip()
            # Remove trailing periods only if they seem added artificially
            if result_text.endswith('.') and not result_text.endswith('etc.') and not result_text.endswith('vs.'):
                result_text = result_text[:-1]
            
            return result_text
        
        return "No valid response generated."
            
    except Exception as e:
        logger.error(f"Gemini API error (attempt {retry_count + 1}): {str(e)}")
        
        # Simple retry logic for transient errors
        if retry_count < 1 and ("quota" not in str(e).lower() and "limit" not in str(e).lower()):
            logger.info("Retrying with next API key...")
            return await ask_llm(prompt, retry_count + 1)
        
        return f"Unable to generate answer due to API error. Please try again."
    
    
@app.get("/")
async def root():
    return {
        "message": "Server Running", 
        "status": "accuracy_optimized",
        "api_keys_count": len(GEMINI_API_KEYS),
        "current_key_index": current_key_index + 1,
        "requests_on_current_key": request_count
    }

# ---- Main API Endpoint ----
@app.post("/hackrx/run", response_model=QueryResponse)
async def run_query(request: QueryRequest, authorization: str = Header(None)):
    global qa_storage  # Access global array
    
    total_start_time = time.time()
    logger.info(f"Received request with {len(request.questions)} questions")
    
    # Auth check
    if not authorization or not authorization.startswith("Bearer "):
        logger.error("Missing or malformed authorization header")
        raise HTTPException(status_code=401, detail="Authorization header missing or malformed.")

    token = authorization.split(" ")[1]
    if token != TEAM_TOKEN:
        logger.error("Invalid token provided")
        raise HTTPException(status_code=403, detail="Invalid token")

    try:
        # Process document
        logger.info("Starting document processing")
        document_text = await download_pdf_text(request.documents)
        
        if not document_text.strip():
            logger.error("No text extracted from document")
            raise HTTPException(status_code=400, detail="No text could be extracted from the document")
            
        chunks = split_text_into_chunks(document_text)
        if not chunks:
            logger.error("No chunks created from document")
            raise HTTPException(status_code=400, detail="No meaningful chunks could be created from the document")
            
        # Get embeddings
        logger.info("Getting embeddings")
        chunk_embeddings = await get_embeddings(chunks, input_type="search_document")
        question_embeddings = await get_embeddings(request.questions, input_type="search_query")
        
        # Process questions
        logger.info("Processing questions")
        answers = []
        for i, (question, q_emb) in enumerate(zip(request.questions, question_embeddings)):
            try:
                logger.info(f"Processing question {i+1}/{len(request.questions)}: {question[:50]}...")
                
                relevant_chunks = await search_similar_chunks(q_emb, chunk_embeddings, chunks, question)
                if not relevant_chunks:
                    answer = "The information is not available in the provided context."
                    answers.append(answer)
                    # Store Q&A pair
                    qa_storage.append([question, answer])
                    continue
                    
                prompt = build_prompt(question, relevant_chunks)
                response = await ask_llm(prompt)
                
                # Clean response
                response = response.strip()
                response = re.sub(r'^(Answer:|ANSWER:|Response:)\s*', '', response, flags=re.IGNORECASE)
                response = re.sub(r'\s*\.$', '', response)
                
                final_answer = response.strip()
                answers.append(final_answer)
                
                # Store Q&A pair in nested array format
                qa_storage.append([question, final_answer])
                
                logger.info(f"Question {i+1} processed successfully")
                
            except Exception as e:
                logger.error(f"Error processing question {i+1} '{question}': {str(e)}")
                error_answer = "Error processing the question. Please try again."
                answers.append(error_answer)
                # Store Q&A pair even for errors
                qa_storage.append([question, error_answer])

        total_time = time.time() - total_start_time
        logger.info(f"Total request processed in {total_time:.2f}s")
        
        # ---- LOG ALL Q&A PAIRS AFTER SUCCESSFUL EXECUTION ----
        logger.info("📋 ALL QUESTIONS AND ANSWERS:")
        logger.info(f"{qa_storage}")
        
        # Empty the array after printing to save space
        qa_storage.clear()
        logger.info("✅ Q&A storage cleared to save memory")
        
        return QueryResponse(answers=answers)
        
    except HTTPException as he:
        logger.error(f"HTTP Exception: {he.detail}")
        raise he
    except Exception as e:
        logger.error(f"Unexpected error in run_query: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/health")
async def health_check():
    return {
        "status": "healthy", 
        "memory_usage": "optimized", 
        "version": "accuracy_improved_with_number_conversion",
        "api_keys_available": len(GEMINI_API_KEYS),
        "current_key": current_key_index + 1,
        "requests_on_current_key": request_count,
        "qa_storage_size": len(qa_storage)
    }

@app.get("/api-status")
async def api_status():
    """Endpoint to check API key rotation status"""
    return {
        "total_api_keys": len(GEMINI_API_KEYS),
        "current_key_index": current_key_index + 1,
        "requests_on_current_key": request_count,
        "requests_per_key_limit": REQUESTS_PER_KEY,
        "qa_storage_current_size": len(qa_storage)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")