from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel
import httpx
import asyncio
import fitz
from google import genai
import os
from dotenv import load_dotenv
from typing import List, Dict, Tuple
import numpy as np
import faiss
import json
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import functools
import re
from sentence_transformers import SentenceTransformer
import torch
from collections import defaultdict
import hashlib

# ---- Load API Keys ----
load_dotenv()
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

app = FastAPI()
faiss_cache = {}
embedding_cache = {}  # New: Cache for embeddings

# ---- Thread Pool Executors ----
io_executor = ThreadPoolExecutor(max_workers=6)
cpu_executor = ProcessPoolExecutor(max_workers=4)

# ---- Auth Token ----
TEAM_TOKEN = "833695cad1c0d2600066bf2b08aab7614d0dec93b4b6f0ae3acd37ef7d6fcb1c"

# ---- Data Models ----
class QueryRequest(BaseModel):
    documents: str
    questions: List[str]

class QueryResponse(BaseModel):
    answers: List[str]

# ---- PDF Downloader with improved error handling ----
async def download_pdf_text(url: str) -> str:
    try:
        timeout = httpx.Timeout(30.0)
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.get(url)
        response.raise_for_status()
        
        with open("temp.pdf", "wb") as f:
            f.write(response.content)

        def extract_text():
            with fitz.open("temp.pdf") as doc:
                text = ""
                for page in doc:
                    # Improved text extraction with better formatting
                    blocks = page.get_text("dict")
                    page_text = ""
                    for block in blocks["blocks"]:
                        if "lines" in block:
                            for line in block["lines"]:
                                line_text = ""
                                for span in line["spans"]:
                                    line_text += span["text"]
                                if line_text.strip():
                                    page_text += line_text + " "
                            page_text += "\n"
                    
                    if page_text.strip():
                        text += page_text + "\n\n"
                return text
        
        loop = asyncio.get_event_loop()
        text = await loop.run_in_executor(io_executor, extract_text)
        
        try:
            os.remove("temp.pdf")
        except:
            pass
            
        return text.strip()
    except Exception as e:
        raise Exception(f"Download or extraction failed: {str(e)}")

# ---- Improved Chunking with better semantic boundaries ----
def split_text_into_chunks(text: str, chunk_size: int = 450, overlap: int = 75) -> List[str]:
    # Clean and normalize text first
    text = re.sub(r'\s+', ' ', text.strip())
    
    # Enhanced sentence splitting that handles insurance terminology
    sentence_patterns = [
        r'(?<=[.!?])\s+(?=[A-Z])',
        r'(?<=\.)\s+(?=\d+\.)',  # Numbered lists
        r'(?<=:)\s*\n\s*(?=[A-Z])',  # After colons
        r'(?<=;)\s+(?=[A-Z])',  # After semicolons
    ]
    
    sentences = [text]
    for pattern in sentence_patterns:
        new_sentences = []
        for sent in sentences:
            new_sentences.extend(re.split(pattern, sent))
        sentences = [s.strip() for s in new_sentences if s.strip()]
    
    chunks = []
    current_chunk = []
    current_length = 0
    
    for sentence in sentences:
        words_in_sentence = len(sentence.split())
        
        if current_length + words_in_sentence > chunk_size and current_chunk:
            chunk_text = " ".join(current_chunk).strip()
            if chunk_text and len(chunk_text.split()) >= 15:  # Minimum meaningful chunk size
                chunks.append(chunk_text)
            
            # Smart overlap: preserve context around key terms
            overlap_text = " ".join(current_chunk)
            overlap_words = overlap_text.split()[-overlap:]
            
            # Extend overlap if it cuts off in the middle of important terms
            important_terms = ['coverage', 'limit', 'deductible', 'premium', 'exclusion', 'benefit', 'claim']
            if overlap_words and any(term in overlap_words[-1].lower() for term in important_terms):
                # Find a better breaking point
                for i in range(min(10, len(overlap_words))):
                    if overlap_words[-(i+1)].endswith(('.', '!', '?', ';')):
                        overlap_words = overlap_words[-(i+1):]
                        break
            
            current_chunk = overlap_words + [sentence]
            current_length = len(overlap_words) + words_in_sentence
        else:
            current_chunk.append(sentence)
            current_length += words_in_sentence
    
    if current_chunk:
        chunk_text = " ".join(current_chunk).strip()
        if chunk_text and len(chunk_text.split()) >= 15:
            chunks.append(chunk_text)
    
    return chunks

# ---- Enhanced embedding caching ----
def get_text_hash(text: str) -> str:
    return hashlib.md5(text.encode()).hexdigest()

# ---- Cohere v3 Embeddings with caching ----
async def get_embeddings(texts: List[str], input_type: str = "search_document") -> List[List[float]]:
    # Check cache first
    cached_embeddings = []
    texts_to_embed = []
    text_indices = []
    
    for i, text in enumerate(texts):
        text_hash = get_text_hash(text)
        cache_key = f"{text_hash}_{input_type}"
        
        if cache_key in embedding_cache:
            cached_embeddings.append((i, embedding_cache[cache_key]))
        else:
            texts_to_embed.append(text)
            text_indices.append(i)
    
    # Get embeddings for uncached texts
    new_embeddings = []
    if texts_to_embed:
        new_embeddings = await _get_embeddings_from_api(texts_to_embed, input_type)
        
        # Cache new embeddings
        for text, embedding in zip(texts_to_embed, new_embeddings):
            text_hash = get_text_hash(text)
            cache_key = f"{text_hash}_{input_type}"
            embedding_cache[cache_key] = embedding
    
    # Combine cached and new embeddings in correct order
    all_embeddings = [None] * len(texts)
    
    for i, embedding in cached_embeddings:
        all_embeddings[i] = embedding
    
    for i, embedding in zip(text_indices, new_embeddings):
        all_embeddings[i] = embedding
    
    return all_embeddings

async def _get_embeddings_from_api(texts: List[str], input_type: str = "search_document") -> List[List[float]]:
    url = "https://api.cohere.com/v1/embed"
    headers = {
        "Authorization": f"Bearer {COHERE_API_KEY}",
        "Content-Type": "application/json",
        "Accept": "application/json"
    }
    
    if not isinstance(texts, list) or not all(isinstance(t, str) for t in texts):
        raise ValueError("Input to get_embeddings must be a list of strings.")

    clean_texts = [re.sub(r'\s+', ' ', text.strip()) for text in texts if text.strip()]
    
    if not clean_texts:
        raise ValueError("No valid texts provided for embedding")

    BATCH_SIZE = 90
    batches = [clean_texts[i:i+BATCH_SIZE] for i in range(0, len(clean_texts), BATCH_SIZE)]

    async def fetch_batch(batch, batch_num):
        data = {
            "model": "embed-english-v3.0",
            "texts": batch,
            "input_type": input_type
        }
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                timeout = httpx.Timeout(60.0)
                async with httpx.AsyncClient(timeout=timeout) as client:
                    response = await client.post(url, headers=headers, json=data)
                    
                    if response.status_code == 200:
                        response_data = response.json()
                        
                        batch_embeddings = []
                        embeddings_data = response_data.get("embeddings", response_data.get("data", []))
                            
                        for embedding in embeddings_data:
                            if isinstance(embedding, dict):
                                vec_data = embedding.get("embedding", embedding)
                            else:
                                vec_data = embedding
                            
                            vec = np.array(vec_data, dtype="float32")
                            vec = vec / np.linalg.norm(vec)
                            batch_embeddings.append(vec.tolist())
                                
                        return batch_embeddings
                    elif response.status_code == 429:
                        wait_time = 2 ** attempt
                        await asyncio.sleep(wait_time)
                        continue
                    else:
                        response.raise_for_status()
                        
            except Exception as e:
                if attempt == max_retries - 1:
                    raise e
                await asyncio.sleep(1)
        
        raise Exception("Failed to get embeddings after all retries")

    all_results = []
    for i, batch in enumerate(batches):
        if i > 0:
            await asyncio.sleep(0.1)
        result = await fetch_batch(batch, i + 1)
        all_results.append(result)
    
    return [emb for batch in all_results for emb in batch]

# ---- Enhanced FAISS with better index configuration ----
def create_faiss_index(embeddings_list):
    embeddings_array = np.array(embeddings_list).astype("float32")
    dim = embeddings_array.shape[1]
    
    # Use more sophisticated index for better accuracy
    if len(embeddings_list) < 1000:
        # For smaller datasets, use exact search
        index = faiss.IndexFlatIP(dim)  # Inner Product for normalized vectors
    else:
        # For larger datasets, use HNSW with optimized parameters
        index = faiss.IndexHNSWFlat(dim, 32)
        index.hnsw.efConstruction = 200
        index.hnsw.efSearch = 80  # Increased for better recall
    
    index.add(embeddings_array)
    return index

# ---- Enhanced FAISS Indexing ----
async def build_faiss_index(chunks: List[str]):
    embeddings = await get_embeddings(chunks, input_type="search_document")
    
    loop = asyncio.get_event_loop()
    index = await loop.run_in_executor(cpu_executor, create_faiss_index, embeddings)
    return index, chunks

# ---- Improved retrieval with hybrid scoring ----
async def search_faiss(index, query_embedding: List[float], chunks: List[str], question: str, k=25) -> List[str]:
    query_vec = np.array([query_embedding]).astype("float32")
    D, I = index.search(query_vec, min(k, len(chunks)))

    # Extract key terms from question for relevance scoring
    question_lower = question.lower()
    key_terms = extract_key_terms(question_lower)
    
    retrieved_with_scores = []
    for j, i in enumerate(I[0]):
        if i != -1:
            chunk = chunks[i]
            semantic_score = float(D[0][j])
            
            # Calculate keyword relevance score
            keyword_score = calculate_keyword_relevance(chunk.lower(), key_terms)
            
            # Calculate number/percentage relevance if question asks for specific values
            number_score = calculate_number_relevance(chunk, question_lower)
            
            # Combined score (semantic similarity + keyword relevance + number relevance)
            combined_score = semantic_score + (keyword_score * 0.3) + (number_score * 0.2)
            
            retrieved_with_scores.append((chunk, combined_score))
    
    # Sort by combined score (higher is better for combined scoring)
    retrieved_with_scores.sort(key=lambda x: x[1], reverse=True)
    
    # Return top chunks with deduplication
    return deduplicate_chunks([chunk for chunk, _ in retrieved_with_scores[:12]])

def extract_key_terms(question: str) -> List[str]:
    """Extract important terms from the question"""
    # Insurance-specific important terms
    important_terms = {
        'coverage', 'limit', 'deductible', 'premium', 'claim', 'benefit', 'exclusion',
        'copay', 'coinsurance', 'maximum', 'minimum', 'annual', 'lifetime', 'waiting',
        'period', 'ppn', 'network', 'out-of-network', 'preauthorization', 'emergency'
    }
    
    # Extract numbers and percentages
    numbers = re.findall(r'\b\d+(?:\.\d+)?%?\b', question)
    
    # Extract quoted terms
    quoted_terms = re.findall(r'"([^"]*)"', question)
    
    # Extract capitalized terms (likely important)
    capitalized = re.findall(r'\b[A-Z][A-Za-z]+\b', question)
    
    # Combine all important terms
    question_words = set(question.split())
    key_terms = list(important_terms.intersection(question_words))
    key_terms.extend(numbers)
    key_terms.extend(quoted_terms)
    key_terms.extend([term.lower() for term in capitalized if len(term) > 2])
    
    return list(set(key_terms))

def calculate_keyword_relevance(chunk: str, key_terms: List[str]) -> float:
    """Calculate how many key terms appear in the chunk"""
    if not key_terms:
        return 0.0
    
    chunk_words = set(chunk.split())
    matches = sum(1 for term in key_terms if term in chunk_words or term in chunk)
    return matches / len(key_terms)

def calculate_number_relevance(chunk: str, question: str) -> float:
    """Boost chunks that contain numbers when question asks for specific values"""
    question_has_numbers = bool(re.search(r'\b(?:how much|what.*(?:amount|cost|limit|percentage)|[\d%$])\b', question))
    
    if question_has_numbers:
        # Count numbers, percentages, and currency in chunk
        numbers = len(re.findall(r'\b\d+(?:\.\d+)?(?:%|\$|dollars?\b)?\b', chunk))
        return min(numbers * 0.1, 1.0)  # Cap at 1.0
    
    return 0.0

def deduplicate_chunks(chunks: List[str], similarity_threshold: float = 0.7) -> List[str]:
    """Remove highly similar chunks to avoid redundancy"""
    if len(chunks) <= 1:
        return chunks
    
    unique_chunks = [chunks[0]]
    
    for chunk in chunks[1:]:
        chunk_words = set(chunk.lower().split())
        is_duplicate = False
        
        for existing_chunk in unique_chunks:
            existing_words = set(existing_chunk.lower().split())
            if len(chunk_words) > 0 and len(existing_words) > 0:
                intersection = len(chunk_words.intersection(existing_words))
                union = len(chunk_words.union(existing_words))
                jaccard_sim = intersection / union if union > 0 else 0
                
                if jaccard_sim > similarity_threshold:
                    is_duplicate = True
                    break
        
        if not is_duplicate:
            unique_chunks.append(chunk)
    
    return unique_chunks

# ---- Enhanced Prompt Builder with better structure ----
def build_prompt(question: str, context_chunks: List[str]) -> str:
    # Prioritize chunks based on content type
    prioritized_chunks = prioritize_chunks(context_chunks, question)
    context = "\n\n---\n\n".join(prioritized_chunks)
    
    # Determine if it's a yes/no question
    is_yes_no = is_yes_no_question(question)
    
    response_instruction = """Answer in ONE concise, direct paragraph (25-40 words maximum).
Only start your response with "Yes" or "No" if this is a direct yes/no question. Otherwise, respond naturally."""

    if is_yes_no:
        response_instruction = """This is a yes/no question. Start your response with "Yes" or "No", then provide a brief explanation in 20-35 words total."""

    return f"""You are an expert insurance policy analyst. Provide precise, actionable answers based strictly on the provided policy context.

**INSTRUCTIONS:**
1. {response_instruction}
2. ALWAYS include specific numbers, percentages, limits, deductibles, and conditions from the context
3. If context mentions exceptions, sub-limits, or special cases, include them explicitly
4. Quote exact amounts, percentages, and terms from the policy
5. If information isn't in context or question is unrelated to insurance, respond: "The information is not available in the provided context."
6. Use professional language without markdown or bullet points

**CONTEXT:**
{context}

**QUESTION:** {question}

**ANSWER:**"""

def is_yes_no_question(question: str) -> bool:
    """Detect if a question expects a yes/no answer"""
    question_lower = question.lower().strip()
    
    yes_no_patterns = [
        r'^(?:is|are|do|does|did|will|would|can|could|should|may|might|has|have|was|were)\b',
        r'\b(?:whether|if)\b.*\?',
        r'^(?:true or false|correct or incorrect)',
    ]
    
    return any(re.search(pattern, question_lower) for pattern in yes_no_patterns)

def prioritize_chunks(chunks: List[str], question: str) -> List[str]:
    """Prioritize chunks based on content relevance"""
    if len(chunks) <= 3:
        return chunks
    
    question_lower = question.lower()
    
    # Scoring criteria
    def score_chunk(chunk):
        chunk_lower = chunk.lower()
        score = 0
        
        # Boost chunks with numbers/amounts if question asks for them
        if any(term in question_lower for term in ['how much', 'what amount', 'cost', 'limit', 'percentage']):
            numbers = len(re.findall(r'\b\d+(?:\.\d+)?(?:%|\$)?\b', chunk))
            score += numbers * 2
        
        # Boost chunks with policy-specific terms
        policy_terms = ['coverage', 'benefit', 'limit', 'deductible', 'exclusion', 'claim']
        for term in policy_terms:
            if term in chunk_lower:
                score += 1
        
        # Boost chunks that directly answer the question type
        if 'covered' in question_lower and 'covered' in chunk_lower:
            score += 3
        if 'excluded' in question_lower and ('exclusion' in chunk_lower or 'excluded' in chunk_lower):
            score += 3
            
        return score
    
    scored_chunks = [(chunk, score_chunk(chunk)) for chunk in chunks]
    scored_chunks.sort(key=lambda x: x[1], reverse=True)
    
    return [chunk for chunk, _ in scored_chunks]

# ---- Enhanced LLM Call with better error handling ----
def ask_llm_sync(prompt: str) -> str:
    try:
        client = genai.Client(api_key=GEMINI_API_KEY)
        from google.genai import types
        
        response = client.models.generate_content(
            model="gemini-2.5-flash-lite", 
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.0,
                max_output_tokens=150,
                top_p=0.8,
                top_k=20,
                candidate_count=1,
                stop_sequences=None
            )
        )
        
        if hasattr(response, 'text') and response.text:
            return response.text.strip()
        elif hasattr(response, 'candidates') and response.candidates:
            candidate = response.candidates[0]
            if hasattr(candidate, 'content') and candidate.content.parts:
                return candidate.content.parts[0].text.strip()
            return str(candidate).strip()
        else:
            return "No valid response from Gemini AI."
            
    except Exception as e:
        try:
            response = client.models.generate_content(
                model="gemini-1.5-pro",
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.0,
                    max_output_tokens=150,
                    top_p=0.9
                )
            )
            if hasattr(response, 'text') and response.text:
                return response.text.strip()
        except Exception as fallback_e:
            return f"Error generating answer: {str(fallback_e)}"

async def ask_llm(prompt: str) -> str:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(io_executor, ask_llm_sync, prompt)

@app.get("/")
async def root():
    return {"message": "Server Running"}

# ---- Enhanced API Endpoint ----
@app.post("/hackrx/run", response_model=QueryResponse)
async def run_query(request: QueryRequest, authorization: str = Header(None)):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Authorization header missing or malformed.")

    token = authorization.split(" ")[1]
    if token != TEAM_TOKEN:
        raise HTTPException(status_code=403, detail="Invalid token")

    try:
        cache_key = f"{request.documents}_{hash(str(sorted(request.questions)))}"
        
        if request.documents in faiss_cache:
            index, chunk_list = faiss_cache[request.documents]
        else:
            document_text = await download_pdf_text(request.documents)
            if not document_text.strip():
                raise HTTPException(status_code=400, detail="No text could be extracted from the document")
                
            chunks = split_text_into_chunks(document_text)
            if not chunks:
                raise HTTPException(status_code=400, detail="No meaningful chunks could be created from the document")
                
            index, chunk_list = await build_faiss_index(chunks)
            faiss_cache[request.documents] = (index, chunk_list)

        question_embeddings = await get_embeddings(request.questions, input_type="search_query")
        
        async def process_question(question, q_emb):
            try:
                relevant_chunks = await search_faiss(index, q_emb, chunk_list, question)
                if not relevant_chunks:
                    return "The information is not available in the provided context."
                    
                prompt = build_prompt(question, relevant_chunks)
                response = await ask_llm(prompt)
                
                # Clean up response
                response = response.strip()
                
                if response.startswith('{"') or response.startswith('{'):
                    try:
                        parsed = json.loads(response)
                        if isinstance(parsed, dict) and 'answer' in parsed:
                            response = parsed['answer']
                    except:
                        pass
                
                response = re.sub(r'^(Answer:|ANSWER:|Response:)\s*', '', response, flags=re.IGNORECASE)
                response = re.sub(r'\s*\.$', '', response)
                
                return response.strip()
                
            except Exception as e:
                print(f"Error processing question '{question}': {str(e)}")
                return "Error processing the question. Please try again."
        
        tasks = [process_question(q, emb) for q, emb in zip(request.questions, question_embeddings)]
        answers = await asyncio.gather(*tasks, return_exceptions=True)
        
        final_answers = []
        for i, answer in enumerate(answers):
            if isinstance(answer, Exception):
                print(f"Exception for question {i}: {str(answer)}")
                final_answers.append("Error processing the question. Please try again.")
            else:
                final_answers.append(str(answer))

        return QueryResponse(answers=final_answers)
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Unexpected error in run_query: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_info": {"embedding": "cohere-v3", "llm": "gemini-2.0-flash-exp"}}

@app.on_event("shutdown")
async def shutdown_executors():
    io_executor.shutdown(wait=True)
    cpu_executor.shutdown(wait=True)