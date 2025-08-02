from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel
import httpx
import asyncio
import fitz
from google import genai
import os
from dotenv import load_dotenv
from typing import List
import numpy as np
import faiss
import json
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import functools
import re
from sentence_transformers import SentenceTransformer
import torch

# ---- Load API Keys ----
load_dotenv()
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

app = FastAPI()
faiss_cache = {}

# ---- Thread Pool Executors ----
io_executor = ThreadPoolExecutor(max_workers=6)  # Increased for better concurrency
cpu_executor = ProcessPoolExecutor(max_workers=4)  # Increased for better parallel processing

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
        timeout = httpx.Timeout(30.0)  # 30 second timeout
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.get(url)
        response.raise_for_status()
        
        with open("temp.pdf", "wb") as f:
            f.write(response.content)

        # Use thread pool for IO-bound PDF opening and text extraction
        def extract_text():
            with fitz.open("temp.pdf") as doc:
                text = ""
                for page in doc:
                    # Better text extraction with layout preservation
                    page_text = page.get_text("text", sort=True)
                    if page_text.strip():  # Only add non-empty pages
                        text += page_text + "\n\n"
                return text
        
        loop = asyncio.get_event_loop()
        text = await loop.run_in_executor(io_executor, extract_text)
        
        # Clean up temporary file
        try:
            os.remove("temp.pdf")
        except:
            pass
            
        return text.strip()
    except Exception as e:
        raise Exception(f"Download or extraction failed: {str(e)}")

# ---- Improved Chunking with semantic awareness ----
def split_text_into_chunks(text: str, chunk_size: int = 400, overlap: int = 50) -> List[str]:
    # More sophisticated sentence splitting that handles edge cases
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
    
    chunks = []
    current_chunk = []
    current_length = 0
    
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
            
        words_in_sentence = len(sentence.split())
        
        # If adding this sentence would exceed chunk_size, finalize current chunk
        if current_length + words_in_sentence > chunk_size and current_chunk:
            chunk_text = " ".join(current_chunk).strip()
            if chunk_text:  # Only add non-empty chunks
                chunks.append(chunk_text)
            
            # Start new chunk with overlap from previous chunk
            overlap_words = " ".join(current_chunk).split()[-overlap:]
            current_chunk = overlap_words + [sentence]
            current_length = len(overlap_words) + words_in_sentence
        else:
            current_chunk.append(sentence)
            current_length += words_in_sentence
    
    # Add the last chunk
    if current_chunk:
        chunk_text = " ".join(current_chunk).strip()
        if chunk_text:
            chunks.append(chunk_text)
    
    # Filter out very short chunks that might not be meaningful
    meaningful_chunks = [chunk for chunk in chunks if len(chunk.split()) >= 10]
    return meaningful_chunks if meaningful_chunks else chunks

# ---- Cohere v3 Embeddings with optimizations ----
async def get_embeddings(texts: List[str], input_type: str = "search_document") -> List[List[float]]:
    url = "https://api.cohere.com/v1/embed"
    headers = {
        "Authorization": f"Bearer {COHERE_API_KEY}",
        "Content-Type": "application/json",
        "Accept": "application/json"
    }
    
    if not isinstance(texts, list) or not all(isinstance(t, str) for t in texts):
        print("[get_embeddings] ERROR: input must be a list of strings.")
        raise ValueError("Input to get_embeddings must be a list of strings.")

    # Clean texts - remove excessive whitespace and empty strings
    clean_texts = [re.sub(r'\s+', ' ', text.strip()) for text in texts if text.strip()]
    
    if not clean_texts:
        raise ValueError("No valid texts provided for embedding")

    # Cohere's embed-english-v3.0 supports up to 96 texts per request
    BATCH_SIZE = 90  # Slightly conservative to avoid rate limits
    batches = [clean_texts[i:i+BATCH_SIZE] for i in range(0, len(clean_texts), BATCH_SIZE)]

    async def fetch_batch(batch, batch_num):
        data = {
            "model": "embed-english-v3.0",  # Latest Cohere v3 model
            "texts": batch,
            "input_type": input_type  # "search_document" for docs, "search_query" for queries
            # Removed embedding_types as it's causing the issue
        }
        
        # Add retry logic
        max_retries = 3
        for attempt in range(max_retries):
            try:
                timeout = httpx.Timeout(60.0)  # Longer timeout for embedding requests
                async with httpx.AsyncClient(timeout=timeout) as client:
                    print(f"[get_embeddings] Sending request to Cohere: batch {batch_num}, size {len(batch)}, attempt {attempt + 1}")
                    response = await client.post(url, headers=headers, json=data)
                    
                    if response.status_code == 200:
                        response_data = response.json()
                        print(f"[get_embeddings] Success! Response keys: {list(response_data.keys())}")
                        
                        batch_embeddings = []
                        # Handle different response formats
                        if "embeddings" in response_data:
                            embeddings_data = response_data["embeddings"]
                        else:
                            # Fallback for different API response structure
                            embeddings_data = response_data.get("data", [])
                            
                        for i, embedding in enumerate(embeddings_data):
                            try:
                                # Handle different embedding formats
                                if isinstance(embedding, dict):
                                    # If embedding is wrapped in a dict (like {"embedding": [...])
                                    vec_data = embedding.get("embedding", embedding)
                                else:
                                    # Direct list of numbers
                                    vec_data = embedding
                                
                                vec = np.array(vec_data, dtype="float32")
                                # Normalize for cosine similarity
                                vec = vec / np.linalg.norm(vec)
                                batch_embeddings.append(vec.tolist())
                            except Exception as embed_error:
                                print(f"[get_embeddings] Error processing embedding {i}: {str(embed_error)}")
                                print(f"[get_embeddings] Embedding type: {type(embedding)}")
                                print(f"[get_embeddings] Embedding sample: {str(embedding)[:200]}...")
                                raise embed_error
                                
                        return batch_embeddings
                    elif response.status_code == 429:  # Rate limit
                        wait_time = 2 ** attempt
                        print(f"[get_embeddings] Rate limited, waiting {wait_time}s...")
                        await asyncio.sleep(wait_time)
                        continue
                    else:
                        print(f"[get_embeddings] Response status: {response.status_code}")
                        print(f"[get_embeddings] Response content: {response.text}")
                        response.raise_for_status()
                        
            except Exception as e:
                print(f"[get_embeddings] Exception on attempt {attempt + 1}: {str(e)}")
                if attempt == max_retries - 1:
                    raise e
                await asyncio.sleep(1)
        
        raise Exception("Failed to get embeddings after all retries")

    # Process batches with some delay to avoid rate limits
    all_results = []
    for i, batch in enumerate(batches):
        if i > 0:
            await asyncio.sleep(0.1)  # Small delay between batch requests
        result = await fetch_batch(batch, i + 1)
        all_results.append(result)
    
    return [emb for batch in all_results for emb in batch]

# ---- Enhanced FAISS Helper Function ----
def create_faiss_index(embeddings_list):
    """
    Create optimized FAISS index with better performance.
    """
    embeddings_array = np.array(embeddings_list).astype("float32")
    dim = embeddings_array.shape[1]
    
    # Use IndexHNSWFlat for better search quality with reasonable speed
    # This provides better recall than IndexFlatL2
    index = faiss.IndexHNSWFlat(dim, 32)  # 32 is the number of connections per element
    index.hnsw.efConstruction = 200  # Higher value = better quality, slower build
    index.hnsw.efSearch = 64  # Higher value = better search quality
    
    index.add(embeddings_array)
    return index

# ---- Enhanced FAISS Indexing ----
async def build_faiss_index(chunks: List[str]):
    embeddings = await get_embeddings(chunks, input_type="search_document")
    
    # Use process pool for CPU-intensive FAISS index creation
    loop = asyncio.get_event_loop()
    index = await loop.run_in_executor(cpu_executor, create_faiss_index, embeddings)
    return index, chunks

async def search_faiss(index, query_embedding: List[float], chunks: List[str], k=20) -> List[str]:
    """Enhanced search with better retrieval"""
    query_vec = np.array([query_embedding]).astype("float32")
    D, I = index.search(query_vec, min(k, len(chunks)))

    # Get retrieved chunks with distances
    retrieved = []
    for j, i in enumerate(I[0]):
        if i != -1:  # Valid index
            retrieved.append((chunks[i], float(D[0][j])))
    
    # Sort by distance (lower is better for L2 distance)
    retrieved.sort(key=lambda x: x[1])
    
    # Return top chunks (increased to 10 for better context)
    return [chunk for chunk, _ in retrieved[:10]]

# ---- Enhanced Prompt Builder ----
def build_prompt(question: str, context_chunks: List[str]) -> str:
    # Deduplicate similar chunks to avoid redundancy
    unique_chunks = []
    seen_chunks = []
    
    for chunk in context_chunks:
        chunk_words = set(chunk.lower().split())
        is_duplicate = False
        
        for seen_words_set in seen_chunks:
            # Calculate Jaccard similarity
            intersection_size = len(chunk_words.intersection(seen_words_set))
            union_size = len(chunk_words.union(seen_words_set))
            if union_size > 0 and intersection_size / union_size > 0.8:
                is_duplicate = True
                break
                
        if not is_duplicate:
            unique_chunks.append(chunk)
            seen_chunks.append(chunk_words)
    
    context = "\n\n---\n\n".join(unique_chunks)
    
    return f"""You are an expert insurance policy analyst with deep expertise in policy interpretation and claims analysis. Your role is to provide precise, actionable answers based strictly on the provided policy context.

**CORE DIRECTIVES:**
1. Answer in ONE concise, direct paragraph (25-40 words maximum)
2.Only start your response with "Yes" or "No" if the user's question is a direct yes/no question. Otherwise, respond naturally.
3. Include ALL relevant numbers, percentages, limits, and conditions from the context
4. If the context contains specific numbers, percentages, sub-limits, caps, or exceptions (such as for PPN or listed procedures), you MUST always include them in your answer. If there are exceptions or special cases, mention them explicitly. Quote the exact numbers and conditions from the context. Do not generalize or omit details.
5. If the answer isn't in the context OR the question is unrelated to insurance, respond EXACTLY: "The information is not available in the provided context."
6. Use natural, professional language without markdown or bullet points

**ANALYSIS PRIORITIES:**
- Coverage limits, sub-limits, and caps
- Exclusions and exceptions  
- Waiting periods and conditions
- Specific percentages and amounts
- Policy definitions and terms

**CONTEXT:**
{context}

**QUESTION:** {question}

**ANSWER:**"""

# ---- Enhanced LLM Call with Gemini 2.0 Flash ----
def ask_llm_sync(prompt: str) -> str:
    try:
        client = genai.Client(api_key=GEMINI_API_KEY)
        from google.genai import types
        
        # Use the latest Gemini 2.0 Flash model for best performance
        response = client.models.generate_content(
            model="gemini-2.5-flash-lite", 
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.0,  # Deterministic for factual accuracy
                max_output_tokens=150,  # Concise responses
                top_p=0.8,  # Focused but not too restrictive
                top_k=20,   # Limit vocabulary for precision
                candidate_count=1,
                stop_sequences=None
            )
        )
        
        # Extract response text
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
        print(f"[ask_llm_sync] Error: {str(e)}")
        # Fallback to a more stable model if experimental fails
        try:
            response = client.models.generate_content(
                model="gemini-1.5-pro",  # Fallback model
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

# Async wrapper for LLM calls
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
        # Check cache first
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

        # Get embeddings for all questions at once (with query input type)
        question_embeddings = await get_embeddings(request.questions, input_type="search_query")
        
        # Process questions concurrently with improved error handling
        async def process_question(question, q_emb):
            try:
                relevant_chunks = await search_faiss(index, q_emb, chunk_list)
                if not relevant_chunks:
                    return "The information is not available in the provided context."
                    
                prompt = build_prompt(question, relevant_chunks)
                response = await ask_llm(prompt)
                
                # Clean up response
                response = response.strip()
                
                # Remove any JSON formatting artifacts
                if response.startswith('{"') or response.startswith('{'):
                    try:
                        parsed = json.loads(response)
                        if isinstance(parsed, dict) and 'answer' in parsed:
                            response = parsed['answer']
                    except:
                        pass
                
                # Remove common prefixes/suffixes
                response = re.sub(r'^(Answer:|ANSWER:|Response:)\s*', '', response, flags=re.IGNORECASE)
                response = re.sub(r'\s*\.$', '', response)  # Remove trailing period if it's the only punctuation
                
                return response.strip()
                
            except Exception as e:
                print(f"Error processing question '{question}': {str(e)}")
                return "Error processing the question. Please try again."
        
        # Execute all questions concurrently
        tasks = [process_question(q, emb) for q, emb in zip(request.questions, question_embeddings)]
        answers = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle any exceptions in the results
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

# ---- Health check endpoint ----
@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_info": {"embedding": "cohere-v3", "llm": "gemini-2.0-flash-exp"}}

# ---- Cleanup function for executors ----
@app.on_event("shutdown")
async def shutdown_executors():
    io_executor.shutdown(wait=True)
    cpu_executor.shutdown(wait=True)