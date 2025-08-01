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

# ---- Load API Keys ----
load_dotenv()
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

app = FastAPI()
faiss_cache = {}

# ---- Thread Pool Executors ----
io_executor = ThreadPoolExecutor(max_workers=4)  # For IO-bound tasks
cpu_executor = ProcessPoolExecutor(max_workers=2)  # For CPU-bound tasks

# ---- Auth Token ----
TEAM_TOKEN = "833695cad1c0d2600066bf2b08aab7614d0dec93b4b6f0ae3acd37ef7d6fcb1c"

# ---- Data Models ----
class QueryRequest(BaseModel):
    documents: str
    questions: List[str]

class QueryResponse(BaseModel):
    answers: List[str]

# ---- PDF Downloader ----
async def download_pdf_text(url: str) -> str:
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(url)
        response.raise_for_status()
        with open("temp.pdf", "wb") as f:
            f.write(response.content)

        # Use thread pool for IO-bound PDF opening and text extraction
        def extract_text():
            with fitz.open("temp.pdf") as doc:
                text = ""
                for page in doc:
                    text += page.get_text()
                return text
        
        loop = asyncio.get_event_loop()
        text = await loop.run_in_executor(io_executor, extract_text)
        return text.strip()
    except Exception as e:
        raise Exception(f"Download or extraction failed: {str(e)}")

# ---- Chunking ----
def split_text_into_chunks(text: str, chunk_size: int = 500, overlap: int = 100) -> List[str]:
    # Split by sentences first to maintain semantic coherence
    import re
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    chunks = []
    current_chunk = []
    current_length = 0
    
    for sentence in sentences:
        words_in_sentence = len(sentence.split())
        
        # If adding this sentence would exceed chunk_size, finalize current chunk
        if current_length + words_in_sentence > chunk_size and current_chunk:
            chunks.append(" ".join(current_chunk))
            
            # Start new chunk with overlap from previous chunk
            overlap_words = " ".join(current_chunk).split()[-overlap:]
            current_chunk = overlap_words + [sentence]
            current_length = len(overlap_words) + words_in_sentence
        else:
            current_chunk.append(sentence)
            current_length += words_in_sentence
    
    # Add the last chunk
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    
    return chunks

# ---- Embeddings ----
async def get_embeddings(texts: List[str]) -> List[List[float]]:
    url = "https://api.mistral.ai/v1/embeddings"
    headers = {
        "Authorization": f"Bearer {MISTRAL_API_KEY}",
        "Content-Type": "application/json"
    }
    if not isinstance(texts, list) or not all(isinstance(t, str) for t in texts):
        print("[get_embeddings] ERROR: input must be a list of strings.")
        raise ValueError("Input to get_embeddings must be a list of strings.")

    BATCH_SIZE = 32
    batches = [texts[i:i+BATCH_SIZE] for i in range(0, len(texts), BATCH_SIZE)]

    async def fetch_batch(batch, batch_num):
        data = {
            "model": "mistral-embed",
            "input": batch
        }
        async with httpx.AsyncClient() as client:
            print(f"[get_embeddings] Sending request to Mistral: batch {batch_num}, size {len(batch)}")
            response = await client.post(url, headers=headers, json=data)
            print(f"[get_embeddings] Response status: {response.status_code}")
            if response.status_code != 200:
                print(f"[get_embeddings] Response content: {response.text}")
                response.raise_for_status()
            batch_embeddings = []
            for e in response.json()["data"]:
                vec = np.array(e["embedding"], dtype="float32")
                vec /= np.linalg.norm(vec)
                batch_embeddings.append(vec)
            return batch_embeddings

    tasks = [fetch_batch(batch, idx+1) for idx, batch in enumerate(batches)]
    all_results = await asyncio.gather(*tasks)
    return [emb for batch in all_results for emb in batch]

# ---- FAISS Helper Function (Module Level) ----
def create_faiss_index(embeddings_list):
    """
    Module-level function for FAISS index creation.
    This can be pickled and sent to ProcessPoolExecutor.
    """
    embeddings_array = np.array(embeddings_list).astype("float32")
    dim = embeddings_array.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings_array)
    return index

# ---- FAISS Indexing ----
async def build_faiss_index(chunks: List[str]):
    embeddings = await get_embeddings(chunks)
    
    # Use process pool for CPU-intensive FAISS index creation
    loop = asyncio.get_event_loop()
    index = await loop.run_in_executor(cpu_executor, create_faiss_index, embeddings)
    return index, chunks

async def search_faiss(index, query: str, chunks: List[str], k=15) -> List[str]:
    # This function will now accept a precomputed embedding instead of a query string
    D, I = index.search(np.array([query]).astype("float32"), k)

    retrieved = [(chunks[i], float(D[0][j])) for j, i in enumerate(I[0])]
    # Sort by distance (lower is better) and return top 8 for better context
    reranked = sorted(retrieved, key=lambda x: x[1])
    return [chunk for chunk, _ in reranked[:8]]

# ---- Prompt Builder ----
def build_prompt(question: str, context_chunks: List[str]) -> str:
    context = "\n\n".join(context_chunks)
    return f"""
You are an expert insurance policy analyst and answering engine. Your primary directive is to answer questions about insurance policies with extreme precision, using ONLY the information available in the provided CONTEXT.

**CRITICAL RULES:**
1. Your answer MUST be a single, concise paragraph that directly addresses the question. The answer must be to the point, and should not include any unnecessary details.
2. If the user's question is not related to the insurance policy in the CONTEXT or answer to a relevant question is not explicitly stated in the CONTEXT (e.g., asking for code, general knowledge, unrelated topics), you MUST respond with the exact phrase: "The information is not available in the provided context."
3. Do not use markdown, lists, or conversational phrases.
4. For insurance-specific questions, be precise with numbers, percentages, time periods, and policy terms.
5. If calculations are needed, show your reasoning clearly.
6. Use exact policy language when available in the context.
7. If the context contains specific numbers, percentages, sub-limits, caps, or exceptions (such as for PPN or listed procedures), you MUST always include them in your answer. If there are exceptions or special cases, mention them explicitly. Quote the exact numbers and conditions from the context. Do not generalize or omit details.
8. Start directly with "Yes" or "No" when appropriate, followed by specific details
9. Keep responses comprehensive but focused (aim for 30-40 words typically) no more than that.
10. Write in a natural, conversational tone as if explaining to a colleague or friend


**INSURANCE POLICY ANALYSIS GUIDELINES:**
- Pay special attention to coverage limits, exclusions, waiting periods, and conditions
- Look for specific terms like "Sum Insured", "Premium", "Waiting Period", "Exclusions", "Coverage"
- For medical procedures, check for specific coverage details and sub-limits
- For calculations, use exact percentages and amounts mentioned in the policy
- If a question asks about coverage for a specific condition or procedure, search for relevant clauses
- When describing definitions or terms, be complete but concise

== CONTEXT ==
{context}

== QUESTION ==
{question}

== ANSWER ==
{{"answer": "<your short, clear answer here>"}}
"""

# ---- LLM Call ----
def ask_llm_sync(prompt: str) -> str:
    try:
        # Use the official Gemini 2.5 Flash Lite model
        client = genai.Client(api_key=GEMINI_API_KEY)
        from google.genai import types
        response = client.models.generate_content(
            model="gemini-2.5-flash-lite",
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.1,  # Slightly higher for more natural language
                max_output_tokens=200,  # Allow for longer responses
                top_p=0.9,
                top_k=40
            )
        )
        # Gemini returns .text for the main output
        if hasattr(response, 'text') and response.text:
            return response.text.strip()
        # Fallback for structured output
        elif hasattr(response, 'candidates') and response.candidates:
            return response.candidates[0].content.parts[0].text.strip()
        else:
            return "No response from Gemini AI."
    except Exception as e:
        return f"Error generating answer: {str(e)}"

# Async wrapper for LLM calls using thread pool
async def ask_llm(prompt: str) -> str:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(io_executor, ask_llm_sync, prompt)

# ---- API Endpoint ----
@app.post("/hackrx/run", response_model=QueryResponse)
async def run_query(request: QueryRequest, authorization: str = Header(None)):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Authorization header missing or malformed.")

    token = authorization.split(" ")[1]
    if token != TEAM_TOKEN:
        raise HTTPException(status_code=403, detail="Invalid token")

    if request.documents in faiss_cache:
        index, chunk_list = faiss_cache[request.documents]
    else:
        document_text = await download_pdf_text(request.documents)
        chunks = split_text_into_chunks(document_text)
        index, chunk_list = await build_faiss_index(chunks)
        faiss_cache[request.documents] = (index, chunk_list)

    try:
        # Batch embed all questions at once
        question_embeddings = await get_embeddings(request.questions)
        
        # Process all questions concurrently
        async def process_question(question, q_emb):
            relevant_chunks = await search_faiss(index, q_emb, chunk_list)
            prompt = build_prompt(question, relevant_chunks)
            return await ask_llm(prompt)
        
        # Create tasks for concurrent processing
        tasks = [process_question(question, q_emb) for question, q_emb in zip(request.questions, question_embeddings)]
        llm_outputs = await asyncio.gather(*tasks)
        
        answers = []
        for llm_output in llm_outputs:
            # Clean up the response and extract the answer
            answer = llm_output.strip()
            
            # Remove any JSON formatting if present
            if answer.startswith('{"answer":') and answer.endswith('}'):
                try:
                    parsed = json.loads(answer)
                    answer = parsed.get("answer", answer)
                except json.JSONDecodeError:
                    pass
            
            # Remove any residual formatting
            answer = answer.replace('**ANSWER:**', '').strip()
            answers.append(answer)

        return {"answers": answers}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ---- Cleanup function for executors ----
@app.on_event("shutdown")
async def shutdown_executors():
    io_executor.shutdown(wait=True)
    cpu_executor.shutdown(wait=True)