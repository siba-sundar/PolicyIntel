from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel
import httpx
import fitz
from groq import Groq
import os
from dotenv import load_dotenv
from typing import List
import numpy as np
import faiss
import json

# ---- Load API Keys ----
load_dotenv()
groq_client = Groq()
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")

app = FastAPI()
faiss_cache = {}

# ---- Auth Token ----
TEAM_TOKEN = "hackrx2025teamtoken"

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
        doc = fitz.open("temp.pdf")
        text = ""
        for page in doc:
            text += page.get_text()
        doc.close()
        return text.strip()
    except Exception as e:
        raise Exception(f"Download or extraction failed: {str(e)}")

# ---- Chunking ----
def split_text_into_chunks(text: str, chunk_size: int = 300, overlap: int = 50) -> List[str]:
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = words[i:i + chunk_size]
        chunks.append(" ".join(chunk))
        i += chunk_size - overlap if i + chunk_size < len(words) else len(words)
    return chunks

# ---- Embeddings ----
def get_embeddings(texts: List[str]) -> List[List[float]]:
    url = "https://api.mistral.ai/v1/embeddings"
    headers = {
        "Authorization": f"Bearer {MISTRAL_API_KEY}",
        "Content-Type": "application/json"
    }
    # Validate input: must be a list of strings
    if not isinstance(texts, list) or not all(isinstance(t, str) for t in texts):
        print("[get_embeddings] ERROR: input must be a list of strings.")
        raise ValueError("Input to get_embeddings must be a list of strings.")

    BATCH_SIZE = 32  # Mistral API batch limit
    all_embeddings = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i:i+BATCH_SIZE]
        data = {
            "model": "mistral-embed",
            "input": batch
        }
        print(f"[get_embeddings] Sending request to Mistral: batch {i//BATCH_SIZE+1}, size {len(batch)}")
        response = httpx.post(url, headers=headers, json=data)
        print(f"[get_embeddings] Response status: {response.status_code}")
        if response.status_code != 200:
            print(f"[get_embeddings] Response content: {response.text}")
            response.raise_for_status()
        batch_embeddings = []
        for e in response.json()["data"]:
            vec = np.array(e["embedding"], dtype="float32")
            vec /= np.linalg.norm(vec)
            batch_embeddings.append(vec)
        all_embeddings.extend(batch_embeddings)
    return all_embeddings

# ---- FAISS Indexing ----
def build_faiss_index(chunks: List[str]):
    embeddings = get_embeddings(chunks)
    dim = len(embeddings[0])
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings).astype("float32"))
    return index, chunks

def search_faiss(index, query: str, chunks: List[str], k=10) -> List[str]:
    query_embedding = get_embeddings([query])[0]
    D, I = index.search(np.array([query_embedding]).astype("float32"), k)

    retrieved = [(chunks[i], float(D[0][j])) for j, i in enumerate(I[0])]
    # Sort by distance (lower is better)
    reranked = sorted(retrieved, key=lambda x: x[1])
    return [chunk for chunk, _ in reranked[:5]]  # top 5 after re-ranking

# ---- Prompt Builder ----
def build_prompt(question: str, context_chunks: List[str]) -> str:
    context = "\n\n".join(context_chunks)
    return f"""
You are an expert insurance policy analyst and answering engine. Your primary directive is to answer questions about insurance policies with extreme precision, using ONLY the information available in the provided CONTEXT.

**CRITICAL RULES:**
1. Your answer MUST be a single, concise paragraph that directly addresses the question.
2. If the user's question is not related to the insurance policy in the CONTEXT (e.g., asking for code, general knowledge, unrelated topics), you MUST respond with the exact phrase: "The information is not available in the provided context."
3. If the answer to a relevant question is not explicitly stated in the CONTEXT, you MUST also respond with the exact phrase: "The information is not available in the provided context."
4. Do not use markdown, lists, or conversational phrases.
5. For insurance-specific questions, be precise with numbers, percentages, time periods, and policy terms.
6. If calculations are needed, show your reasoning clearly.
7. Use exact policy language when available in the context.

**INSURANCE POLICY ANALYSIS GUIDELINES:**
- Pay special attention to coverage limits, exclusions, waiting periods, and conditions
- Look for specific terms like "Sum Insured", "Premium", "Waiting Period", "Exclusions", "Coverage"
- For medical procedures, check for specific coverage details and sub-limits
- For calculations, use exact percentages and amounts mentioned in the policy
- If a question asks about coverage for a specific condition or procedure, search for relevant clauses

== CONTEXT ==
{context}

== QUESTION ==
{question}

== ANSWER ==
{{"answer": "<your short, clear answer here>"}}
"""

# ---- LLM Call ----
def ask_llm(prompt: str) -> str:
    try:
        completion = groq_client.chat.completions.create(
            model="moonshotai/kimi-k2-instruct",
            messages=[
                {"role": "system", "content": prompt}
            ],
            temperature=0.2,
            stream=False,
            stop=None,
        )
        return completion.choices[0].message.content.strip()
    except Exception as e:
        return json.dumps({"answer": f"Error generating answer: {str(e)}"})

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
        index, chunk_list = build_faiss_index(chunks)
        faiss_cache[request.documents] = (index, chunk_list)

    try:
        answers = []
        for question in request.questions:
            relevant_chunks = search_faiss(index, question, chunk_list)
            prompt = build_prompt(question, relevant_chunks)
            llm_output = ask_llm(prompt)

            # ✅ Extract actual value from JSON response
            try:
                parsed = json.loads(llm_output)
                answers.append(parsed.get("answer", llm_output))
            except json.JSONDecodeError:
                answers.append(llm_output)

        return {"answers": answers}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
