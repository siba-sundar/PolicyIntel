import json
import re
import nltk
from typing import Dict, List
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFacePipeline
from transformers import pipeline
import PyPDF2
from docx import Document

nltk.download('punkt')

# Set up LLM
llm_pipeline = pipeline("text-generation", model="gpt2", max_new_tokens=300, temperature=0.7)
llm = HuggingFacePipeline(pipeline=llm_pipeline)

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Extract PDF text
def extract_text_from_pdf(file_path: str) -> str:
    with open(file_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        return "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])

# Extract DOCX
def extract_text_from_docx(file_path: str) -> str:
    doc = Document(file_path)
    return "\n".join([p.text for p in doc.paragraphs])

# Handle plain text/email
def extract_text_from_email(content: str) -> str:
    return content

# Process a document
def process_document(path: str, file_type: str) -> List[str]:
    if file_type == "pdf":
        text = extract_text_from_pdf(path)
    elif file_type == "docx":
        text = extract_text_from_docx(path)
    elif file_type == "email":
        text = extract_text_from_email(path)
    else:
        raise ValueError("Unsupported file type")

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return splitter.split_text(text)

# Parse query with LLM
def parse_query(query: str) -> Dict:
    prompt = f"""
    Extract details from this query:
    "{query}"
    Format: {{
        "age": int or null,
        "gender": "male"/"female"/null,
        "procedure": str or null,
        "location": str or null,
        "policy_duration": str or null
    }}
    """
    response = llm(prompt)
    try:
        return json.loads(response.split("```json")[1].split("```")[0]) if "```json" in response else json.loads(response)
    except:
        fallback = {"age": None, "gender": None, "procedure": None, "location": None, "policy_duration": None}
        if "male" in query.lower():
            fallback["gender"] = "male"
        if "female" in query.lower():
            fallback["gender"] = "female"
        if match := re.search(r"\d+", query):
            fallback["age"] = int(match.group())
        if "surgery" in query.lower():
            fallback["procedure"] = query.lower().split("surgery")[0].strip() + " surgery"
        if "month" in query.lower():
            fallback["policy_duration"] = query.lower().split("policy")[0].strip()
        for city in ["Pune", "Mumbai", "Delhi"]:
            if city.lower() in query.lower():
                fallback["location"] = city
        return fallback

# Search relevant clauses
def retrieve_clauses(query: str, docs: List[str]) -> List[Dict]:
    store = FAISS.from_texts(docs, embeddings)
    results = store.similarity_search(query, k=3)
    return [{"clause": r.page_content, "score": 0.9} for r in results]

# Make decision
def evaluate_decision(parsed_query: Dict, clauses: List[Dict]) -> Dict:
    prompt = f"""
    User query details:
    {json.dumps(parsed_query, indent=2)}

    Found clauses:
    {json.dumps(clauses, indent=2)}

    Decide:
    {{
        "decision": "approved/rejected",
        "amount": int (if applicable),
        "justification": str,
        "clause_references": [list of clause snippets]
    }}
    """
    response = llm(prompt)
    try:
        return json.loads(response.split("```json")[1].split("```")[0]) if "```json" in response else json.loads(response)
    except:
        return {
            "decision": "rejected",
            "amount": 0,
            "justification": "LLM failed to parse clauses accurately.",
            "clause_references": [c["clause"] for c in clauses]
        }

# Orchestrate
def process_query_and_documents(query: str, docs: List[Dict]) -> Dict:
    all_chunks = []
    for d in docs:
        all_chunks += process_document(d["path"], d["type"])
    parsed = parse_query(query)
    clauses = retrieve_clauses(query, all_chunks)
    return evaluate_decision(parsed, clauses)

# Run it
if __name__ == "__main__":
    query = "46M, knee surgery, Pune, 3-month policy"
    documents = [{"path": "p-1.pdf", "type": "pdf"}]
    result = process_query_and_documents(query, documents)
    print(json.dumps(result, indent=2))
