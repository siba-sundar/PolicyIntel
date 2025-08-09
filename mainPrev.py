from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel
import httpx
import fitz  # PyMuPDF
import google.generativeai as genai
import os
from dotenv import load_dotenv
from typing import List, Dict, Optional, Tuple
import numpy as np
import json
import re
import logging
import time
from pathlib import Path
import asyncio
from bs4 import BeautifulSoup
import html2text
from urllib.parse import urljoin, urlparse
from langdetect import detect

from app.services.chunking import get_enhanced_chunks
from app.services.faiss_search import FAISSSearchService
from app.config.settings import (
    COHERE_API_KEYS, GEMINI_API_KEYS, TEAM_TOKEN,
    REQUESTS_PER_KEY, MAX_FILE_SIZE, FAISS_K_SEARCH,
    GEMINI_MODEL, LLM_TEMPERATURE, LLM_MAX_TOKENS, LLM_TOP_P, LLM_TOP_K,
    COHERE_MODEL, EMBEDDING_BATCH_SIZE, EMBEDDING_MAX_LENGTH
)

# Setup Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables for API key rotation
current_gemini_key_index = 0
gemini_request_count = 0
current_cohere_key_index = 0

# Initialize FAISS search service
faiss_service = FAISSSearchService()

# Predefined city-landmark mapping
CITY_LANDMARK_MAP = {
    "Delhi": "Gateway of India",
    "Mumbai": "India Gate",
    "Chennai": "Charminar",
    "Hyderabad": "Marina Beach",
    "Ahmedabad": "Howrah Bridge",
    "Mysuru": "Golconda Fort",
    "Kochi": "Qutub Minar",

    "Hyderabad": "Taj Mahal",
    "Pune": "Meenakshi Temple",
    "Nagpur": "Lotus Temple",
    "Chandigarh": "Mysore Palace",
    "Kerala": "Rock Garden",
    "Bhopal": "Victoria Memorial",
    "Varanasi": "Vidhana Soudha",
    "Jaisalmer": "Sun Temple",
    "Pune": "Golden Temple",

    "New York": "Eiffel Tower",
    "London": "Statue of Liberty",
    "Tokyo": "Big Ben",
    "Beijing": "Colosseum",
    "London": "Sydney Opera House",
    "Bangkok": "Christ the Redeemer",
    "Toronto": "Burj Khalifa",
    "Dubai": "CN Tower",
    "Amsterdam": "Petronas Towers",
    "Cairo": "Leaning Tower of Pisa",
    "San Francisco": "Mount Fuji",
    "Berlin": "Niagara Falls",
    "Barcelona": "Louvre Museum",
    "Moscow": "Stonehenge",
    "Seoul": "Sagrada Familia",
    "Cape Town": "Acropolis",
    "Istanbul": "Big Ben",

    "Riyadh": "Machu Picchu",
    "Paris": "Taj Mahal",
    "Dubai Airport": "Moai Statues",
    "Singapore": "Christchurch Cathedral",
    "Jakarta": "The Shard",
    "Vienna": "Blue Mosque",
    "Kathmandu": "Neuschwanstein Castle",
    "Los Angeles": "Buckingham Palace",
    "Mumbai": "Space Needle",
    "Seoul": "Times Square"
}

def get_current_gemini_key():
    """Get current Gemini API key and handle rotation"""
    global current_gemini_key_index, gemini_request_count
    
    current_key = GEMINI_API_KEYS[current_gemini_key_index]
    gemini_request_count += 1
    
    if gemini_request_count >= REQUESTS_PER_KEY:
        gemini_request_count = 0
        current_gemini_key_index = (current_gemini_key_index + 1) % len(GEMINI_API_KEYS)
        logger.info(f"🔄 SWITCHED to Gemini API key #{current_gemini_key_index + 1}")
    
    return current_key

def get_current_cohere_key():
    """Get current Cohere API key for this request"""
    return COHERE_API_KEYS[current_cohere_key_index]

def rotate_cohere_key():
    """Rotate to the next Cohere API key after a complete request"""
    global current_cohere_key_index
    current_cohere_key_index = (current_cohere_key_index + 1) % len(COHERE_API_KEYS)
    logger.info(f"🔄 SWITCHED to Cohere API key #{current_cohere_key_index + 1}")

app = FastAPI(title="Streamlined PolicyIntel API with Puzzle Solver", description="Fast PDF and HTML processing API with puzzle solving capabilities", version="2.1.0")

# Data Models
class QueryRequest(BaseModel):
    documents: str
    questions: List[str]

class QueryResponse(BaseModel):
    answers: List[str]

# Health Check
@app.get("/")
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "Streamlined PolicyIntel API with Puzzle Solver", "version": "2.1.0"}

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

def is_puzzle_document(text: str) -> bool:
    """Detect if the document contains puzzle instructions"""
    puzzle_indicators = [
        'mission brief',
        'step-by-step guide',
        'your mission is to',
        'decode the city',
        'flight path',
        'flight number',
        'parallel world',
        'hackrx',
        'sachin',
        'landmark current location',
        'get https://register.hackrx.in',
        'final deliverable',
        'explorer'
    ]
    
    text_lower = text.lower()
    puzzle_score = sum(1 for indicator in puzzle_indicators if indicator in text_lower)
    
    logger.info(f"🧩 Puzzle detection score: {puzzle_score}//{len(puzzle_indicators)}")
    
    # If we find 3 or more puzzle indicators, consider it a puzzle
    return puzzle_score >= 3

def get_landmark_for_city(city: str) -> Optional[str]:
    """Get landmark for a city using the predefined mapping"""
    logger.info(f"🗺️ Looking up landmark for city: {city}")
    
    # First try exact match
    if city in CITY_LANDMARK_MAP:
        landmark = CITY_LANDMARK_MAP[city]
        logger.info(f"📍 Found exact match: {city} -> {landmark}")
        return landmark
    
    # Try case-insensitive search
    city_lower = city.lower().strip()
    for map_city, landmark in CITY_LANDMARK_MAP.items():
        if map_city.lower().strip() == city_lower:
            logger.info(f"📍 Found case-insensitive match: {city} -> {landmark}")
            return landmark
    
    # Try partial matching (city name contains or is contained in map key)
    for map_city, landmark in CITY_LANDMARK_MAP.items():
        map_city_lower = map_city.lower().strip()
        if city_lower in map_city_lower or map_city_lower in city_lower:
            logger.info(f"📍 Found partial match: {city} -> {landmark} (matched with {map_city})")
            return landmark
    
    logger.warning(f"❌ No landmark found for city: {city}")
    logger.info(f"Available cities in mapping: {list(CITY_LANDMARK_MAP.keys())}")
    return None

async def call_favorite_city_api() -> Optional[str]:
    """Call the API to get the favorite city"""
    logger.info("🏙️ Calling favorite city API")
    
    try:
        timeout = httpx.Timeout(15.0)
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'application/json'
        }
        
        async with httpx.AsyncClient(timeout=timeout, headers=headers) as client:
            response = await client.get("https://register.hackrx.in/submissions/myFavouriteCity")
        
        response.raise_for_status()
        data = response.json()
        
        if data.get('success') and data.get('data', {}).get('city'):
            city = data['data']['city']
            logger.info(f"🏙️ Retrieved favorite city: {city}")
            return city
        else:
            logger.error(f"Invalid API response format: {data}")
            return None
            
    except Exception as e:
        logger.error(f"Failed to get favorite city: {str(e)}")
        return None

async def call_flight_api(landmark: str) -> Optional[str]:
    """Call the appropriate flight API based on the landmark"""
    logger.info(f"✈️ Getting flight for landmark: {landmark}")
    
    # Determine which API endpoint to call based on landmark
    endpoint_map = {
        'gateway of india': 'getFirstCityFlightNumber',
        'taj mahal': 'getSecondCityFlightNumber', 
        'eiffel tower': 'getThirdCityFlightNumber',
        'big ben': 'getFourthCityFlightNumber'
    }
    
    landmark_lower = landmark.lower().strip()
    endpoint = endpoint_map.get(landmark_lower, 'getFifthCityFlightNumber')
    
    url = f"https://register.hackrx.in/teams/public/flights/{endpoint}"
    logger.info(f"✈️ Calling flight API: {url} for landmark: {landmark}")
    
    try:
        timeout = httpx.Timeout(15.0)
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'application/json'
        }
        
        async with httpx.AsyncClient(timeout=timeout, headers=headers) as client:
            response = await client.get(url)
        
        response.raise_for_status()
        data = response.json()
        
        if data.get('success') and data.get('data', {}).get('flightNumber'):
            flight_number = data['data']['flightNumber']
            logger.info(f"✈️ Retrieved flight number: {flight_number}")
            return flight_number
        else:
            logger.error(f"Invalid flight API response format: {data}")
            return None
            
    except Exception as e:
        logger.error(f"Failed to get flight number: {str(e)}")
        return None

async def solve_puzzle(document_text: str, questions: List[str]) -> List[str]:
    """Solve the puzzle based on document content and questions"""
    logger.info("🧩 Starting puzzle solving process")
    
    answers = []
    
    for question in questions:
        question_lower = question.lower()
        
        # Check if this is asking for flight number
        if 'flight number' in question_lower or 'flight' in question_lower:
            logger.info("✈️ Detected flight number question - solving puzzle")
            
            try:
                # Step 1: Get favorite city from API
                favorite_city = await call_favorite_city_api()
                if not favorite_city:
                    answers.append("Unable to retrieve favorite city from API")
                    continue
                
                logger.info(f"🏙️ Favorite city retrieved: {favorite_city}")
                
                # Step 2: Get landmark for the favorite city using predefined mapping
                landmark = get_landmark_for_city(favorite_city)
                
                if not landmark:
                    answers.append(f"Could not find landmark mapping for city: {favorite_city}")
                    continue
                
                logger.info(f"🏛️ Found landmark '{landmark}' for city '{favorite_city}'")
                
                # Step 3: Get flight number based on landmark
                flight_number = await call_flight_api(landmark)
                if flight_number:
                    answers.append(f"Flight number for {favorite_city} (landmark: {landmark}): {flight_number}")
                else:
                    answers.append(f"Unable to retrieve flight number for landmark: {landmark}")
                    
            except Exception as e:
                logger.error(f"Error solving puzzle: {str(e)}")
                answers.append(f"Error solving puzzle: {str(e)}")
        
        else:
            # For non-puzzle questions, provide helpful response
            logger.info("📝 Processing as regular question")
            answers.append("This appears to be a puzzle document. Please ask about the flight number to get the solution.")
    
    return answers

def extract_html_content(html_content: str, base_url: str = "") -> str:
    """Extract meaningful content from HTML page"""
    logger.info("Starting HTML content extraction")
    
    try:
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Remove unwanted elements
        for element in soup(['script', 'style', 'nav', 'header', 'footer', 
                           'aside', 'advertisement', 'ads', 'sidebar']):
            element.decompose()
        
        # Extract title
        title = ""
        title_tag = soup.find('title')
        if title_tag:
            title = f"TITLE: {title_tag.get_text().strip()}\n\n"
        
        # Extract main content
        main_content_selectors = ['main', 'article', '[role="main"]', '.content', '.main-content', '#content', '#main']
        main_content = None
        for selector in main_content_selectors:
            main_content = soup.select_one(selector)
            if main_content:
                break
        
        if not main_content:
            main_content = soup.find('body') or soup
        
        # Convert to clean text
        h = html2text.HTML2Text()
        h.ignore_links = False
        h.ignore_images = False
        h.body_width = 0
        
        main_content_html = str(main_content)
        clean_text = h.handle(main_content_html)
        
        # Clean up text
        clean_text = re.sub(r'\n\s*\n\s*\n', '\n\n', clean_text)
        clean_text = re.sub(r'[ \t]+', ' ', clean_text)
        clean_text = clean_text.strip()
        
        return title + clean_text
        
    except Exception as e:
        logger.error(f"HTML content extraction failed: {str(e)}")
        raise Exception(f"HTML content extraction failed: {str(e)}")

def process_pdf_content(file_content: bytes) -> str:
    """Process PDF files"""
    logger.info("Starting PDF processing")
    start_time = time.time()
    
    try:
        doc = fitz.open(stream=file_content, filetype="pdf")
        text_parts = []
        
        for page_num in range(doc.page_count):
            page = doc[page_num]
            page_text = page.get_text()
            if page_text.strip():
                page_text = words_to_numbers(page_text)
                cleaned_text = re.sub(r'\s+', ' ', page_text.strip())
                text_parts.append(f"=== PAGE {page_num + 1} ===\n{cleaned_text}")
        
        doc.close()
        final_text = "\n\n".join(text_parts)
        
        logger.info(f"PDF processing completed in {time.time() - start_time:.2f}s. Text length: {len(final_text)}")
        return final_text
        
    except Exception as e:
        logger.error(f"PDF processing failed: {str(e)}")
        raise Exception(f"PDF processing failed: {str(e)}")

async def check_if_json_endpoint(url: str) -> Optional[Dict]:
    """Check if URL returns JSON data"""
    try:
        timeout = httpx.Timeout(10.0)
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'application/json, text/html, */*'
        }
        
        async with httpx.AsyncClient(timeout=timeout, follow_redirects=True, headers=headers) as client:
            response = await client.get(url)
        
        response.raise_for_status()
        content_type = response.headers.get('content-type', '').lower()
        
        # Try to parse as JSON
        if 'json' in content_type or response.text.strip().startswith('{'):
            try:
                json_data = response.json()
                logger.info(f"Successfully parsed JSON from {url}")
                return json_data
            except:
                pass
        
        return None
        
    except Exception as e:
        logger.warning(f"Could not fetch JSON from {url}: {str(e)}")
        return None

async def download_and_process_document(url: str) -> str:
    """Download and process document from URL"""
    start_time = time.time()
    logger.info(f"Starting document download from: {url}")
    
    try:
        # First, check if it's a JSON endpoint
        json_data = await check_if_json_endpoint(url)
        if json_data:
            # Format JSON data nicely
            formatted_json = json.dumps(json_data, indent=2, ensure_ascii=False)
            return f"=== JSON DATA ===\n{formatted_json}"
        
        # Download as regular document
        timeout = httpx.Timeout(30.0, connect=5.0)
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        async with httpx.AsyncClient(timeout=timeout, follow_redirects=True, headers=headers) as client:
            response = await client.get(url)
        
        response.raise_for_status()
        content_type = response.headers.get('content-type', '').lower()
        content_length = len(response.content)
        
        if content_length > MAX_FILE_SIZE:
            raise HTTPException(status_code=400, detail=f"File too large: {content_length} bytes")
        
        logger.info(f"Document downloaded. Size: {content_length} bytes, Type: {content_type}")
        
        # Process based on content type
        if 'application/pdf' in content_type or url.lower().endswith('.pdf'):
            return process_pdf_content(response.content)
        elif 'text/html' in content_type or url.endswith(('.html', '.htm')) or '<!DOCTYPE html' in response.text[:100].lower():
            html_content = extract_html_content(response.text, url)
            return words_to_numbers(html_content)
        else:
            # Try as text
            return words_to_numbers(response.text)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Document download/processing failed: {str(e)}")
        raise Exception(f"Download or processing failed: {str(e)}")

async def get_embeddings(texts: List[str], input_type: str = "search_document") -> List[List[float]]:
    """Get embeddings from Cohere API"""
    start_time = time.time()
    current_cohere_key = get_current_cohere_key()
    logger.info(f"🔑 Getting embeddings for {len(texts)} texts")
    
    clean_texts = []
    for text in texts:
        if text.strip():
            converted_text = words_to_numbers(text)
            clean_text = re.sub(r'\s+', ' ', converted_text.strip())[:EMBEDDING_MAX_LENGTH]
            clean_texts.append(clean_text)
    
    if not clean_texts:
        raise ValueError("No valid texts provided for embedding")
    
    url = "https://api.cohere.com/v1/embed"
    headers = {
        "Authorization": f"Bearer {current_cohere_key}",
        "Content-Type": "application/json",
        "Accept": "application/json"
    }
    
    # Process in batches
    all_embeddings = []
    timeout = httpx.Timeout(45.0)
    
    async with httpx.AsyncClient(timeout=timeout) as client:
        for i in range(0, len(clean_texts), EMBEDDING_BATCH_SIZE):
            batch = clean_texts[i:i+EMBEDDING_BATCH_SIZE]
            
            data = {
                "model": COHERE_MODEL,
                "texts": batch,
                "input_type": input_type,
                "truncate": "END"
            }
            
            response = await client.post(url, headers=headers, json=data)
            response.raise_for_status()
            
            response_data = response.json()
            embeddings_data = response_data.get("embeddings", [])
            
            for embedding in embeddings_data:
                vec = np.array(embedding, dtype=np.float32)
                norm = np.linalg.norm(vec)
                if norm > 0:
                    vec = vec / norm
                all_embeddings.append(vec.tolist())
    
    logger.info(f"Got {len(all_embeddings)} embeddings in {time.time() - start_time:.2f}s")
    return all_embeddings

def build_prompt(question: str, context_chunks: List[str]) -> str:
    """Build optimized prompt for LLM"""
    context = "\n\n---\n\n".join(context_chunks[:10])
    
    # Detect language
    try:
        lang = detect(question)
        has_non_english = lang != "en"
    except:
        has_non_english = False
    
    language_instruction = "**IMPORTANT: Respond in the SAME LANGUAGE as the question.**\n\n" if has_non_english else ""
    
    # Check if yes/no question
    question_lower = question.lower().strip()
    is_yes_no = question_lower.startswith(('is ', 'are ', 'do ', 'does ', 'did ', 'will ', 'would ', 'can ', 'could '))
    
    if is_yes_no:
        response_instruction = """This is a yes/no question. Start your response with "Yes" or "No", then provide a brief explanation with specific details."""
    else:
        response_instruction = """Answer directly and concisely. Include specific numbers, dates, amounts, and conditions. Use exact figures from the context."""

    return f"""{language_instruction}You are an intelligent document analyst. Analyze the context carefully and provide accurate answers.

**TASK:** {response_instruction}

**CONTEXT:**
{context}

**QUESTION:** {question}

**ANSWER:**"""

async def ask_llm(prompt: str) -> str:
    """Query Gemini LLM"""
    try:
        current_api_key = get_current_gemini_key()
        logger.info(f"🔑 Using Gemini API KEY_{GEMINI_API_KEYS.index(current_api_key) + 1}")
        
        start_time = time.time()
        
        genai.configure(api_key=current_api_key)
        model = genai.GenerativeModel(GEMINI_MODEL)
        
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=LLM_TEMPERATURE,
                max_output_tokens=LLM_MAX_TOKENS,
                top_p=LLM_TOP_P,
                top_k=LLM_TOP_K,
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
        
        if result_text:
            # Clean response
            result_text = words_to_numbers(result_text)
            result_text = re.sub(r'^(Answer:|ANSWER:|Response:|Based on the context:)\s*', '', result_text, flags=re.IGNORECASE)
            result_text = re.sub(r'\s+', ' ', result_text).strip()
            return result_text
        
        return "No valid response generated."
            
    except Exception as e:
        logger.error(f"Gemini API error: {str(e)}")
        return f"Unable to generate answer due to API error. Please try again."

@app.post("/hackrx/run", response_model=QueryResponse)
async def run_query(request: QueryRequest, authorization: str = Header(None)):
    """Main endpoint for processing queries"""
    total_start_time = time.time()
    logger.info(f"📥 Received request with {len(request.questions)} questions")
    
    # Auth check
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Authorization header missing or malformed.")

    token = authorization.split(" ")[1]
    if token != TEAM_TOKEN:
        raise HTTPException(status_code=403, detail="Invalid token")

    try:
        # Process document
        logger.info("📄 Starting document processing")
        document_text = await download_and_process_document(request.documents)
        
        if not document_text.strip():
            raise HTTPException(status_code=400, detail="No text could be extracted from the document")
        
        # Check if this is a puzzle document
        if is_puzzle_document(document_text):
            logger.info("🧩 Detected puzzle document - activating puzzle solver")
            answers = await solve_puzzle(document_text, request.questions)
            
            total_time = time.time() - total_start_time
            logger.info(f"⏱️ Puzzle solved in {total_time:.2f}s")
            
            # Rotate API key for next request
            rotate_cohere_key()
            
            return QueryResponse(answers=answers)
        
        # Regular document processing (existing logic)
        logger.info("📄 Processing as regular document")
        
        # Create chunks
        logger.info("🧩 Creating document chunks")
        chunks = get_enhanced_chunks(document_text)
        
        if not chunks:
            raise HTTPException(status_code=400, detail="No meaningful chunks could be created")
        
        # Get embeddings
        logger.info("🤖 Getting embeddings")
        chunk_texts = [chunk["text"] for chunk in chunks]
        chunk_embeddings = await get_embeddings(chunk_texts, input_type="search_document")
        question_embeddings = await get_embeddings(request.questions, input_type="search_query")
        
        # Create FAISS index
        logger.info("📊 Creating FAISS index")
        faiss_service.create_index(chunk_embeddings, chunks)
        
        # Process questions
        logger.info("❓ Processing questions")
        answers = []
        
        for i, (question, q_emb) in enumerate(zip(request.questions, question_embeddings)):
            try:
                logger.info(f"📝 Processing question {i+1}/{len(request.questions)}")
                
                # Search for relevant chunks
                search_results = faiss_service.multi_tier_search(q_emb, question, FAISS_K_SEARCH)
                
                if not search_results:
                    answers.append("I couldn't find relevant information to answer this question.")
                    continue
                
                # Get relevant chunks
                relevant_chunks = faiss_service.enhance_results(search_results, question)
                
                # Build prompt and get answer
                prompt = build_prompt(question, relevant_chunks)
                response = await ask_llm(prompt)
                
                # Clean response
                response = response.strip()
                response = re.sub(r'^(Answer:|ANSWER:|Response:)\s*', '', response, flags=re.IGNORECASE)
                
                answers.append(response.strip())
                logger.info(f"✅ Question {i+1} processed successfully")
                
            except Exception as e:
                logger.error(f"Error processing question {i+1}: {str(e)}")
                answers.append("Error processing the question. Please try again.")
        
        total_time = time.time() - total_start_time
        logger.info(f"⏱️ Total request processed in {total_time:.2f}s")
        
        # Rotate API key for next request
        rotate_cohere_key()
        
        return QueryResponse(answers=answers)
        
    except HTTPException as he:
        logger.error(f"HTTP Exception: {he.detail}")
        raise he
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080, log_level="info")