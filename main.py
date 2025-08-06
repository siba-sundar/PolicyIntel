from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel
import httpx
import asyncio
import fitz  # PyMuPDF
import google.generativeai as genai
import os
from dotenv import load_dotenv
from typing import List, Dict, Optional, Tuple
import numpy as np
import json
import re
import hashlib
import logging
import time
import tempfile
import zipfile
import io
from pathlib import Path
import gc

# New imports for additional formats
import openpyxl
from openpyxl import load_workbook
from pptx import Presentation
from PIL import Image
import pytesseract
import docx
from docx import Document

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

# ---- Supported File Formats ----
SUPPORTED_FORMATS = {
    'pdf', 'jpg', 'jpeg', 'png', 'gif', 'bmp', 'tiff', 'webp',
    'xlsx', 'xls', 'pptx', 'ppt', 'docx', 'doc', 'txt', 'zip'
}

UNSUPPORTED_FORMATS = {
    'exe', 'bat', 'sh', 'dll', 'sys', 'bin', 'iso', 'dmg',
    'mp4', 'avi', 'mov', 'mp3', 'wav', 'flac', 'mkv'
}

# Memory management settings
MAX_FILE_SIZE = 500 * 1024 * 1024  # 500MB
MAX_IMAGE_SIZE = (2048, 2048)  # Resize large images
OCR_BATCH_SIZE = 5  # Process images in small batches

def get_current_gemini_key():
    """Get current Gemini API key and handle rotation"""
    global current_key_index, request_count
    
    current_key = GEMINI_API_KEYS[current_key_index]
    request_count += 1
    
    if request_count >= REQUESTS_PER_KEY:
        request_count = 0
        current_key_index = (current_key_index + 1) % len(GEMINI_API_KEYS)
        logger.info(f"🔄 SWITCHED to Gemini API key #{current_key_index + 1}")
    
    return current_key

app = FastAPI(title="PolicyIntel API", description="Document processing and Q&A API", version="1.0.0")

# ---- Health Check Endpoint ----
@app.get("/")
@app.get("/health")
async def health_check():
    """Health check endpoint for Render deployment"""
    return {"status": "healthy", "service": "PolicyIntel API", "version": "1.0.0"}

# ---- Auth Token ----
TEAM_TOKEN = "833695cad1c0d2600066bf2b08aab7614d0dec93b4b6f0ae3acd37ef7d6fcb1c"

# ---- Data Models ----
class QueryRequest(BaseModel):
    documents: str
    questions: List[str]

class QueryResponse(BaseModel):
    answers: List[str]

# ---- Memory Management Utilities ----
def clear_memory():
    """Force garbage collection to free memory"""
    gc.collect()

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

# ---- Excel Parser with Proper Header-Value Association ----
def parse_excel_with_headers(file_content: bytes) -> str:
    """Parse Excel file with proper header-value association"""
    logger.info("Starting Excel parsing with header association")
    start_time = time.time()
    
    try:
        # Load workbook from bytes
        workbook = load_workbook(io.BytesIO(file_content), data_only=True)
        all_text_parts = []
        
        for sheet_name in workbook.sheetnames:
            logger.info(f"Processing Excel sheet: {sheet_name}")
            sheet = workbook[sheet_name]
            
            # Get sheet dimensions efficiently
            max_row = sheet.max_row
            max_col = sheet.max_column
            
            if max_row == 1 or max_col == 1:
                continue  # Skip empty or single-cell sheets
                
            # Convert to list of lists for easier processing
            data_rows = []
            for row in sheet.iter_rows(min_row=1, max_row=min(max_row, 1000), values_only=True):  # Limit rows for memory
                # Convert None to empty string and clean data
                cleaned_row = [str(cell).strip() if cell is not None else "" for cell in row]
                if any(cleaned_row):  # Only add non-empty rows
                    data_rows.append(cleaned_row)
            
            if len(data_rows) < 2:
                continue  # Need at least header + 1 data row
                
            # Identify header row (usually first row)
            headers = data_rows[0]
            data_rows = data_rows[1:]
            
            # Clean and validate headers
            clean_headers = []
            for i, header in enumerate(headers):
                if header and str(header).strip():
                    clean_headers.append(str(header).strip())
                else:
                    clean_headers.append(f"Column_{i+1}")  # Default name for empty headers
            
            # Build structured text with header-value pairs
            sheet_text_parts = [f"=== EXCEL SHEET: {sheet_name} ===\n"]
            
            # Method 1: Row-wise representation with clear header-value mapping
            for row_idx, row_data in enumerate(data_rows[:500], 1):  # Limit rows for memory
                if not any(str(cell).strip() for cell in row_data if cell):  # Skip empty rows
                    continue
                    
                row_text_parts = [f"Row {row_idx}:"]
                for header, value in zip(clean_headers, row_data):
                    if value and str(value).strip():
                        clean_value = str(value).strip()
                        # Create clear associations
                        row_text_parts.append(f"{header}: {clean_value}")
                
                if len(row_text_parts) > 1:  # Has actual data
                    sheet_text_parts.append(" | ".join(row_text_parts))
            
            # Method 2: Column-wise summary for better context
            sheet_text_parts.append(f"\n--- COLUMN SUMMARIES for {sheet_name} ---")
            for col_idx, header in enumerate(clean_headers):
                if col_idx >= len(data_rows[0]) if data_rows else True:
                    continue
                    
                # Extract column values
                column_values = []
                for row in data_rows[:100]:  # Sample first 100 rows
                    if col_idx < len(row) and row[col_idx] and str(row[col_idx]).strip():
                        val = str(row[col_idx]).strip()
                        if val not in column_values:  # Avoid duplicates
                            column_values.append(val)
                        if len(column_values) >= 10:  # Limit unique values shown
                            break
                
                if column_values:
                    values_preview = ", ".join(column_values[:5])
                    if len(column_values) > 5:
                        values_preview += f"... ({len(column_values)} unique values)"
                    sheet_text_parts.append(f"{header} contains: {values_preview}")
            
            all_text_parts.extend(sheet_text_parts)
            all_text_parts.append("\n" + "="*50 + "\n")
            
            # Memory cleanup
            del data_rows
            clear_memory()
        
        workbook.close()
        final_text = "\n".join(all_text_parts)
        
        logger.info(f"Excel parsing completed in {time.time() - start_time:.2f}s. Text length: {len(final_text)}")
        return final_text
        
    except Exception as e:
        logger.error(f"Excel parsing failed: {str(e)}")
        raise Exception(f"Excel parsing failed: {str(e)}")

# ---- PowerPoint Parser ----
def parse_powerpoint(file_content: bytes) -> str:
    """Parse PowerPoint files extracting text and handling images with OCR"""
    logger.info("Starting PowerPoint parsing with image OCR support")
    start_time = time.time()
    
    try:
        presentation = Presentation(io.BytesIO(file_content))
        all_text_parts = []
        
        for slide_num, slide in enumerate(presentation.slides, 1):
            slide_text_parts = [f"=== SLIDE {slide_num} ==="]
            
            # Extract text from all text boxes and shapes
            slide_text = []
            for shape in slide.shapes:
                # Extract regular text
                if hasattr(shape, 'text') and shape.text.strip():
                    slide_text.append(shape.text.strip())
                
                # Handle tables in slides
                if shape.has_table:
                    table = shape.table
                    table_text = ["TABLE DATA:"]
                    for row in table.rows:
                        row_data = []
                        for cell in row.cells:
                            if cell.text.strip():
                                row_data.append(cell.text.strip())
                        if row_data:
                            table_text.append(" | ".join(row_data))
                    slide_text.append("\n".join(table_text))
                
                # Handle images with OCR
                try:
                    if shape.shape_type == 13:  # Picture type
                        logger.info(f"Found image in slide {slide_num}, processing with OCR")
                        
                        # Extract image bytes
                        image_stream = shape.image.blob
                        
                        # Process with OCR
                        ocr_text = process_image_ocr(image_stream, f"slide_{slide_num}_image")
                        
                        if ocr_text and ocr_text.strip() and "failed" not in ocr_text.lower():
                            slide_text.append(f"IMAGE_TEXT: {ocr_text.strip()}")
                            logger.info(f"Successfully extracted text from image in slide {slide_num}")
                        else:
                            logger.info(f"No meaningful text found in image from slide {slide_num}")
                            
                except Exception as img_error:
                    # Log but don't fail the entire slide processing
                    logger.warning(f"Could not process image in slide {slide_num}: {str(img_error)}")
                    continue
            
            # Add slide content if we have any
            if slide_text:
                slide_text_parts.extend(slide_text)
            
            # Always add the slide (even if empty) to maintain structure
            all_text_parts.append("\n".join(slide_text_parts))
            
            # Memory management - process slides in batches
            if slide_num % 10 == 0:  # More frequent cleanup due to image processing
                clear_memory()
                logger.info(f"Processed {slide_num} slides so far...")
        
        final_text = "\n\n".join(all_text_parts)
        logger.info(f"PowerPoint parsing completed in {time.time() - start_time:.2f}s. Text length: {len(final_text)} characters")
        return final_text
        
    except Exception as e:
        logger.error(f"PowerPoint parsing failed: {str(e)}")
        raise Exception(f"PowerPoint parsing failed: {str(e)}")

# ---- Word Document Parser ----
def parse_word_document(file_content: bytes) -> str:
    """Parse Word documents"""
    logger.info("Starting Word document parsing")
    start_time = time.time()
    
    try:
        document = Document(io.BytesIO(file_content))
        text_parts = []
        
        # Extract paragraphs
        for para in document.paragraphs:
            if para.text.strip():
                text_parts.append(para.text.strip())
        
        # Extract tables with proper structure
        for table in document.tables:
            table_text = ["TABLE DATA:"]
            for row in table.rows:
                row_data = []
                for cell in row.cells:
                    if cell.text.strip():
                        row_data.append(cell.text.strip())
                if row_data:
                    table_text.append(" | ".join(row_data))
            if len(table_text) > 1:
                text_parts.append("\n".join(table_text))
        
        final_text = "\n\n".join(text_parts)
        logger.info(f"Word document parsing completed in {time.time() - start_time:.2f}s. Text length: {len(final_text)}")
        return final_text
        
    except Exception as e:
        logger.error(f"Word document parsing failed: {str(e)}")
        raise Exception(f"Word document parsing failed: {str(e)}")

# ---- Image OCR Processing ----
def process_image_ocr(image_content: bytes, filename: str = "") -> str:
    """Process image with OCR, memory-optimized"""
    logger.info(f"Starting OCR processing for image: {filename}")
    start_time = time.time()
    
    try:
        # Load and optimize image
        image = Image.open(io.BytesIO(image_content))
        
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize if too large to save memory
        if image.size[0] > MAX_IMAGE_SIZE[0] or image.size[1] > MAX_IMAGE_SIZE[1]:
            image.thumbnail(MAX_IMAGE_SIZE, Image.Resampling.LANCZOS)
            logger.info(f"Resized image to {image.size} for memory efficiency")
        
        # Perform OCR with optimized settings
        ocr_config = '--oem 3 --psm 6 -l eng'  # Optimized OCR settings
        extracted_text = pytesseract.image_to_string(image, config=ocr_config)
        
        # Clean up image from memory
        image.close()
        clear_memory()
        
        # Clean extracted text
        cleaned_text = re.sub(r'\s+', ' ', extracted_text.strip())
        
        logger.info(f"OCR completed in {time.time() - start_time:.2f}s. Text length: {len(cleaned_text)}")
        return cleaned_text if cleaned_text else "No text detected in image"
        
    except Exception as e:
        logger.error(f"OCR processing failed for {filename}: {str(e)}")
        return f"OCR processing failed for {filename}"

# ---- ZIP File Handler ----
async def extract_and_process_zip(file_content: bytes) -> str:
    """Extract and process ZIP files recursively"""
    logger.info("Starting ZIP file processing")
    start_time = time.time()
    
    all_extracted_text = []
    processed_files = []
    
    try:
        with zipfile.ZipFile(io.BytesIO(file_content), 'r') as zip_file:
            file_list = zip_file.namelist()
            logger.info(f"Found {len(file_list)} files in ZIP")
            
            for file_name in file_list:
                # Skip directories and hidden files
                if file_name.endswith('/') or file_name.startswith('.'):
                    continue
                
                # Check file size before extraction
                file_info = zip_file.getinfo(file_name)
                if file_info.file_size > MAX_FILE_SIZE:
                    logger.warning(f"Skipping {file_name}: file too large ({file_info.file_size} bytes)")
                    continue
                
                # Check if format is supported
                is_supported, error_msg = is_supported_format(file_name)
                if not is_supported:
                    logger.info(f"Skipping {file_name}: {error_msg}")
                    continue
                
                try:
                    # Extract file content
                    extracted_content = zip_file.read(file_name)
                    logger.info(f"Processing extracted file: {file_name}")
                    
                    # Process based on file type
                    file_text = await process_file_content(extracted_content, file_name)
                    
                    if file_text and file_text.strip():
                        all_extracted_text.append(f"=== FILE: {file_name} ===\n{file_text}\n")
                        processed_files.append(file_name)
                    
                    # Memory cleanup after each file
                    del extracted_content
                    clear_memory()
                    
                except Exception as e:
                    logger.error(f"Error processing {file_name} from ZIP: {str(e)}")
                    all_extracted_text.append(f"=== FILE: {file_name} ===\nError processing file: {str(e)}\n")
        
        final_text = "\n".join(all_extracted_text)
        logger.info(f"ZIP processing completed in {time.time() - start_time:.2f}s. Processed {len(processed_files)} files")
        
        return final_text if final_text.strip() else "No processable content found in ZIP file"
        
    except Exception as e:
        logger.error(f"ZIP processing failed: {str(e)}")
        raise Exception(f"ZIP processing failed: {str(e)}")

# ---- Unified File Content Processor ----
async def process_file_content(file_content: bytes, filename: str) -> str:
    """Process file content based on file type"""
    ext = get_file_extension(filename)
    
    try:
        if ext == 'pdf':
            # Use existing PDF processing
            doc = fitz.open(stream=file_content, filetype="pdf")
            text_parts = []
            max_pages = min(50, doc.page_count)
            
            for page_num in range(max_pages):
                page = doc[page_num]
                page_text = page.get_text()
                if page_text.strip():
                    page_text = words_to_numbers(page_text)
                    cleaned_text = re.sub(r'\s+', ' ', page_text.strip())
                    text_parts.append(cleaned_text)
            
            doc.close()
            return "\n\n".join(text_parts)
            
        elif ext in ['xlsx', 'xls']:
            return parse_excel_with_headers(file_content)
            
        elif ext in ['pptx', 'ppt']:
            return parse_powerpoint(file_content)
            
        elif ext in ['docx', 'doc']:
            return parse_word_document(file_content)
            
        elif ext in ['jpg', 'jpeg', 'png', 'gif', 'bmp', 'tiff', 'webp']:
            return process_image_ocr(file_content, filename)
            
        elif ext == 'txt':
            # Handle text files
            try:
                text_content = file_content.decode('utf-8')
            except UnicodeDecodeError:
                try:
                    text_content = file_content.decode('latin-1')
                except UnicodeDecodeError:
                    text_content = file_content.decode('utf-8', errors='ignore')
            
            return words_to_numbers(text_content.strip())
            
        elif ext == 'zip':
            return await extract_and_process_zip(file_content)
            
        else:
            raise Exception(f"Unsupported file format: {ext}")
            
    except Exception as e:
        logger.error(f"Error processing {filename}: {str(e)}")
        raise Exception(f"Error processing {filename}: {str(e)}")

# ---- Enhanced Document Downloader ----
async def download_and_process_document(url: str) -> str:
    """Download and process document from URL"""
    start_time = time.time()
    logger.info(f"Starting document download from: {url}")
    
    try:
        # Determine expected file type from URL
        filename = url.split('/')[-1].split('?')[0]  # Remove query parameters
        if not filename or '.' not in filename:
            filename = "document.pdf"  # Default assumption
        
        # Check if format is supported
        is_supported, error_msg = is_supported_format(filename)
        if not is_supported:
            raise HTTPException(status_code=400, detail=error_msg)
        
        # Download with optimized settings
        timeout = httpx.Timeout(30.0, connect=5.0)
        limits = httpx.Limits(max_keepalive_connections=5, max_connections=10)
        
        async with httpx.AsyncClient(timeout=timeout, limits=limits, follow_redirects=True) as client:
            response = await client.get(url)
        
        response.raise_for_status()
        
        # Check file size
        content_length = len(response.content)
        if content_length > MAX_FILE_SIZE:
            raise HTTPException(status_code=400, detail=f"File too large: {content_length} bytes. Maximum allowed: {MAX_FILE_SIZE} bytes")
        
        logger.info(f"Document downloaded successfully. Size: {content_length} bytes")
        
        # Process based on file type
        document_text = await process_file_content(response.content, filename)
        
        # Clean up
        del response
        clear_memory()
        
        logger.info(f"Document processing completed in {time.time() - start_time:.2f}s. Text length: {len(document_text)}")
        return document_text
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Document download/processing failed: {str(e)}")
        raise Exception(f"Download or processing failed: {str(e)}")

# ---- Keep all existing functions unchanged ----
def split_text_into_chunks(text: str, chunk_size: int = 500, overlap: int = 100) -> List[str]:
    start_time = time.time()
    logger.info("Starting enhanced text chunking")
    
    text = re.sub(r'\s+', ' ', text.strip())
    
    sentences = []
    sentence_pattern = r'(?<=[.!?])\s+(?=[A-Z])'
    raw_sentences = re.split(sentence_pattern, text)
    
    for sentence in raw_sentences:
        sentence = sentence.strip()
        if len(sentence) > 15:
            sentences.append(sentence)
    
    logger.info(f"Split into {len(sentences)} sentences")
    
    chunks = []
    words = text.split()
    
    for i in range(0, len(words), chunk_size - overlap):
        chunk_words = words[i:i + chunk_size]
        if len(chunk_words) >= 20:
            chunk_text = " ".join(chunk_words)
            if len(chunk_text) < len(text) and not chunk_text.endswith(('.', '!', '?')):
                next_words = words[i + chunk_size:i + chunk_size + 20]
                if next_words:
                    extended = " ".join(chunk_words + next_words)
                    next_boundary = extended.find('.', len(chunk_text))
                    if next_boundary != -1 and next_boundary < len(chunk_text) + 100:
                        chunk_text = extended[:next_boundary + 1]
            
            chunks.append(chunk_text)
    
    logger.info(f"Created {len(chunks)} enhanced chunks in {time.time() - start_time:.2f}s")
    return chunks

async def get_embeddings(texts: List[str], input_type: str = "search_document") -> List[List[float]]:
    start_time = time.time()
    logger.info(f"Getting embeddings for {len(texts)} texts")
    
    url = "https://api.cohere.com/v1/embed"
    headers = {
        "Authorization": f"Bearer {COHERE_API_KEY}",
        "Content-Type": "application/json",
        "Accept": "application/json"
    }
    
    clean_texts = []
    for text in texts:
        if text.strip():
            converted_text = words_to_numbers(text)
            clean_text = re.sub(r'\s+', ' ', converted_text.strip())[:2000]
            clean_texts.append(clean_text)
    
    if not clean_texts:
        raise ValueError("No valid texts provided for embedding")

    BATCH_SIZE = 96
    all_embeddings = []
    
    timeout = httpx.Timeout(45.0)
    limits = httpx.Limits(max_keepalive_connections=5, max_connections=10)
    
    async with httpx.AsyncClient(timeout=timeout, limits=limits) as client:
        for i in range(0, len(clean_texts), BATCH_SIZE):
            batch = clean_texts[i:i+BATCH_SIZE]
            
            data = {
                "model": "embed-english-v3.0",
                "texts": batch,
                "input_type": input_type,
                "truncate": "END"
            }
            
            try:
                response = await client.post(url, headers=headers, json=data)
                
                if response.status_code == 200:
                    response_data = response.json()
                    embeddings_data = response_data.get("embeddings", [])
                    
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

def cosine_similarity_batch(query_vec: List[float], chunk_embeddings: List[List[float]]) -> List[float]:
    """Vectorized cosine similarity for speed"""
    query_arr = np.array(query_vec, dtype=np.float32)
    chunk_arr = np.array(chunk_embeddings, dtype=np.float32)
    similarities = np.dot(chunk_arr, query_arr)
    return similarities.tolist()

async def search_similar_chunks(query_embedding: List[float], chunk_embeddings: List[List[float]], 
                               chunks: List[str], question: str, k=15) -> List[str]:
    start_time = time.time()
    
    similarities = cosine_similarity_batch(query_embedding, chunk_embeddings)
    max_similarity = max(similarities) if similarities else 0
    
    key_terms = extract_key_terms(question.lower())
    question_lower = question.lower()
    
    scored_chunks = []
    
    for i, sim_score in enumerate(similarities):
        chunk = chunks[i]
        chunk_lower = chunk.lower()
        
        keyword_score = 0
        if key_terms:
            exact_matches = sum(1 for term in key_terms if f" {term} " in f" {chunk_lower} ")
            partial_matches = sum(1 for term in key_terms if term in chunk_lower and f" {term} " not in f" {chunk_lower} ")
            
            numeric_matches = 0
            for term in key_terms:
                if re.search(r'\d', term):
                    if term in chunk_lower:
                        numeric_matches += 2
            
            keyword_score = (exact_matches * 0.25 + partial_matches * 0.1 + numeric_matches * 0.15) / len(key_terms)
        
        question_words = set(re.findall(r'\b\w+\b', question_lower))
        chunk_words = set(re.findall(r'\b\w+\b', chunk_lower))
        overlap_score = len(question_words.intersection(chunk_words)) / len(question_words) * 0.15
        
        final_score = sim_score + keyword_score + overlap_score
        scored_chunks.append((i, chunk, final_score, sim_score))
    
    scored_chunks.sort(key=lambda x: x[2], reverse=True)
    
    if max_similarity < 0.25:
        logger.warning(f"🔍 Low semantic similarity ({max_similarity:.3f}) - enhancing with keyword search")
        keyword_boosted = []
        for i, chunk, final_score, sem_score in scored_chunks:
            if any(term in chunk.lower() for term in key_terms):
                keyword_boosted.append((i, chunk, final_score + 0.4, sem_score))
            else:
                keyword_boosted.append((i, chunk, final_score, sem_score))
        
        keyword_boosted.sort(key=lambda x: x[2], reverse=True)
        scored_chunks = keyword_boosted
    
    result_chunks = []
    for idx, (chunk_idx, chunk, final_score, sem_score) in enumerate(scored_chunks[:k]):
        indexed_chunk = f"[CONTEXT {idx+1}] {chunk}"
        result_chunks.append(indexed_chunk)
    
    logger.info(f"Similarity search completed in {time.time() - start_time:.2f}s. Max sim: {max_similarity:.3f}")
    return result_chunks

def extract_key_terms(question: str) -> List[str]:
    important_terms = {
        'coverage', 'limit', 'deductible', 'premium', 'claim', 'benefit', 'exclusion',
        'copay', 'coinsurance', 'maximum', 'minimum', 'annual', 'lifetime', 'policy',
        'insured', 'covered', 'eligible', 'amount', 'percentage', 'network', 'provider',
        'emergency', 'prescription', 'medical', 'dental', 'vision', 'mental', 'health',
        'hospital', 'outpatient', 'inpatient', 'surgery', 'diagnostic', 'preventive',
        'waiting', 'period', 'authorization', 'existing', 'condition'
    }
    
    question_converted = words_to_numbers(question.lower())
    
    question_words = set(re.findall(r'\b\w+\b', question_converted))
    key_terms = list(important_terms.intersection(question_words))
    
    numbers = re.findall(r'\b\d+(?:\.\d+)?%?\b', question_converted)
    dollar_amounts = re.findall(r'\$[\d,]+(?:\.\d{2})?', question_converted)
    key_terms.extend(numbers + dollar_amounts)
    
    phrases = re.findall(r'\b(?:out of pocket|prior authorization|pre existing|waiting period|per year|per day)\b', question_converted)
    key_terms.extend(phrases)
    
    return key_terms

def is_yes_no_question(question: str) -> bool:
    question_lower = question.lower().strip()
    return question_lower.startswith(('is ', 'are ', 'do ', 'does ', 'did ', 'will ', 'would ', 'can ', 'could '))

def build_prompt(question: str, context_chunks: List[str]) -> str:
    context = "\n\n---\n\n".join(context_chunks[:12])
    
    is_yes_no = is_yes_no_question(question)
    key_terms = extract_key_terms(question.lower())
    
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

    return f"""You are an expert document analyst. Analyze the provided context carefully and provide precise answers.

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

**DOCUMENT CONTEXT:**
{context}

**QUESTION:** {question}

**ANSWER:**"""

async def ask_llm(prompt: str, retry_count: int = 0) -> str:
    try:
        current_api_key = get_current_gemini_key()
        logger.info(f"🔑 Using Gemini API KEY_{GEMINI_API_KEYS.index(current_api_key) + 1} (Request #{request_count}/12)")
        
        start_time = time.time()
        
        client = genai.Client(api_key=current_api_key)
        from google.genai import types
        
        response = client.models.generate_content(
            model="gemini-2.5-flash-lite", 
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.1,
                max_output_tokens=180,
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
        
        if result_text:
            result_text = words_to_numbers(result_text)
            result_text = re.sub(r'^(Answer:|ANSWER:|Response:|Based on the context:)\s*', '', result_text, flags=re.IGNORECASE)
            result_text = re.sub(r'\s+', ' ', result_text).strip()
            if result_text.endswith('.') and not result_text.endswith('etc.') and not result_text.endswith('vs.'):
                result_text = result_text[:-1]
            
            return result_text
        
        return "No valid response generated."
            
    except Exception as e:
        logger.error(f"Gemini API error (attempt {retry_count + 1}): {str(e)}")
        
        if retry_count < 1 and ("quota" not in str(e).lower() and "limit" not in str(e).lower()):
            logger.info("Retrying with next API key...")
            return await ask_llm(prompt, retry_count + 1)
        
        return f"Unable to generate answer due to API error. Please try again."
    
@app.get("/")
async def root():
    return {
        "message": "Enhanced Multi-Format Document Processor Running", 
        "status": "enhanced_multi_format_support",
        "supported_formats": list(SUPPORTED_FORMATS),
        "api_keys_count": len(GEMINI_API_KEYS),
        "current_key_index": current_key_index + 1,
        "requests_on_current_key": request_count,
        "max_file_size_mb": MAX_FILE_SIZE / (1024 * 1024)
    }

@app.post("/hackrx/run", response_model=QueryResponse)
async def run_query(request: QueryRequest, authorization: str = Header(None)):
    global qa_storage
    
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
        # Process document with enhanced multi-format support
        logger.info("Starting enhanced document processing")
        document_text = await download_and_process_document(request.documents)
        
        if not document_text.strip():
            logger.error("No text extracted from document")
            raise HTTPException(status_code=400, detail="No text could be extracted from the document")
            
        chunks = split_text_into_chunks(document_text)
        if not chunks:
            logger.error("No chunks created from document")
            raise HTTPException(status_code=400, detail="No meaningful chunks could be created from the document")
            
        # Clear document text from memory after chunking
        del document_text
        clear_memory()
        
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
                
                # Memory cleanup after each question
                clear_memory()
                
            except Exception as e:
                logger.error(f"Error processing question {i+1} '{question}': {str(e)}")
                error_answer = "Error processing the question. Please try again."
                answers.append(error_answer)
                qa_storage.append([question, error_answer])

        total_time = time.time() - total_start_time
        logger.info(f"Total request processed in {total_time:.2f}s")
        
        # Log all Q&A pairs after successful execution
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
        "version": "enhanced_multi_format_with_excel_headers",
        "supported_formats": list(SUPPORTED_FORMATS),
        "unsupported_formats": list(UNSUPPORTED_FORMATS),
        "api_keys_available": len(GEMINI_API_KEYS),
        "current_key": current_key_index + 1,
        "requests_on_current_key": request_count,
        "qa_storage_size": len(qa_storage),
        "max_file_size_mb": MAX_FILE_SIZE / (1024 * 1024)
    }

@app.get("/api-status")
async def api_status():
    """Endpoint to check API key rotation status"""
    return {
        "total_api_keys": len(GEMINI_API_KEYS),
        "current_key_index": current_key_index + 1,
        "requests_on_current_key": request_count,
        "requests_per_key_limit": REQUESTS_PER_KEY,
        "qa_storage_current_size": len(qa_storage),
        "supported_formats": list(SUPPORTED_FORMATS),
        "memory_limit_mb": MAX_FILE_SIZE / (1024 * 1024)
    }

@app.get("/supported-formats")
async def supported_formats():
    """Endpoint to check supported file formats"""
    return {
        "supported_formats": {
            "documents": ["pdf", "docx", "doc", "txt"],
            "spreadsheets": ["xlsx", "xls"],
            "presentations": ["pptx", "ppt"],
            "images": ["jpg", "jpeg", "png", "gif", "bmp", "tiff", "webp"],
            "archives": ["zip"]
        },
        "unsupported_formats": list(UNSUPPORTED_FORMATS),
        "max_file_size_mb": MAX_FILE_SIZE / (1024 * 1024),
        "features": {
            "excel_header_mapping": True,
            "nested_zip_extraction": True,
            "image_ocr": True,
            "memory_optimized": True,
            "format_validation": True
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")