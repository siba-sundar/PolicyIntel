import re
import logging
import time
from typing import List, Dict, Tuple, Optional
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import spacy
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# Download required NLTK data with comprehensive fallback
def ensure_nltk_data():
    """Ensure all required NLTK data is available"""
    required_data = [
        ('tokenizers/punkt', 'punkt'),
        ('tokenizers/punkt_tab', 'punkt_tab'),
        ('corpora/stopwords', 'stopwords'),
        ('taggers/averaged_perceptron_tagger', 'averaged_perceptron_tagger')
    ]
    
    for data_path, download_name in required_data:
        try:
            nltk.data.find(data_path)
        except LookupError:
            try:
                print(f"Downloading NLTK data: {download_name}")
                nltk.download(download_name, quiet=True)
            except Exception as e:
                print(f"Failed to download {download_name}: {e}")
                # Continue anyway, we'll handle it in the code

# Ensure NLTK data is available
ensure_nltk_data()

# Load spaCy model for advanced NLP
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    # Fallback to basic spaCy model
    nlp = None

from app.config.settings import (
    CHUNK_SIZE, CHUNK_OVERLAP, MIN_CHUNK_SIZE, MAX_CHUNK_SIZE,
    SENTENCE_MIN_LENGTH, PARAGRAPH_MIN_LENGTH, SEMANTIC_THRESHOLD,
    MAX_WORKERS, PARALLEL_CHUNK_SIZE
)

logger = logging.getLogger(__name__)

class EnhancedChunker:
    """Enhanced chunking service for precise Q&A with multiple strategies"""
    
    def __init__(self):
        try:
            self.stop_words = set(stopwords.words('english'))
        except Exception as e:
            logger.warning(f"Failed to load NLTK stopwords: {e}. Using basic fallback.")
            # Basic English stopwords fallback
            self.stop_words = set([
                'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
                'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the',
                'to', 'was', 'were', 'will', 'with', 'the', 'this', 'but', 'they',
                'have', 'had', 'what', 'said', 'each', 'which', 'she', 'do',
                'how', 'their', 'if', 'will', 'up', 'other', 'about', 'out',
                'many', 'then', 'them', 'these', 'so', 'some', 'her', 'would',
                'make', 'like', 'into', 'him', 'time', 'has', 'two', 'more'
            ])
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2)
        )
    
    def preprocess_text(self, text: str) -> str:
        """Clean and preprocess text"""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Fix common OCR errors
        text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)  # Add space between camelCase
        text = re.sub(r'([.!?])([A-Z])', r'\1 \2', text)  # Space after punctuation
        text = re.sub(r'(\d)([A-Z])', r'\1 \2', text)     # Space between number and letter
        
        # Normalize quotes and dashes
        text = re.sub(r'["""]', '"', text)
        text = re.sub(r'[––—]', '-', text)
        
        return text
    
    def extract_semantic_units(self, text: str) -> List[Dict]:
        """Extract semantic units (sentences, paragraphs, sections)"""
        units = []
        
        # Try advanced NLP if available
        if nlp:
            doc = nlp(text)
            
            # Extract named entities and key phrases
            entities = [(ent.text, ent.label_) for ent in doc.ents]
            
            # Extract sentences with metadata
            for sent in doc.sents:
                if len(sent.text.strip()) >= SENTENCE_MIN_LENGTH:
                    units.append({
                        'text': sent.text.strip(),
                        'type': 'sentence',
                        'entities': [e for e in entities if e[0] in sent.text],
                        'start_char': sent.start_char,
                        'end_char': sent.end_char
                    })
        else:
            # Fallback to NLTK with error handling
            try:
                sentences = sent_tokenize(text)
            except Exception as e:
                logger.warning(f"NLTK sentence tokenization failed: {e}. Using simple fallback.")
                # Simple fallback: split on sentence endings
                sentences = re.split(r'[.!?]+\s+', text)
            
            char_pos = 0
            
            for sent in sentences:
                if len(sent.strip()) >= SENTENCE_MIN_LENGTH:
                    units.append({
                        'text': sent.strip(),
                        'type': 'sentence',
                        'entities': [],
                        'start_char': char_pos,
                        'end_char': char_pos + len(sent)
                    })
                char_pos += len(sent) + 1
        
        return units
    
    def create_sliding_window_chunks(self, text: str) -> List[Dict]:
        """Create sliding window chunks with overlap"""
        words = text.split()
        chunks = []
        
        # Calculate word-based chunk size and overlap
        word_chunk_size = CHUNK_SIZE // 5  # Approximate words per chunk
        word_overlap = CHUNK_OVERLAP // 5
        
        for i in range(0, len(words), word_chunk_size - word_overlap):
            chunk_words = words[i:i + word_chunk_size]
            
            if len(chunk_words) < 10:  # Skip very small chunks
                continue
                
            chunk_text = ' '.join(chunk_words)
            
            # Ensure chunks end at sentence boundaries when possible
            if i + word_chunk_size < len(words):  # Not the last chunk
                # Look for sentence endings in the last 50 characters
                last_part = chunk_text[-50:]
                sentence_end = max(
                    last_part.rfind('.'),
                    last_part.rfind('!'),
                    last_part.rfind('?')
                )
                if sentence_end > 20:  # Found a good breaking point
                    chunk_text = chunk_text[:-(50-sentence_end)]
            
            if MIN_CHUNK_SIZE <= len(chunk_text) <= MAX_CHUNK_SIZE:
                chunks.append({
                    'text': chunk_text.strip(),
                    'type': 'sliding_window',
                    'start_word': i,
                    'end_word': i + len(chunk_words),
                    'word_count': len(chunk_words)
                })
        
        return chunks
    
    def create_semantic_chunks(self, text: str) -> List[Dict]:
        """Create semantic chunks based on topic coherence"""
        semantic_units = self.extract_semantic_units(text)
        
        if not semantic_units:
            return []
        
        # Group related sentences
        chunks = []
        current_chunk = []
        current_length = 0
        
        for unit in semantic_units:
            unit_length = len(unit['text'])
            
            # Check if adding this unit would exceed max size
            if current_length + unit_length > MAX_CHUNK_SIZE and current_chunk:
                # Save current chunk
                chunk_text = ' '.join([u['text'] for u in current_chunk])
                chunks.append({
                    'text': chunk_text,
                    'type': 'semantic',
                    'units': len(current_chunk),
                    'entities': [e for u in current_chunk for e in u.get('entities', [])]
                })
                
                # Start new chunk
                current_chunk = [unit]
                current_length = unit_length
            else:
                current_chunk.append(unit)
                current_length += unit_length
                
                # If chunk is large enough, check for natural break
                if current_length >= MIN_CHUNK_SIZE:
                    # Look for paragraph breaks or section boundaries
                    if (unit['text'].endswith(('.', '!', '?')) and 
                        len(current_chunk) >= 3):
                        
                        chunk_text = ' '.join([u['text'] for u in current_chunk])
                        chunks.append({
                            'text': chunk_text,
                            'type': 'semantic',
                            'units': len(current_chunk),
                            'entities': [e for u in current_chunk for e in u.get('entities', [])]
                        })
                        
                        current_chunk = []
                        current_length = 0
        
        # Handle remaining chunk
        if current_chunk and current_length >= MIN_CHUNK_SIZE:
            chunk_text = ' '.join([u['text'] for u in current_chunk])
            chunks.append({
                'text': chunk_text,
                'type': 'semantic',
                'units': len(current_chunk),
                'entities': [e for u in current_chunk for e in u.get('entities', [])]
            })
        
        return chunks
    
    def create_topic_based_chunks(self, text: str) -> List[Dict]:
        """Create chunks based on topic changes using TF-IDF similarity"""
        try:
            sentences = sent_tokenize(text)
        except Exception as e:
            logger.warning(f"NLTK sentence tokenization failed in topic chunking: {e}. Using simple fallback.")
            sentences = re.split(r'[.!?]+\s+', text)
        
        if len(sentences) < 5:
            return []
        
        # Filter valid sentences
        valid_sentences = [s for s in sentences if len(s.strip()) >= SENTENCE_MIN_LENGTH]
        
        if len(valid_sentences) < 3:
            return []
        
        try:
            # Create TF-IDF vectors
            tfidf_matrix = self.tfidf_vectorizer.fit_transform(valid_sentences)
            
            # Calculate similarity between consecutive sentences
            similarities = []
            for i in range(len(valid_sentences) - 1):
                sim = cosine_similarity(
                    tfidf_matrix[i:i+1], 
                    tfidf_matrix[i+1:i+2]
                )[0][0]
                similarities.append(sim)
            
            # Find topic boundaries (low similarity points)
            boundaries = [0]  # Start with first sentence
            
            for i, sim in enumerate(similarities):
                if sim < SEMANTIC_THRESHOLD:  # Topic change detected
                    boundaries.append(i + 1)
            
            boundaries.append(len(valid_sentences))  # End with last sentence
            
            # Create chunks from boundaries
            chunks = []
            for i in range(len(boundaries) - 1):
                start_idx = boundaries[i]
                end_idx = boundaries[i + 1]
                
                chunk_sentences = valid_sentences[start_idx:end_idx]
                chunk_text = ' '.join(chunk_sentences)
                
                if MIN_CHUNK_SIZE <= len(chunk_text) <= MAX_CHUNK_SIZE:
                    chunks.append({
                        'text': chunk_text,
                        'type': 'topic_based',
                        'sentence_count': len(chunk_sentences),
                        'topic_coherence': np.mean(similarities[start_idx:end_idx-1]) if end_idx > start_idx + 1 else 1.0
                    })
            
            return chunks
            
        except Exception as e:
            logger.warning(f"Topic-based chunking failed: {e}")
            return []
    
    def create_structured_chunks(self, text: str) -> List[Dict]:
        """Create chunks based on document structure (headings, sections, tables)"""
        chunks = []
        
        # Look for structured elements
        sections = re.split(r'\n\s*(?:=+\s*|#+\s*|SECTION|Chapter|Part\s+\d+)', text, flags=re.IGNORECASE)
        
        for section in sections:
            section = section.strip()
            if len(section) < MIN_CHUNK_SIZE:
                continue
            
            # Further split large sections
            if len(section) > MAX_CHUNK_SIZE:
                # Split by paragraphs
                paragraphs = re.split(r'\n\s*\n', section)
                current_chunk = ""
                
                for para in paragraphs:
                    para = para.strip()
                    if not para:
                        continue
                    
                    if len(current_chunk) + len(para) > MAX_CHUNK_SIZE:
                        if current_chunk and len(current_chunk) >= MIN_CHUNK_SIZE:
                            chunks.append({
                                'text': current_chunk.strip(),
                                'type': 'structured',
                                'structure_type': 'section'
                            })
                        current_chunk = para
                    else:
                        current_chunk += "\n\n" + para if current_chunk else para
                
                # Add remaining chunk
                if current_chunk and len(current_chunk) >= MIN_CHUNK_SIZE:
                    chunks.append({
                        'text': current_chunk.strip(),
                        'type': 'structured',
                        'structure_type': 'section'
                    })
            else:
                chunks.append({
                    'text': section,
                    'type': 'structured',
                    'structure_type': 'section'
                })
        
        return chunks
    
    def create_chunks_parallel(self, text_segments: List[str], chunk_method: str) -> List[Dict]:
        """Create chunks using parallel processing for different text segments"""
        if len(text_segments) <= 1:
            # Not worth parallelizing for small inputs
            if text_segments:
                return getattr(self, f'create_{chunk_method}_chunks')(text_segments[0])
            return []
        
        all_chunks = []
        max_workers = min(MAX_WORKERS, len(text_segments))
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit chunking tasks for each text segment
            future_to_segment = {
                executor.submit(getattr(self, f'create_{chunk_method}_chunks'), segment): i
                for i, segment in enumerate(text_segments)
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_segment):
                segment_idx = future_to_segment[future]
                try:
                    chunks = future.result()
                    if chunks:
                        # Add segment information to chunks
                        for chunk in chunks:
                            chunk['segment_idx'] = segment_idx
                        all_chunks.extend(chunks)
                        logger.info(f"Completed parallel chunking for segment {segment_idx + 1}")
                except Exception as e:
                    logger.error(f"Parallel chunking failed for segment {segment_idx}: {str(e)}")
        
        return all_chunks
    
    def merge_and_rank_chunks(self, chunk_sets: List[List[Dict]]) -> List[Dict]:
        """Merge chunks from different strategies and rank by quality"""
        all_chunks = []
        
        # Flatten all chunk sets
        for chunk_set in chunk_sets:
            all_chunks.extend(chunk_set)
        
        # Remove duplicates and very similar chunks
        unique_chunks = []
        seen_texts = set()
        
        for chunk in all_chunks:
            text = chunk['text']
            
            # Create a normalized version for comparison
            normalized = re.sub(r'\s+', ' ', text.lower().strip())
            
            # Check for exact duplicates
            if normalized in seen_texts:
                continue
            
            # Check for high overlap with existing chunks
            is_similar = False
            for seen_text in seen_texts:
                # Simple overlap check
                words1 = set(normalized.split())
                words2 = set(seen_text.split())
                
                if len(words1.intersection(words2)) / len(words1.union(words2)) > 0.8:
                    is_similar = True
                    break
            
            if not is_similar:
                seen_texts.add(normalized)
                unique_chunks.append(chunk)
        
        # Rank chunks by quality metrics
        for chunk in unique_chunks:
            score = self.calculate_chunk_quality(chunk)
            chunk['quality_score'] = score
        
        # Sort by quality and return
        unique_chunks.sort(key=lambda x: x['quality_score'], reverse=True)
        
        return unique_chunks
    
    def calculate_chunk_quality(self, chunk: Dict) -> float:
        """Calculate quality score for a chunk"""
        text = chunk['text']
        score = 0.0
        
        # Length score (prefer moderate length)
        length = len(text)
        if MIN_CHUNK_SIZE <= length <= MAX_CHUNK_SIZE:
            # Optimal length gets higher score
            optimal_length = (MIN_CHUNK_SIZE + MAX_CHUNK_SIZE) // 2
            length_score = 1.0 - abs(length - optimal_length) / optimal_length
            score += length_score * 0.3
        
        # Sentence completeness score
        try:
            sentences = sent_tokenize(text)
        except Exception:
            sentences = re.split(r'[.!?]+\s+', text)
        
        complete_sentences = sum(1 for s in sentences if s.strip().endswith(('.', '!', '?')))
        if sentences:
            completeness_score = complete_sentences / len(sentences)
            score += completeness_score * 0.2
        
        # Information density score
        try:
            words = word_tokenize(text.lower())
        except Exception:
            # Simple fallback word tokenization
            words = re.findall(r'\b\w+\b', text.lower())
        
        unique_words = set(words) - self.stop_words
        if words:
            density_score = len(unique_words) / len(words)
            score += density_score * 0.2
        
        # Structure bonus
        structure_bonus = {
            'semantic': 0.3,
            'topic_based': 0.25,
            'structured': 0.2,
            'sliding_window': 0.1
        }
        score += structure_bonus.get(chunk.get('type', 'sliding_window'), 0.1)
        
        # Entity bonus (if available)
        entities = chunk.get('entities', [])
        if entities:
            score += min(len(entities) * 0.05, 0.2)
        
        return score

def get_enhanced_chunks(text: str) -> List[Dict]:
    """Main function to get enhanced chunks using multiple strategies with parallel processing"""
    start_time = time.time()
    logger.info("Starting enhanced chunking with parallel processing")
    
    chunker = EnhancedChunker()
    
    # Preprocess text
    text = chunker.preprocess_text(text)
    
    if len(text) < MIN_CHUNK_SIZE:
        logger.warning("Text too short for meaningful chunking")
        return [{'text': text, 'type': 'single', 'quality_score': 0.5}]
    
    # Split text into segments for parallel processing if it's large enough
    text_segments = []
    if len(text) > MAX_CHUNK_SIZE * 10:  # Only parallelize for large texts
        # Split by double newlines (paragraphs) or sections
        segments = re.split(r'\n\s*\n|===|---', text)
        segments = [seg.strip() for seg in segments if len(seg.strip()) > MIN_CHUNK_SIZE]
        
        if len(segments) > 1:
            text_segments = segments
            logger.info(f"Split large text into {len(text_segments)} segments for parallel processing")
    
    # If no segments or text is not large enough, use single text
    if not text_segments:
        text_segments = [text]
    
    # Apply multiple chunking strategies with optional parallelization
    chunk_sets = []
    
    # 1. Sliding window chunks (baseline) - with parallel processing for large texts
    if len(text_segments) > 1:
        sliding_chunks = chunker.create_chunks_parallel(text_segments, 'sliding_window')
    else:
        sliding_chunks = chunker.create_sliding_window_chunks(text)
    
    if sliding_chunks:
        chunk_sets.append(sliding_chunks)
        logger.info(f"Created {len(sliding_chunks)} sliding window chunks")
    
    # 2. Semantic chunks - with parallel processing for large texts
    if len(text_segments) > 1:
        semantic_chunks = chunker.create_chunks_parallel(text_segments, 'semantic')
    else:
        semantic_chunks = chunker.create_semantic_chunks(text)
    
    if semantic_chunks:
        chunk_sets.append(semantic_chunks)
        logger.info(f"Created {len(semantic_chunks)} semantic chunks")
    
    # 3. Topic-based chunks - with parallel processing for large texts
    if len(text_segments) > 1:
        topic_chunks = chunker.create_chunks_parallel(text_segments, 'topic_based')
    else:
        topic_chunks = chunker.create_topic_based_chunks(text)
    
    if topic_chunks:
        chunk_sets.append(topic_chunks)
        logger.info(f"Created {len(topic_chunks)} topic-based chunks")
    
    # 4. Structured chunks - with parallel processing for large texts
    if len(text_segments) > 1:
        structured_chunks = chunker.create_chunks_parallel(text_segments, 'structured')
    else:
        structured_chunks = chunker.create_structured_chunks(text)
    
    if structured_chunks:
        chunk_sets.append(structured_chunks)
        logger.info(f"Created {len(structured_chunks)} structured chunks")
    
    # Merge and rank chunks
    if chunk_sets:
        final_chunks = chunker.merge_and_rank_chunks(chunk_sets)
    else:
        # Fallback to simple chunking
        final_chunks = chunker.create_sliding_window_chunks(text)
    
    logger.info(f"Enhanced chunking with parallel processing completed in {time.time() - start_time:.2f}s. "
                f"Created {len(final_chunks)} high-quality chunks")
    
    return final_chunks
