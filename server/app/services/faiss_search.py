import logging
import time
from typing import List, Dict, Tuple, Optional, Any
import numpy as np
import faiss
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

from app.config.settings import (
    FAISS_INDEX_TYPE, FAISS_NPROBE, FAISS_K_SEARCH,
    KEYWORD_BOOST_FACTOR, SEMANTIC_SIMILARITY_THRESHOLD,
    CONTEXT_WINDOW_SIZE
)

logger = logging.getLogger(__name__)

class FAISSSearchService:
    """Advanced FAISS-based search service with hybrid retrieval"""
    
    def __init__(self):
        self.index = None
        self.chunks = []
        self.embeddings = []
        self.dimension = None
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 3),
            lowercase=True
        )
        self.chunk_tfidf_matrix = None
        self.is_trained = False
        
        # Important terms for domain-specific boosting - expanded
        self.important_terms = {
            # Core insurance terms
            'coverage', 'limit', 'deductible', 'premium', 'claim', 'benefit', 'exclusion',
            'copay', 'coinsurance', 'maximum', 'minimum', 'annual', 'lifetime', 'policy',
            'insured', 'covered', 'eligible', 'amount', 'percentage', 'network', 'provider',
            'emergency', 'prescription', 'medical', 'dental', 'vision', 'mental', 'health',
            'hospital', 'outpatient', 'inpatient', 'surgery', 'diagnostic', 'preventive',
            'waiting', 'period', 'authorization', 'existing', 'condition', 'copayment',
            'out-of-pocket', 'pre-existing', 'reimbursement', 'approval', 'referral',
            
            # Specific medical procedures and conditions
            'cataract', 'surgery', 'surgical', 'operation', 'treatment', 'therapy',
            'consultation', 'diagnosis', 'root', 'canal', 'dental', 'orthodontic',
            'implant', 'prosthetic', 'hospitalization', 'admission', 'discharge',
            
            # Insurance specific terms
            'settlement', 'settled', 'settle', 'processing', 'processed', 'approved',
            'rejected', 'pending', 'timeline', 'timeframe', 'days', 'working',
            'business', 'cashless', 'reimbursement', 'network', 'empanelled',
            
            # Financial terms
            'sum', 'insured', 'basic', 'addon', 'add-on', 'sub-limit', 'sublimit',
            'co-pay', 'copay', 'deductible', 'excess', 'loadings', 'discount',
            'inr', 'rupees', 'rs', 'lakh', 'lakhs', 'thousand', 'crore',
            
            # Common insurance companies and terms
            'hdfc', 'ergo', 'star', 'health', 'care', 'medicare', 'mediclaim',
            'family', 'floater', 'individual', 'group', 'corporate', 'retail'
        }
    
    def create_index(self, embeddings: List[List[float]], chunks: List[Dict], 
                    index_type: str = None) -> None:
        """Create and train FAISS index with embeddings and chunks"""
        start_time = time.time()
        logger.info(f"Creating FAISS index with {len(embeddings)} embeddings")
        
        if not embeddings or not chunks:
            raise ValueError("Embeddings and chunks cannot be empty")
        
        if len(embeddings) != len(chunks):
            raise ValueError("Embeddings and chunks must have the same length")
        
        # Convert embeddings to numpy array
        embeddings_array = np.array(embeddings, dtype=np.float32)
        self.dimension = embeddings_array.shape[1]
        self.embeddings = embeddings_array
        self.chunks = chunks
        
        # Choose index type based on data size and requirements
        index_type = index_type or FAISS_INDEX_TYPE
        n_vectors = len(embeddings)
        
        if n_vectors < 1000:
            # Use exact search for small datasets
            self.index = faiss.IndexFlatL2(self.dimension)
            logger.info("Using IndexFlatL2 for exact search (small dataset)")
        elif n_vectors < 10000:
            # Use IVF with reasonable number of clusters
            nlist = min(100, int(np.sqrt(n_vectors)))
            quantizer = faiss.IndexFlatL2(self.dimension)
            self.index = faiss.IndexIVFFlat(quantizer, self.dimension, nlist)
            logger.info(f"Using IndexIVFFlat with {nlist} clusters")
        else:
            # Use IVF-PQ for large datasets
            nlist = min(500, int(np.sqrt(n_vectors)))
            m = 8  # Number of sub-quantizers
            quantizer = faiss.IndexFlatL2(self.dimension)
            self.index = faiss.IndexIVFPQ(quantizer, self.dimension, nlist, m, 8)
            logger.info(f"Using IndexIVFPQ with {nlist} clusters and {m} sub-quantizers")
        
        # Train index if needed
        if hasattr(self.index, 'train'):
            logger.info("Training FAISS index...")
            self.index.train(embeddings_array)
            self.index.nprobe = min(FAISS_NPROBE, self.index.nlist if hasattr(self.index, 'nlist') else 10)
        
        # Add vectors to index
        self.index.add(embeddings_array)
        self.is_trained = True
        
        # Create TF-IDF matrix for keyword search
        chunk_texts = [chunk['text'] for chunk in chunks]
        try:
            self.chunk_tfidf_matrix = self.tfidf_vectorizer.fit_transform(chunk_texts)
            logger.info("Created TF-IDF matrix for hybrid search")
        except Exception as e:
            logger.warning(f"Failed to create TF-IDF matrix: {e}")
            self.chunk_tfidf_matrix = None
        
        logger.info(f"FAISS index created in {time.time() - start_time:.2f}s. "
                   f"Index size: {self.index.ntotal} vectors")
    
    def search(self, query_embedding: List[float], k: int = None) -> List[Dict]:
        """Search for similar chunks using FAISS"""
        if not self.is_trained:
            raise ValueError("Index not trained. Call create_index first.")
        
        k = k or FAISS_K_SEARCH
        k = min(k, len(self.chunks))  # Don't search for more chunks than available
        
        start_time = time.time()
        
        # Convert query embedding to numpy array
        query_array = np.array([query_embedding], dtype=np.float32)
        
        # Search FAISS index
        distances, indices = self.index.search(query_array, k)
        
        # Prepare results
        results = []
        for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
            if idx != -1:  # Valid index
                similarity_score = 1.0 / (1.0 + distance)  # Convert distance to similarity
                results.append({
                    'chunk': self.chunks[idx],
                    'similarity_score': float(similarity_score),
                    'distance': float(distance),
                    'rank': i + 1
                })
        
        logger.info(f"FAISS search completed in {time.time() - start_time:.3f}s. "
                   f"Found {len(results)} results")
        
        return results
    
    def hybrid_search(self, query_embedding: List[float], query_text: str, 
                     k: int = None) -> List[Dict]:
        """Combine FAISS semantic search with TF-IDF keyword search"""
        if not self.is_trained:
            raise ValueError("Index not trained. Call create_index first.")
        
        k = k or FAISS_K_SEARCH
        
        # Get semantic search results
        semantic_results = self.search(query_embedding, k * 2)  # Get more for reranking
        
        # Get keyword search results if TF-IDF is available
        keyword_results = []
        if self.chunk_tfidf_matrix is not None:
            try:
                query_tfidf = self.tfidf_vectorizer.transform([query_text])
                
                # Calculate TF-IDF similarities
                tfidf_similarities = cosine_similarity(query_tfidf, self.chunk_tfidf_matrix).flatten()
                
                # Get top keyword matches
                top_indices = np.argsort(tfidf_similarities)[::-1][:k]
                
                for idx in top_indices:
                    if tfidf_similarities[idx] > 0.01:  # Minimum similarity threshold
                        keyword_results.append({
                            'chunk': self.chunks[idx],
                            'tfidf_score': float(tfidf_similarities[idx]),
                            'chunk_index': int(idx)
                        })
                        
            except Exception as e:
                logger.warning(f"Keyword search failed: {e}")
        
        # Combine and rerank results
        combined_results = self._combine_search_results(
            semantic_results, keyword_results, query_text, k
        )
        
        return combined_results
    
    def _combine_search_results(self, semantic_results: List[Dict], 
                               keyword_results: List[Dict], query_text: str,
                               k: int) -> List[Dict]:
        """Combine semantic and keyword search results with intelligent scoring"""
        
        # Create a mapping of chunk indices to results
        result_map = {}
        
        # Add semantic results
        for result in semantic_results:
            chunk_idx = self.chunks.index(result['chunk'])
            result_map[chunk_idx] = {
                'chunk': result['chunk'],
                'semantic_score': result['similarity_score'],
                'semantic_rank': result['rank'],
                'tfidf_score': 0.0,
                'final_score': 0.0
            }
        
        # Add/update with keyword results
        for result in keyword_results:
            chunk_idx = result['chunk_index']
            if chunk_idx in result_map:
                result_map[chunk_idx]['tfidf_score'] = result['tfidf_score']
            else:
                result_map[chunk_idx] = {
                    'chunk': result['chunk'],
                    'semantic_score': 0.0,
                    'semantic_rank': len(semantic_results) + 1,
                    'tfidf_score': result['tfidf_score'],
                    'final_score': 0.0
                }
        
        # Calculate final scores
        query_lower = query_text.lower()
        key_terms = self._extract_key_terms(query_lower)
        
        for chunk_idx, result in result_map.items():
            chunk_text = result['chunk']['text'].lower()
            
            # Base scores
            semantic_score = result['semantic_score']
            tfidf_score = result['tfidf_score']
            
            # Keyword matching bonus
            keyword_bonus = self._calculate_keyword_bonus(chunk_text, key_terms)
            
            # Domain-specific term bonus
            domain_bonus = self._calculate_domain_bonus(chunk_text, query_lower)
            
            # Exact phrase matching bonus
            phrase_bonus = self._calculate_phrase_bonus(chunk_text, query_lower)
            
            # Numeric matching bonus (important for policy documents)
            numeric_bonus = self._calculate_numeric_bonus(chunk_text, query_lower)
            
            # Balanced scoring prioritizing semantic understanding
            final_score = (
                semantic_score * 0.55 +     # Prioritize semantic similarity
                tfidf_score * 0.25 +        # TF-IDF for content relevance
                keyword_bonus * 0.12 +      # Reduced keyword hunting
                domain_bonus * 0.05 +       # Minimal domain boosting
                phrase_bonus * 0.02 +       # Reduced phrase matching
                numeric_bonus * 0.01        # Minimal numeric matching
            )
            
            # Quality bonus for semantic coherence
            if semantic_score > 0.4:  # High semantic similarity
                final_score *= 1.15
            
            result['final_score'] = final_score
            result['keyword_bonus'] = keyword_bonus
            result['domain_bonus'] = domain_bonus
            result['phrase_bonus'] = phrase_bonus
            result['numeric_bonus'] = numeric_bonus
        
        # Sort by final score and return top k
        sorted_results = sorted(result_map.values(), key=lambda x: x['final_score'], reverse=True)
        
        return sorted_results[:k]
    
    def _extract_key_terms(self, query_text: str) -> List[str]:
        """Extract important terms from query - more comprehensive"""
        original_query = query_text
        
        # Remove common question words but keep important ones
        query_text = re.sub(r'\b(what|when|where|who|why|how|is|are|do|does|did|will|would)\b', 
                           '', query_text)
        
        # Extract ALL meaningful words (2+ characters for broader coverage)
        words = re.findall(r'\b\w{2,}\b', query_text)
        
        # Add numbers and currency (more patterns)
        numbers = re.findall(r'\b\d+(?:,\d{3})*(?:\.\d+)?%?\b', query_text)
        currency = re.findall(r'(?:\$|INR|Rs\.?)[\s]*[\d,]+(?:\.\d{2})?\b', query_text)
        
        # More comprehensive phrase patterns
        phrases = re.findall(r'\b(?:out of pocket|prior authorization|pre existing|waiting period|per year|per day|sum insured|basic si|add[- ]on|co[- ]pay|deductible|settlement|claim|coverage|treatment|surgery|cataract|root canal|hdfc)\b', 
                           query_text, re.IGNORECASE)
        
        # Add partial word matches for important terms
        important_partials = []
        for word in words:
            if len(word) >= 4:  # Only for longer words
                if any(term in word.lower() for term in ['claim', 'cover', 'treat', 'surg', 'pay']):
                    important_partials.append(word)
        
        # Combine all terms and remove duplicates
        all_terms = list(set(words + numbers + currency + phrases + important_partials))
        
        # Also add terms from original query for exact matching
        original_words = re.findall(r'\b\w{3,}\b', original_query)
        all_terms.extend(original_words)
        
        return list(set(all_terms))
    
    def _calculate_keyword_bonus(self, chunk_text: str, key_terms: List[str]) -> float:
        """Calculate bonus for keyword matches"""
        if not key_terms:
            return 0.0
        
        exact_matches = 0
        partial_matches = 0
        
        for term in key_terms:
            term_lower = term.lower()
            if f' {term_lower} ' in f' {chunk_text} ':
                exact_matches += 1
            elif term_lower in chunk_text:
                partial_matches += 1
        
        # Calculate bonus
        exact_bonus = (exact_matches / len(key_terms)) * 0.3
        partial_bonus = (partial_matches / len(key_terms)) * 0.1
        
        return exact_bonus + partial_bonus
    
    def _calculate_domain_bonus(self, chunk_text: str, query_text: str) -> float:
        """Calculate bonus for domain-specific terms"""
        query_terms = set(re.findall(r'\b\w+\b', query_text))
        important_in_query = query_terms.intersection(self.important_terms)
        
        if not important_in_query:
            return 0.0
        
        matches = 0
        for term in important_in_query:
            if term in chunk_text:
                matches += 1
        
        return (matches / len(important_in_query)) * 0.2
    
    def _calculate_phrase_bonus(self, chunk_text: str, query_text: str) -> float:
        """Calculate bonus for exact phrase matches"""
        # Extract phrases of 2-4 words from query
        words = query_text.split()
        phrases = []
        
        for i in range(len(words)):
            for length in range(2, min(5, len(words) - i + 1)):
                phrase = ' '.join(words[i:i+length])
                if len(phrase) > 5:  # Minimum phrase length
                    phrases.append(phrase)
        
        if not phrases:
            return 0.0
        
        matches = sum(1 for phrase in phrases if phrase in chunk_text)
        return (matches / len(phrases)) * 0.15
    
    def _calculate_numeric_bonus(self, chunk_text: str, query_text: str) -> float:
        """Calculate bonus for matching numbers (important for policy documents)"""
        query_numbers = set(re.findall(r'\b\d+(?:\.\d+)?%?\b', query_text))
        query_currency = set(re.findall(r'\$[\d,]+(?:\.\d{2})?\b', query_text))
        
        all_query_nums = query_numbers.union(query_currency)
        
        if not all_query_nums:
            return 0.0
        
        matches = sum(1 for num in all_query_nums if num in chunk_text)
        return (matches / len(all_query_nums)) * 0.25
    
    def multi_tier_search(self, query_embedding: List[float], query_text: str, k: int = None) -> List[Dict]:
        """Multi-tier search prioritizing semantic understanding"""
        k = k or FAISS_K_SEARCH
        
        # Tier 1: Pure semantic search (prioritize understanding)
        semantic_results = self.search(query_embedding, k)
        
        # Check if we have good semantic matches
        if semantic_results and semantic_results[0]['similarity_score'] > 0.25:
            logger.info(f"Using semantic search results (best score: {semantic_results[0]['similarity_score']:.3f})")
            return semantic_results
        
        logger.info("Semantic similarity low, trying hybrid approach")
        
        # Tier 2: Hybrid search for broader coverage
        hybrid_results = self.hybrid_search(query_embedding, query_text, k)
        
        # If hybrid gives us better results, use those
        if hybrid_results and hybrid_results[0]['final_score'] > 0.2:
            return hybrid_results
        
        logger.info("All searches had low scores, using best available results")
        
        # Return the best results we found, preferring semantic over hybrid
        if semantic_results:
            return semantic_results
        elif hybrid_results:
            return hybrid_results
        else:
            return []
    
    def enhance_results(self, search_results: List[Dict], query_text: str) -> List[str]:
        """Enhance search results for better context formatting"""
        enhanced_chunks = []
        
        for i, result in enumerate(search_results):
            chunk = result['chunk']
            chunk_text = chunk['text']
            
            # Add context markers
            context_marker = f"[CONTEXT {i+1}]"
            
            # Add metadata if available
            metadata_parts = []
            if 'type' in chunk:
                metadata_parts.append(f"Type: {chunk['type']}")
            if 'quality_score' in chunk:
                metadata_parts.append(f"Quality: {chunk['quality_score']:.2f}")
            if 'final_score' in result:
                metadata_parts.append(f"Relevance: {result['final_score']:.2f}")
            
            # Format enhanced chunk
            if metadata_parts:
                enhanced_text = f"{context_marker} ({', '.join(metadata_parts)}) {chunk_text}"
            else:
                enhanced_text = f"{context_marker} {chunk_text}"
            
            enhanced_chunks.append(enhanced_text)
        
        return enhanced_chunks
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the search index"""
        if not self.is_trained:
            return {"status": "not_trained"}
        
        stats = {
            "status": "trained",
            "total_vectors": self.index.ntotal,
            "dimension": self.dimension,
            "index_type": type(self.index).__name__,
            "total_chunks": len(self.chunks),
            "tfidf_available": self.chunk_tfidf_matrix is not None
        }
        
        if hasattr(self.index, 'nlist'):
            stats["nlist"] = self.index.nlist
        if hasattr(self.index, 'nprobe'):
            stats["nprobe"] = self.index.nprobe
            
        return stats
