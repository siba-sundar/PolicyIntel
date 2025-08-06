"""
Enhanced Search Module - OPTIONAL
This module provides additional search capabilities without modifying the existing system.
Can be enabled via environment variable: USE_ENHANCED_SEARCH=true
"""

import logging
import os
import re
from typing import List, Dict, Optional
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

class EnhancedSearchStrategy:
    """
    Additional search strategy that can be optionally enabled.
    Does NOT replace existing search - adds to it.
    """
    
    def __init__(self):
        self.enabled = os.getenv("USE_ENHANCED_SEARCH", "false").lower() == "true"
        if self.enabled:
            logger.info("🔍 ENHANCED SEARCH ENABLED - Using additional search strategies")
        
        # More comprehensive term extraction for academic/technical documents
        self.academic_terms = {
            'newton', 'force', 'gravity', 'gravitation', 'motion', 'law', 'laws',
            'mass', 'particle', 'attraction', 'inverse', 'square', 'distance',
            'orbit', 'orbital', 'planet', 'planetary', 'moon', 'celestial',
            'kepler', 'ellipse', 'elliptical', 'centripetal', 'universal',
            'principia', 'calculus', 'fluxionary', 'mathematics', 'proof',
            'derive', 'derivation', 'demonstrate', 'explanation', 'theory',
            'space', 'time', 'absolute', 'relative', 'mechanics', 'physics'
        }
    
    def extract_key_concepts(self, question: str) -> List[str]:
        """Extract key concepts from question more comprehensively"""
        if not self.enabled:
            return []
        
        question_lower = question.lower()
        concepts = []
        
        # Extract academic/scientific terms
        words = set(re.findall(r'\b\w{3,}\b', question_lower))
        academic_matches = words.intersection(self.academic_terms)
        concepts.extend(list(academic_matches))
        
        # Extract quoted phrases
        quoted = re.findall(r'"([^"]*)"', question_lower)
        concepts.extend(quoted)
        
        # Extract specific patterns for Newton questions
        patterns = [
            r'\b(three laws?)\b',
            r'\b(inverse square)\b', 
            r'\b(universal gravitation)\b',
            r'\b(centripetal force)\b',
            r'\b(orbital motion)\b',
            r'\b(absolute space)\b',
            r'\b(relative motion)\b',
            r'\b(quantity of motion)\b'
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, question_lower)
            concepts.extend(matches)
        
        return list(set(concepts))
    
    def create_expanded_queries(self, original_question: str) -> List[str]:
        """Create additional query variations to catch more relevant content"""
        if not self.enabled:
            return [original_question]
        
        queries = [original_question]  # Always include original
        question_lower = original_question.lower()
        
        # For Newton-specific questions, create variations
        variations = []
        
        # If asking about laws, try different phrasings
        if 'laws' in question_lower or 'law' in question_lower:
            variations.extend([
                question_lower.replace('laws of motion', 'motion laws'),
                question_lower.replace('three laws', 'first second third law'),
                re.sub(r'\blaw\b', 'principle', question_lower)
            ])
        
        # If asking about gravity/gravitation
        if any(word in question_lower for word in ['gravity', 'gravitation', 'force']):
            variations.extend([
                question_lower.replace('gravity', 'gravitational force'),
                question_lower.replace('gravitation', 'attractive force'),
                question_lower.replace('force', 'attraction')
            ])
        
        # If asking about derivation/explanation
        if any(word in question_lower for word in ['derive', 'explain', 'demonstrate']):
            variations.extend([
                question_lower.replace('derive', 'show'),
                question_lower.replace('explain', 'describe'),
                question_lower.replace('demonstrate', 'prove')
            ])
        
        # Add meaningful variations
        for var in variations:
            if var.strip() and var != original_question.lower():
                queries.append(var.strip())
        
        return queries[:5]  # Limit to prevent too many queries
    
    def enhanced_chunk_scoring(self, chunks: List[Dict], question: str) -> List[Dict]:
        """
        Enhanced scoring that doesn't replace existing scores but adds additional signals
        """
        if not self.enabled or not chunks:
            return chunks
        
        key_concepts = self.extract_key_concepts(question)
        if not key_concepts:
            return chunks
        
        # Add enhanced scores to existing chunks
        enhanced_chunks = []
        for chunk in chunks:
            enhanced_chunk = chunk.copy()  # Don't modify original
            
            chunk_text = chunk.get('text', '').lower()
            
            # Concept matching score
            concept_matches = sum(1 for concept in key_concepts if concept in chunk_text)
            concept_score = concept_matches / len(key_concepts) if key_concepts else 0
            
            # Contextual proximity score (how close key terms are to each other)
            proximity_score = self._calculate_proximity_score(chunk_text, key_concepts)
            
            # Add enhanced score (don't replace existing scores)
            enhanced_score = concept_score * 0.4 + proximity_score * 0.3
            enhanced_chunk['enhanced_score'] = enhanced_score
            
            # Boost overall relevance if enhanced score is high
            if enhanced_score > 0.3:
                if 'final_score' in enhanced_chunk:
                    enhanced_chunk['final_score'] *= 1.2  # Boost existing score
                elif 'similarity_score' in enhanced_chunk:
                    enhanced_chunk['similarity_score'] *= 1.15  # Boost existing score
            
            enhanced_chunks.append(enhanced_chunk)
        
        # Sort by enhanced relevance while preserving original ranking logic
        enhanced_chunks.sort(key=lambda x: (
            x.get('enhanced_score', 0) * 0.3 + 
            x.get('final_score', x.get('similarity_score', 0)) * 0.7
        ), reverse=True)
        
        return enhanced_chunks
    
    def _calculate_proximity_score(self, text: str, concepts: List[str]) -> float:
        """Calculate how close key concepts are to each other in the text"""
        if len(concepts) < 2:
            return 0.0
        
        # Find positions of concepts
        concept_positions = {}
        for concept in concepts:
            positions = [m.start() for m in re.finditer(re.escape(concept), text)]
            if positions:
                concept_positions[concept] = positions
        
        if len(concept_positions) < 2:
            return 0.0
        
        # Calculate average distance between concept pairs
        distances = []
        concept_items = list(concept_positions.items())
        
        for i in range(len(concept_items)):
            for j in range(i + 1, len(concept_items)):
                concept1_positions = concept_items[i][1]
                concept2_positions = concept_items[j][1]
                
                # Find minimum distance between any occurrence of the two concepts
                min_distance = float('inf')
                for pos1 in concept1_positions:
                    for pos2 in concept2_positions:
                        distance = abs(pos1 - pos2)
                        min_distance = min(min_distance, distance)
                
                if min_distance != float('inf'):
                    distances.append(min_distance)
        
        if not distances:
            return 0.0
        
        # Convert to proximity score (closer = higher score)
        avg_distance = sum(distances) / len(distances)
        max_reasonable_distance = 500  # characters
        proximity_score = max(0, 1 - (avg_distance / max_reasonable_distance))
        
        return proximity_score
    
    def should_expand_search(self, search_results: List[Dict], question: str) -> bool:
        """
        Determine if we should try additional search strategies
        """
        if not self.enabled:
            return False
        
        if not search_results:
            return True
        
        # Check if best result has low confidence
        best_score = 0
        if search_results:
            first_result = search_results[0]
            best_score = first_result.get('final_score', 
                          first_result.get('similarity_score', 0))
        
        # If best score is low, try enhanced search
        return best_score < 0.25
    
    def log_enhancement_usage(self, original_count: int, enhanced_count: int, question: str):
        """Log when enhanced search provides better results"""
        if self.enabled and enhanced_count > original_count:
            logger.info(f"🎯 Enhanced search improved results: {original_count} → {enhanced_count} relevant chunks for question: {question[:50]}...")


def apply_enhanced_search_if_enabled(search_results: List[Dict], question: str) -> List[Dict]:
    """
    Apply enhanced search strategies if enabled via environment variable.
    This function can be called from existing code without breaking anything.
    """
    enhancer = EnhancedSearchStrategy()
    
    if not enhancer.enabled:
        return search_results  # Return unchanged if not enabled
    
    # Apply enhancements
    original_count = len([r for r in search_results if r.get('final_score', r.get('similarity_score', 0)) > 0.2])
    enhanced_results = enhancer.enhanced_chunk_scoring(search_results, question)
    enhanced_count = len([r for r in enhanced_results if r.get('final_score', r.get('similarity_score', 0)) > 0.2])
    
    enhancer.log_enhancement_usage(original_count, enhanced_count, question)
    
    return enhanced_results
