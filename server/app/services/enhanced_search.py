# app/services/enhanced_search.py
import logging
import re
from typing import List, Dict, Any, Optional
from app.processors.enhanced_pdf_processor import ExtractedLink

logger = logging.getLogger(__name__)

class EnhancedSearchService:
    """Enhanced search service that incorporates link context and fetched content"""
    
    def __init__(self):
        self.search_enhancement_enabled = True
        
    def apply_enhanced_search_if_enabled(self, search_results: List[Dict], 
                                      question: str, 
                                      search_context: Dict[str, Any] = None) -> List[Dict]:
        """
        Apply enhanced search using link context and fetched content
        
        Args:
            search_results: Original search results from FAISS
            question: The question being asked
            search_context: Additional context including extracted links and fetched content
        """
        
        if not self.search_enhancement_enabled or not search_context:
            return search_results
        
        try:
            extracted_links = search_context.get('extracted_links', [])
            fetched_content = search_context.get('fetched_content', {})
            
            if not extracted_links and not fetched_content:
                return search_results
            
            logger.info(f"🔍 Applying enhanced search with {len(extracted_links)} links "
                       f"and {len(fetched_content)} fetched items")
            
            # Enhance search results with link context
            enhanced_results = self._enhance_with_link_context(
                search_results, question, extracted_links, fetched_content
            )
            
            # Re-rank results considering link relevance
            reranked_results = self._rerank_with_link_relevance(
                enhanced_results, question, fetched_content
            )
            
            return reranked_results
            
        except Exception as e:
            logger.error(f"Enhanced search failed: {str(e)}")
            return search_results
    
    def _enhance_with_link_context(self, search_results: List[Dict], 
                                 question: str,
                                 extracted_links: List[ExtractedLink],
                                 fetched_content: Dict[str, Any]) -> List[Dict]:
        """Enhance search results by adding link context information"""
        
        enhanced_results = search_results.copy()
        question_keywords = set(question.lower().split())
        
        # For each search result, check if it references any links
        for result in enhanced_results:
            result_text = result.get('text', '').lower()
            link_references = []
            
            # Check if this result mentions any of the extracted links
            for link in extracted_links:
                if any(keyword in result_text for keyword in question_keywords):
                    # Check if link context is relevant
                    link_context = f"{link.context} {link.surrounding_text}".lower()
                    link_keywords = set(link_context.split())
                    
                    overlap = len(question_keywords.intersection(link_keywords))
                    if overlap > 0:
                        link_references.append({
                            'url': link.url,
                            'type': link.link_type,
                            'relevance_score': overlap / len(question_keywords),
                            'context': link.context[:200]
                        })
            
            # Add link references to the result
            if link_references:
                result['linked_references'] = link_references
                # Boost similarity score slightly for results with relevant links
                if 'similarity_score' in result:
                    boost = min(0.1, len(link_references) * 0.03)
                    result['similarity_score'] += boost
                    result['enhanced'] = True
        
        return enhanced_results
    
    def _rerank_with_link_relevance(self, search_results: List[Dict],
                                   question: str,
                                   fetched_content: Dict[str, Any]) -> List[Dict]:
        """Re-rank search results considering relevance of fetched link content"""
        
        if not fetched_content:
            return search_results
        
        question_keywords = set(question.lower().split())
        
        # Calculate link content relevance scores
        for result in search_results:
            link_boost = 0.0
            
            # Check if any fetched content is highly relevant to this question
            for url, content_data in fetched_content.items():
                content_text = content_data.get('content', '').lower()
                content_words = set(content_text.split())
                
                # Calculate relevance between question and fetched content
                overlap = len(question_keywords.intersection(content_words))
                if overlap > 1:  # Require at least 2 keyword matches
                    relevance = overlap / len(question_keywords)
                    link_boost = max(link_boost, relevance * 0.15)  # Max 15% boost
            
            # Apply boost to similarity score
            if link_boost > 0 and 'similarity_score' in result:
                result['similarity_score'] += link_boost
                result['link_boost'] = link_boost
        
        # Re-sort by enhanced similarity score
        return sorted(search_results, 
                     key=lambda x: x.get('similarity_score', 0), 
                     reverse=True)
    
    def get_relevant_fetched_content(self, question: str, 
                                   fetched_content: Dict[str, Any],
                                   max_items: int = 3) -> List[Dict[str, Any]]:
        """Get the most relevant fetched content for a specific question"""
        
        if not fetched_content:
            return []
        
        question_keywords = set(question.lower().split())
        relevance_scores = []
        
        for url, content_data in fetched_content.items():
            content_text = content_data.get('content', '').lower()
            content_words = set(content_text.split())
            
            # Calculate relevance score
            overlap = len(question_keywords.intersection(content_words))
            if overlap > 0:
                relevance = overlap / len(question_keywords)
                relevance_scores.append({
                    'url': url,
                    'content_data': content_data,
                    'relevance_score': relevance,
                    'keyword_matches': overlap
                })
        
        # Sort by relevance and return top items
        relevance_scores.sort(key=lambda x: x['relevance_score'], reverse=True)
        return relevance_scores[:max_items]
    
    def create_enhanced_prompt(self, question: str, 
                             regular_context: List[str],
                             relevant_links: List[Dict[str, Any]],
                             max_link_content: int = 500) -> str:
        """Create an enhanced prompt that includes both regular context and link content"""
        
        prompt_parts = [
            "Based on the following context and supplementary information, please answer the question accurately and concisely.",
            "",
            "**PRIMARY CONTEXT:**"
        ]
        
        # Add regular context
        for i, context in enumerate(regular_context, 1):
            prompt_parts.append(f"{i}. {context}")
        
        # Add relevant link content
        if relevant_links:
            prompt_parts.extend([
                "",
                "**SUPPLEMENTARY INFORMATION FROM LINKED SOURCES:**"
            ])
            
            for i, link_info in enumerate(relevant_links, 1):
                content = link_info['content_data'].get('content', '')[:max_link_content]
                url = link_info['url']
                relevance = link_info['relevance_score']
                
                prompt_parts.extend([
                    f"{i}. Source: {url} (Relevance: {relevance:.2f})",
                    f"   Content: {content}...",
                    ""
                ])
        
        prompt_parts.extend([
            f"**QUESTION:** {question}",
            "",
            "**INSTRUCTIONS:**",
            "- Use the primary context as your main source of information",
            "- Supplement with linked information when it adds value",
            "- Be precise and factual in your response",
            "- Use figures (1, 2, 3) instead of words for numbers",
            "- Keep answer concise but complete",
            "",
            "**ANSWER:**"
        ])
        
        return "\n".join(prompt_parts)
    
    def analyze_search_effectiveness(self, original_results: List[Dict],
                                   enhanced_results: List[Dict],
                                   question: str) -> Dict[str, Any]:
        """Analyze the effectiveness of search enhancement"""
        
        analysis = {
            'original_result_count': len(original_results),
            'enhanced_result_count': len(enhanced_results),
            'results_with_links': len([r for r in enhanced_results if r.get('linked_references')]),
            'results_boosted': len([r for r in enhanced_results if r.get('enhanced')]),
            'average_boost': 0.0,
            'top_result_improved': False
        }
        
        # Calculate average boost
        boosted_results = [r for r in enhanced_results if r.get('link_boost')]
        if boosted_results:
            analysis['average_boost'] = sum(r.get('link_boost', 0) for r in boosted_results) / len(boosted_results)
        
        # Check if top result improved
        if original_results and enhanced_results:
            original_top_score = original_results[0].get('similarity_score', 0)
            enhanced_top_score = enhanced_results[0].get('similarity_score', 0)
            analysis['top_result_improved'] = enhanced_top_score > original_top_score
        
        return analysis


# Global instance
enhanced_search_service = EnhancedSearchService()

# Main function for backwards compatibility
def apply_enhanced_search_if_enabled(search_results: List[Dict], 
                                   question: str,
                                   search_context: Dict[str, Any] = None) -> List[Dict]:
    """Backwards compatible function for enhanced search"""
    return enhanced_search_service.apply_enhanced_search_if_enabled(
        search_results, question, search_context
    )