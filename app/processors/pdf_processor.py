# app/processors/enhanced_pdf_processor.py
import fitz  # PyMuPDF
import re
import logging
import json
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
from urllib.parse import urlparse, urljoin
import requests
from app.processors.base_processor import SyncBaseProcessor
from app.utils.text_utils import words_to_numbers

logger = logging.getLogger(__name__)

@dataclass
class ExtractedLink:
    """Data structure for extracted links with context"""
    url: str
    context: str
    link_type: str  # 'api', 'reference', 'documentation', 'data'
    page_number: int
    surrounding_text: str
    confidence_score: float
    requires_fetching: bool = False
    fetch_priority: int = 0  # 1=high, 2=medium, 3=low

@dataclass
class APICallDecision:
    """Decision structure for API calls"""
    should_call: bool
    reason: str
    priority: int
    expected_data_type: str

class SmartLinkAnalyzer:
    """Intelligent analyzer to decide if links should be fetched"""
    
    def __init__(self):
        # Keywords that suggest important data/API endpoints
        self.high_priority_keywords = {
            'api', 'endpoint', 'data', 'fetch', 'retrieve', 'get', 'query',
            'database', 'service', 'real-time', 'current', 'latest', 'live'
        }
        
        # Keywords that suggest documentation/reference (lower priority)
        self.reference_keywords = {
            'documentation', 'docs', 'readme', 'guide', 'manual', 'help',
            'wiki', 'about', 'info', 'example', 'tutorial'
        }
        
        # Patterns for different types of APIs/services
        self.api_patterns = {
            'rest_api': r'(?:api|endpoint|service).*(?:\/v\d+|\/api)',
            'data_service': r'(?:data|query|search).*(?:\.json|\.xml|\.csv)',
            'real_time': r'(?:live|real-time|current|latest)',
            'parameter_endpoint': r'\{[^}]+\}|%[sd]|\$\{[^}]+\}'
        }
    
    def analyze_link_context(self, link: str, context: str, surrounding_text: str) -> ExtractedLink:
        """Analyze a link and its context to determine importance"""
        
        context_lower = context.lower()
        surrounding_lower = surrounding_text.lower()
        combined_text = f"{context_lower} {surrounding_lower}"
        
        # Determine link type
        link_type = self._classify_link_type(link, combined_text)
        
        # Calculate confidence score
        confidence = self._calculate_confidence_score(link, combined_text)
        
        # Determine if fetching is required
        requires_fetching = self._should_fetch_link(link, combined_text, link_type)
        
        # Set priority
        priority = self._get_fetch_priority(link_type, confidence, combined_text)
        
        return ExtractedLink(
            url=link,
            context=context,
            link_type=link_type,
            page_number=0,  # Will be set by caller
            surrounding_text=surrounding_text,
            confidence_score=confidence,
            requires_fetching=requires_fetching,
            fetch_priority=priority
        )
    
    def _classify_link_type(self, link: str, text: str) -> str:
        """Classify the type of link based on URL and context"""
        link_lower = link.lower()
        
        if any(pattern in link_lower for pattern in ['api', 'service', 'endpoint']):
            return 'api'
        elif any(pattern in link_lower for pattern in ['data', 'query', 'search']):
            return 'data'
        elif any(keyword in text for keyword in self.reference_keywords):
            return 'reference'
        elif any(keyword in text for keyword in self.high_priority_keywords):
            return 'api'
        else:
            return 'reference'
    
    def _calculate_confidence_score(self, link: str, text: str) -> float:
        """Calculate confidence score for link importance"""
        score = 0.0
        
        # URL-based scoring
        if 'api' in link.lower():
            score += 0.3
        if any(ext in link.lower() for ext in ['.json', '.xml', '.csv']):
            score += 0.2
        if re.search(r'\/v\d+\/', link):
            score += 0.1
        
        # Context-based scoring
        high_priority_matches = sum(1 for keyword in self.high_priority_keywords if keyword in text)
        score += min(high_priority_matches * 0.1, 0.3)
        
        # Pattern-based scoring
        for pattern_name, pattern in self.api_patterns.items():
            if re.search(pattern, text):
                score += 0.1
        
        return min(score, 1.0)
    
    def _should_fetch_link(self, link: str, text: str, link_type: str) -> bool:
        """Decide if a link should be fetched based on multiple factors"""
        
        # Don't fetch obvious documentation links
        if link_type == 'reference' and any(keyword in text for keyword in self.reference_keywords):
            if not any(keyword in text for keyword in self.high_priority_keywords):
                return False
        
        # Don't fetch if it's clearly a file download that's not data
        if any(ext in link.lower() for ext in ['.pdf', '.doc', '.zip', '.exe']):
            return False
        
        # Fetch if it's an API or data service
        if link_type in ['api', 'data']:
            return True
        
        # Fetch if context suggests it contains important data
        important_indicators = ['current', 'latest', 'real-time', 'live', 'data', 'fetch', 'retrieve']
        if any(indicator in text for indicator in important_indicators):
            return True
        
        return False
    
    def _get_fetch_priority(self, link_type: str, confidence: float, text: str) -> int:
        """Assign priority level (1=high, 2=medium, 3=low)"""
        if link_type == 'api' and confidence > 0.7:
            return 1
        elif confidence > 0.5 and any(keyword in text for keyword in self.high_priority_keywords):
            return 1
        elif link_type in ['api', 'data'] or confidence > 0.3:
            return 2
        else:
            return 3

class EnhancedPDFProcessor(SyncBaseProcessor):
    """Enhanced PDF processor with smart link extraction and fetching"""
    
    def __init__(self):
        super().__init__()
        self.link_analyzer = SmartLinkAnalyzer()
        self.extracted_links: List[ExtractedLink] = []
        self.fetched_content: Dict[str, str] = {}
        
        # Configure request session with sensible defaults
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (compatible; PDFProcessor/1.0)',
            'Accept': 'application/json, text/html, text/plain, */*'
        })
    
    def get_supported_extensions(self) -> set:
        """Return supported file extensions"""
        return {'pdf'}
    
    def supports_format(self, file_format: str) -> bool:
        """Check if the processor supports the given file format"""
        return file_format.lower() in self.get_supported_extensions()
    
    def process_sync(self, file_content: bytes, filename: str = "", 
                    fetch_links: bool = True, max_links_to_fetch: int = 10,
                    question_context: str = "", **kwargs) -> str:
        """
        Process PDF with intelligent link extraction and fetching
        
        Args:
            file_content: PDF file content as bytes
            filename: Name of the file being processed
            fetch_links: Whether to fetch content from discovered links
            max_links_to_fetch: Maximum number of links to fetch
            question_context: Context about the questions being asked (helps prioritize links)
        """
        start_time = self.log_processing_start(filename, "Enhanced PDF processing")
        
        try:
            # Step 1: Extract text and links from PDF
            doc = fitz.open(stream=file_content, filetype="pdf")
            text_parts = []
            self.extracted_links = []
            
            max_pages = min(50, doc.page_count)
            logger.info(f"Processing {max_pages} pages from PDF: {filename}")
            
            for page_num in range(max_pages):
                page = doc[page_num]
                page_text = page.get_text()
                
                if page_text.strip():
                    # Apply word-to-number conversion
                    page_text = words_to_numbers(page_text)
                    # Clean excessive whitespace
                    cleaned_text = re.sub(r'\s+', ' ', page_text.strip())
                    text_parts.append(cleaned_text)
                    
                    # Extract links from this page
                    page_links = self._extract_links_from_page(page, page_num, cleaned_text)
                    self.extracted_links.extend(page_links)
            
            doc.close()
            base_text = "\n\n".join(text_parts)
            
            if not base_text.strip():
                logger.warning(f"No text extracted from PDF: {filename}")
                return "No text content found in PDF"
            
            # Step 2: Analyze and prioritize links
            if self.extracted_links:
                self._prioritize_links(question_context)
                logger.info(f"Found {len(self.extracted_links)} links, "
                          f"{sum(1 for l in self.extracted_links if l.requires_fetching)} marked for fetching")
            
            # Step 3: Fetch content from high-priority links
            enhanced_content = base_text
            if fetch_links and self.extracted_links:
                fetched_content = self._fetch_priority_links(max_links_to_fetch, question_context)
                if fetched_content:
                    enhanced_content = self._merge_content(base_text, fetched_content)
            
            self.log_processing_complete(start_time, len(enhanced_content), filename, 
                                       f"Processed {len(text_parts)} pages, "
                                       f"fetched {len(self.fetched_content)} links")
            
            return enhanced_content
            
        except Exception as e:
            error_msg = self.handle_processing_error(e, filename, "Enhanced PDF processing")
            raise Exception(error_msg)
    
    def _extract_links_from_page(self, page, page_num: int, page_text: str) -> List[ExtractedLink]:
        """Extract links from a PDF page with context"""
        links = []
        
        # Extract links using PyMuPDF
        page_links = page.get_links()
        
        for link_dict in page_links:
            if 'uri' in link_dict and link_dict['uri']:
                url = link_dict['uri']
                
                # Get surrounding text context (approximate)
                rect = link_dict.get('from', None)
                surrounding_text = self._get_surrounding_text(page_text, url)
                
                # Analyze the link
                analyzed_link = self.link_analyzer.analyze_link_context(
                    url, page_text, surrounding_text
                )
                analyzed_link.page_number = page_num
                links.append(analyzed_link)
        
        # Also extract URLs from text using regex
        url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+[^\s<>"{}|\\^`\[\].,;:]'
        text_urls = re.findall(url_pattern, page_text)
        
        for url in text_urls:
            # Avoid duplicates
            if not any(link.url == url for link in links):
                surrounding_text = self._get_surrounding_text(page_text, url)
                analyzed_link = self.link_analyzer.analyze_link_context(
                    url, page_text, surrounding_text
                )
                analyzed_link.page_number = page_num
                links.append(analyzed_link)
        
        return links
    
    def _get_surrounding_text(self, page_text: str, url: str, context_chars: int = 200) -> str:
        """Extract surrounding text around a URL for context"""
        url_pos = page_text.find(url)
        if url_pos == -1:
            return ""
        
        start = max(0, url_pos - context_chars)
        end = min(len(page_text), url_pos + len(url) + context_chars)
        return page_text[start:end]
    
    def _prioritize_links(self, question_context: str):
        """Prioritize links based on question context and analysis"""
        
        if question_context:
            question_lower = question_context.lower()
            
            # Boost priority for links relevant to the questions
            for link in self.extracted_links:
                context_relevance = self._calculate_context_relevance(link, question_lower)
                if context_relevance > 0.3:
                    link.confidence_score = min(link.confidence_score + context_relevance, 1.0)
                    if link.fetch_priority > 1:
                        link.fetch_priority = max(1, link.fetch_priority - 1)
                    link.requires_fetching = True
        
        # Sort by priority and confidence
        self.extracted_links.sort(key=lambda x: (x.fetch_priority, -x.confidence_score))
    
    def _calculate_context_relevance(self, link: ExtractedLink, question_context: str) -> float:
        """Calculate how relevant a link is to the question context"""
        relevance = 0.0
        
        link_text = f"{link.context} {link.surrounding_text}".lower()
        
        # Simple keyword matching
        question_words = set(question_context.split())
        link_words = set(link_text.split())
        
        common_words = question_words.intersection(link_words)
        if len(question_words) > 0:
            relevance = len(common_words) / len(question_words)
        
        return min(relevance, 1.0)
    
    def _fetch_priority_links(self, max_links: int, question_context: str) -> Dict[str, str]:
        """Fetch content from high-priority links"""
        fetched_content = {}
        links_to_fetch = [link for link in self.extracted_links if link.requires_fetching][:max_links]
        
        logger.info(f"Attempting to fetch {len(links_to_fetch)} priority links")
        
        for link in links_to_fetch:
            try:
                content = self._safe_fetch_url(link.url)
                if content:
                    fetched_content[link.url] = {
                        'content': content,
                        'context': link.context,
                        'type': link.link_type,
                        'page': link.page_number
                    }
                    logger.info(f"Successfully fetched content from: {link.url[:50]}...")
                
            except Exception as e:
                logger.warning(f"Failed to fetch {link.url}: {str(e)}")
                continue
        
        self.fetched_content = fetched_content
        return fetched_content
    
    def _safe_fetch_url(self, url: str, timeout: int = 10, max_size: int = 1024*1024) -> Optional[str]:
        """Safely fetch content from a URL with proper error handling"""
        
        try:
            # Basic URL validation
            parsed = urlparse(url)
            if not parsed.scheme or not parsed.netloc:
                return None
            
            # Make request with timeout and size limits
            response = self.session.get(url, timeout=timeout, stream=True)
            response.raise_for_status()
            
            # Check content type
            content_type = response.headers.get('content-type', '').lower()
            if not any(ct in content_type for ct in ['text', 'json', 'xml', 'application/json']):
                logger.info(f"Skipping non-text content: {content_type}")
                return None
            
            # Read content with size limit
            content = ""
            size = 0
            for chunk in response.iter_content(chunk_size=8192, decode_unicode=True):
                if chunk:
                    size += len(chunk.encode('utf-8'))
                    if size > max_size:
                        logger.warning(f"Content too large, truncating: {url}")
                        break
                    content += chunk
            
            return content.strip() if content else None
            
        except requests.exceptions.RequestException as e:
            logger.warning(f"Request failed for {url}: {str(e)}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error fetching {url}: {str(e)}")
            return None
    
    def _merge_content(self, base_text: str, fetched_content: Dict[str, Any]) -> str:
        """Merge base PDF content with fetched link content"""
        
        content_parts = [base_text]
        content_parts.append("\n" + "="*80 + "\nADDITIONAL CONTENT FROM LINKED SOURCES:\n" + "="*80)
        
        for url, data in fetched_content.items():
            content_parts.append(f"\n\n--- Content from: {url} (Page {data['page']}, Type: {data['type']}) ---")
            content_parts.append(f"Context: {data['context'][:200]}...")
            content_parts.append(f"\nContent:\n{data['content']}")
            content_parts.append("-" * 40)
        
        return "\n".join(content_parts)
    
    def get_extracted_links(self) -> List[ExtractedLink]:
        """Get all extracted links for inspection"""
        return self.extracted_links
    
    def get_fetched_content(self) -> Dict[str, str]:
        """Get all fetched content for inspection"""
        return self.fetched_content