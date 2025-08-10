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
                
                # Extract text with table structure preservation
                page_text = self._extract_text_with_table_structure(page)
                
                if page_text.strip():
                    # Apply word-to-number conversion
                    page_text = words_to_numbers(page_text)
                    # Clean excessive whitespace but preserve table formatting
                    cleaned_text = re.sub(r'\n\s*\n', '\n\n', page_text.strip())
                    cleaned_text = re.sub(r'[ \t]+', ' ', cleaned_text)
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
        """Safely fetch content from a URL with proper error handling and JSON parsing"""
        
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
            raw_content = ""
            size = 0
            for chunk in response.iter_content(chunk_size=8192, decode_unicode=True):
                if chunk:
                    size += len(chunk.encode('utf-8'))
                    if size > max_size:
                        logger.warning(f"Content too large, truncating: {url}")
                        break
                    raw_content += chunk
            
            if not raw_content:
                return None
            
            # Try to parse and extract meaningful data from JSON responses
            processed_content = self._process_fetched_content(raw_content.strip(), url)
            return processed_content
            
        except requests.exceptions.RequestException as e:
            logger.warning(f"Request failed for {url}: {str(e)}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error fetching {url}: {str(e)}")
            return None
    
    def _process_fetched_content(self, raw_content: str, url: str) -> str:
        """Process and extract meaningful data from fetched content, especially JSON responses"""
        
        try:
            # Try to parse as JSON first
            if raw_content.strip().startswith(('{', '[')):
                json_data = json.loads(raw_content)
                return self._extract_meaningful_json_data(json_data, url)
            else:
                # Return raw content for non-JSON responses
                return raw_content
        except json.JSONDecodeError:
            # If JSON parsing fails, return raw content
            return raw_content
        except Exception as e:
            logger.warning(f"Error processing content from {url}: {str(e)}")
            return raw_content
    
    def _extract_meaningful_json_data(self, json_data: Any, url: str) -> str:
        """Extract meaningful information from JSON responses"""
        
        try:
            # Handle different JSON response structures
            meaningful_parts = []
            
            if isinstance(json_data, dict):
                # Look for common data patterns in API responses
                
                # Pattern 1: {"success": true, "data": {...}, "message": "..."}
                if 'data' in json_data and isinstance(json_data['data'], dict):
                    data_content = json_data['data']
                    if 'city' in data_content:
                        meaningful_parts.append(f"Favorite City: {data_content['city']}")
                    
                    # Extract other key-value pairs from data
                    for key, value in data_content.items():
                        if key != 'city' and isinstance(value, (str, int, float)):
                            meaningful_parts.append(f"{key.title()}: {value}")
                
                # Pattern 2: Direct data in root
                elif 'city' in json_data:
                    meaningful_parts.append(f"City: {json_data['city']}")
                
                # Pattern 3: Flight information patterns
                if 'flight' in str(json_data).lower():
                    for key, value in json_data.items():
                        if 'flight' in key.lower() or 'number' in key.lower():
                            meaningful_parts.append(f"{key.title()}: {value}")
                
                # Pattern 4: Extract any other meaningful key-value pairs
                important_keys = ['id', 'code', 'number', 'name', 'location', 'address', 'phone', 'email']
                for key, value in json_data.items():
                    if (key.lower() in important_keys or 
                        any(important in key.lower() for important in ['flight', 'city', 'location']) and
                        isinstance(value, (str, int, float))):
                        meaningful_parts.append(f"{key.title()}: {value}")
                
                # Include success/status messages if meaningful
                if 'message' in json_data and json_data.get('success', False):
                    meaningful_parts.append(f"Status: {json_data['message']}")
            
            elif isinstance(json_data, list):
                # Handle array responses
                for i, item in enumerate(json_data[:5]):  # Limit to first 5 items
                    if isinstance(item, dict):
                        meaningful_parts.append(f"Item {i+1}: {self._extract_meaningful_json_data(item, url)}")
            
            # If we extracted meaningful data, format it nicely
            if meaningful_parts:
                formatted_content = f"API Response from {url}:\n" + "\n".join(meaningful_parts)
                return formatted_content
            else:
                # Fallback to formatted JSON
                return f"JSON Response from {url}:\n{json.dumps(json_data, indent=2, ensure_ascii=False)}"
        
        except Exception as e:
            logger.warning(f"Error extracting meaningful data from JSON: {str(e)}")
            # Fallback to raw JSON string
            return f"Raw JSON from {url}:\n{json.dumps(json_data, indent=2, ensure_ascii=False)}"
    
    def _extract_text_with_table_structure(self, page) -> str:
        """Extract text while preserving table structure using PyMuPDF table detection"""
        
        try:
            # Try to detect and extract tables first
            tables = page.find_tables()
            
            if tables:
                # If tables are found, extract them with structure preserved
                extracted_parts = []
                
                # Get regular text
                regular_text = page.get_text()
                
                # Extract and format tables
                for i, table in enumerate(tables):
                    try:
                        # Extract table data
                        table_data = table.extract()
                        
                        if table_data:
                            extracted_parts.append(f"\n=== TABLE {i+1} ===")
                            
                            # Format table with proper alignment
                            for row_idx, row in enumerate(table_data):
                                if row and any(cell and str(cell).strip() for cell in row):
                                    # Clean and format row
                                    cleaned_row = [str(cell or '').strip() for cell in row]
                                    
                                    # Create table-like formatting
                                    if row_idx == 0:  # Header row
                                        formatted_row = ' | '.join(f"{cell:^15}" for cell in cleaned_row)
                                        extracted_parts.append(formatted_row)
                                        extracted_parts.append('-' * len(formatted_row))
                                    else:
                                        formatted_row = ' | '.join(f"{cell:<15}" for cell in cleaned_row)
                                        extracted_parts.append(formatted_row)
                            
                            extracted_parts.append(f"=== END TABLE {i+1} ===\n")
                    
                    except Exception as e:
                        logger.warning(f"Error extracting table {i}: {str(e)}")
                        continue
                
                # Combine regular text with structured tables
                if extracted_parts:
                    return regular_text + "\n\n" + "\n".join(extracted_parts)
                else:
                    return regular_text
            else:
                # No tables detected, return regular text
                return page.get_text()
        
        except Exception as e:
            logger.warning(f"Error in table structure extraction: {str(e)}")
            # Fallback to regular text extraction
            return page.get_text()
    
    def _merge_content(self, base_text: str, fetched_content: Dict[str, Any]) -> str:
        """Merge base PDF content with fetched link content inline"""
        
        if not fetched_content:
            logger.info("🔍 DEBUG: No fetched content to merge")
            return base_text
        
        logger.info(f"🔍 DEBUG: Merging {len(fetched_content)} fetched items into base text")
        
        # Start with the base text
        enhanced_text = base_text
        
        # For each fetched URL, find its position in the text and insert content inline
        for url, data in fetched_content.items():
            content_text = data['content']
            logger.info(f"🔍 DEBUG: Processing URL: {url[:50]}...")
            logger.info(f"🔍 DEBUG: Content preview: {content_text[:100]}...")
            
            # Try to find the URL in the text
            url_pos = enhanced_text.find(url)
            if url_pos != -1:
                logger.info(f"🔍 DEBUG: Found URL at position {url_pos}, inserting content inline")
                # Insert fetched content right after the URL
                insertion_point = url_pos + len(url)
                
                # Create inline content block
                inline_content = f"\n\n[FETCHED DATA FROM ABOVE LINK]:\n{content_text}\n[END FETCHED DATA]\n"
                
                # Insert the content
                enhanced_text = (
                    enhanced_text[:insertion_point] + 
                    inline_content + 
                    enhanced_text[insertion_point:]
                )
                logger.info("🔍 DEBUG: Content inserted inline successfully")
            else:
                # If URL not found exactly, try to find it in a more flexible way
                url_parts = url.split('/')
                if len(url_parts) >= 3:
                    url_domain = url_parts[2]
                    domain_pos = enhanced_text.find(url_domain)
                    
                    if domain_pos != -1:
                        logger.info(f"🔍 DEBUG: Found domain {url_domain} at position {domain_pos}")
                        # Find the end of the line containing the domain
                        line_end = enhanced_text.find('\n', domain_pos)
                        if line_end == -1:
                            line_end = len(enhanced_text)
                        
                        # Insert content after the line
                        inline_content = f"\n\n[FETCHED DATA FROM {url}]:\n{content_text}\n[END FETCHED DATA]\n"
                        
                        enhanced_text = (
                            enhanced_text[:line_end] + 
                            inline_content + 
                            enhanced_text[line_end:]
                        )
                        logger.info("🔍 DEBUG: Content inserted after domain match")
                    else:
                        logger.warning(f"🔍 DEBUG: Could not find URL or domain in text for {url[:50]}...")
                        # As a fallback, append at the end with a clear marker
                        enhanced_text += f"\n\n[EXTERNAL DATA FROM {url}]:\n{content_text}\n[END EXTERNAL DATA]\n"
                        logger.info("🔍 DEBUG: Content appended at end as fallback")
                else:
                    logger.warning(f"🔍 DEBUG: Invalid URL format: {url}")
        
        logger.info(f"🔍 DEBUG: Final enhanced text length: {len(enhanced_text)} characters")
        return enhanced_text
    
    def get_extracted_links(self) -> List[ExtractedLink]:
        """Get all extracted links for inspection"""
        return self.extracted_links
    
    def get_fetched_content(self) -> Dict[str, str]:
        """Get all fetched content for inspection"""
        return self.fetched_content