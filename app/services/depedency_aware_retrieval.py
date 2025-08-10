import re
import logging
import asyncio
import hashlib
from typing import List, Dict, Any, Optional, Tuple
from urllib.parse import urlparse, urljoin
import aiohttp
from dataclasses import dataclass
from app.services.llm_service import ask_llm

logger = logging.getLogger(__name__)

@dataclass
class LinkDependency:
    """Represents a link found in the document with its context"""
    url: str
    context_before: str
    context_after: str
    chunk_index: int
    dependency_score: float = 0.0
    is_required: bool = False
    content: Optional[str] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

class DependencyAwareRetrieval:
    """
    Intelligent link dependency manager for RAG systems.
    Determines when links need to be fetched based on question context.
    """
    
    def __init__(self, max_concurrent_requests: int = 3, request_timeout: int = 10):
        self.max_concurrent_requests = max_concurrent_requests
        self.request_timeout = request_timeout
        self.link_cache = {}  # Simple in-memory cache
        
        # Patterns that indicate link dependency
        self.dependency_patterns = [
            r'\b(?:check|verify|validate|confirm)\s+(?:status|state|condition)\b',
            r'\b(?:download|fetch|get|retrieve)\s+(?:data|file|report|details)\b',
            r'\b(?:call|invoke|execute)\s+(?:api|endpoint|service)\b',
            r'\b(?:see|view|access|visit)\s+(?:details|more|link|here)\b',
            r'\b(?:next|following|subsequent)\s+step\b',
            r'\b(?:continue|proceed)\s+(?:to|with)\b',
            r'\b(?:refer|reference)\s+(?:to|the)\b.*(?:link|url|site)\b',
            r'\b(?:login|authenticate|access)\b',
            r'\b(?:submit|send|post)\s+(?:form|data)\b',
            r'\b(?:update|modify|change)\s+(?:settings|configuration)\b'
        ]
        
        # Compile patterns for efficiency
        self.compiled_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in self.dependency_patterns]
        
    async def extract_links_from_chunks(self, chunks: List[Dict[str, Any]]) -> List[LinkDependency]:
        """
        Extract and tag URLs from document chunks with context.
        """
        logger.info("🔍 Extracting links from document chunks")
        dependencies = []
        
        # Common URL patterns
        url_patterns = [
            r'https?://[^\s<>"{}|\\^`\[\]]+',  # Standard URLs
            r'www\.[^\s<>"{}|\\^`\[\]]+',       # www URLs
            r'ftp://[^\s<>"{}|\\^`\[\]]+',      # FTP URLs
        ]
        
        for chunk_idx, chunk in enumerate(chunks):
            text = chunk.get('text', '')
            
            for pattern in url_patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                
                for match in matches:
                    url = match.group(0)
                    start_pos = match.start()
                    end_pos = match.end()
                    
                    # Extract context around the URL (50 chars before/after)
                    context_start = max(0, start_pos - 50)
                    context_end = min(len(text), end_pos + 50)
                    
                    context_before = text[context_start:start_pos].strip()
                    context_after = text[end_pos:context_end].strip()
                    
                    # Clean and validate URL
                    cleaned_url = self._clean_url(url)
                    if cleaned_url and self._is_valid_url(cleaned_url):
                        dependency = LinkDependency(
                            url=cleaned_url,
                            context_before=context_before,
                            context_after=context_after,
                            chunk_index=chunk_idx
                        )
                        dependencies.append(dependency)
        
        logger.info(f"📋 Found {len(dependencies)} potential link dependencies")
        return dependencies

    def _clean_url(self, url: str) -> str:
        """Clean and normalize URL"""
        url = url.strip().rstrip('.,;:!?')
        if url.startswith('www.') and not url.startswith('http'):
            url = 'https://' + url
        return url

    def _is_valid_url(self, url: str) -> bool:
        """Basic URL validation"""
        try:
            result = urlparse(url)
            return all([result.scheme, result.netloc])
        except Exception:
            return False

    async def classify_dependencies(self, question: str, dependencies: List[LinkDependency]) -> List[LinkDependency]:
        """
        Classify which links are required based on the question context.
        """
        logger.info("🤖 Classifying link dependencies")
        
        if not dependencies:
            return dependencies
            
        # Quick pattern-based check first
        question_lower = question.lower()
        has_dependency_keywords = any(pattern.search(question) for pattern in self.compiled_patterns)
        
        if not has_dependency_keywords:
            logger.info("❌ No dependency keywords found - skipping link classification")
            return dependencies
        
        # Use LLM for more sophisticated classification
        classification_prompt = f"""
You are a dependency analyzer. Analyze if the given question requires visiting external links to provide a complete answer.

**QUESTION:** {question}

**AVAILABLE LINKS AND CONTEXT:**
"""
        
        for i, dep in enumerate(dependencies[:5]):  # Limit to first 5 for efficiency
            classification_prompt += f"""
Link {i+1}: {dep.url}
Context before: {dep.context_before[-100:]}
Context after: {dep.context_after[:100]}
---
"""
        
        classification_prompt += """
**ANALYSIS INSTRUCTIONS:**
1. Determine if the question asks for information that requires visiting any of these links
2. Look for keywords like: "check status", "get data", "download", "verify", "access", "login", etc.
3. Consider if the link contains necessary steps, data, or authentication needed for the answer

**RESPONSE FORMAT:**
For each link, respond with only: "REQUIRED" or "SKIP"
Example: Link1: REQUIRED, Link2: SKIP, Link3: REQUIRED

**DECISION:**
"""
        
        try:
            response = await ask_llm(classification_prompt)
            logger.info(f"🔍 LLM classification response: {response}")
            
            # Parse LLM response
            required_links = self._parse_classification_response(response, dependencies)
            
            # Update dependency flags
            for dep in dependencies:
                dep.is_required = dep.url in required_links
                if dep.is_required:
                    dep.dependency_score = 0.9
                    
            required_count = sum(1 for dep in dependencies if dep.is_required)
            logger.info(f"✅ Classified {required_count} links as required out of {len(dependencies)}")
            
        except Exception as e:
            logger.error(f"❌ Error in LLM classification: {str(e)}")
            # Fallback to pattern-based classification
            self._fallback_classification(question, dependencies)
        
        return dependencies

    def _parse_classification_response(self, response: str, dependencies: List[LinkDependency]) -> List[str]:
        """Parse LLM classification response"""
        required_links = []
        lines = response.strip().split('\n')
        
        for line in lines:
            if 'REQUIRED' in line.upper():
                # Try to extract link number or URL
                if 'Link' in line:
                    try:
                        link_num = int(re.search(r'Link\s*(\d+)', line).group(1)) - 1
                        if 0 <= link_num < len(dependencies):
                            required_links.append(dependencies[link_num].url)
                    except (AttributeError, ValueError, IndexError):
                        continue
        
        return required_links

    def _fallback_classification(self, question: str, dependencies: List[LinkDependency]):
        """Fallback pattern-based classification when LLM fails"""
        logger.info("🔄 Using fallback pattern-based classification")
        
        question_lower = question.lower()
        
        for dep in dependencies:
            # Check if question contains dependency keywords
            has_keywords = any(pattern.search(question) for pattern in self.compiled_patterns)
            
            # Check if link context suggests it's actionable
            context_combined = (dep.context_before + " " + dep.context_after).lower()
            actionable_terms = ['click', 'visit', 'go to', 'access', 'login', 'download', 'check', 'verify']
            has_actionable_context = any(term in context_combined for term in actionable_terms)
            
            dep.is_required = has_keywords and has_actionable_context
            dep.dependency_score = 0.7 if dep.is_required else 0.2

    async def fetch_required_links(self, dependencies: List[LinkDependency]) -> List[LinkDependency]:
        """
        Fetch content from required links with concurrent processing and caching.
        """
        required_deps = [dep for dep in dependencies if dep.is_required]
        
        if not required_deps:
            logger.info("🚫 No required links to fetch")
            return dependencies
            
        logger.info(f"🌐 Fetching {len(required_deps)} required links")
        
        # Process links in batches to avoid overwhelming servers
        semaphore = asyncio.Semaphore(self.max_concurrent_requests)
        
        async def fetch_single_link(dep: LinkDependency) -> LinkDependency:
            async with semaphore:
                return await self._fetch_link_content(dep)
        
        # Execute concurrent requests
        tasks = [fetch_single_link(dep) for dep in required_deps]
        
        try:
            await asyncio.gather(*tasks, return_exceptions=True)
        except Exception as e:
            logger.error(f"❌ Error in concurrent link fetching: {str(e)}")
        
        successful_fetches = sum(1 for dep in required_deps if dep.content)
        logger.info(f"✅ Successfully fetched {successful_fetches}/{len(required_deps)} required links")
        
        return dependencies

    async def _fetch_link_content(self, dep: LinkDependency) -> LinkDependency:
        """Fetch content from a single link with caching"""
        
        # Check cache first
        cache_key = hashlib.md5(dep.url.encode()).hexdigest()
        if cache_key in self.link_cache:
            logger.info(f"📦 Using cached content for {dep.url}")
            dep.content = self.link_cache[cache_key]['content']
            dep.metadata = self.link_cache[cache_key]['metadata']
            return dep
        
        try:
            timeout = aiohttp.ClientTimeout(total=self.request_timeout)
            
            async with aiohttp.ClientSession(timeout=timeout) as session:
                logger.info(f"🔗 Fetching: {dep.url}")
                
                async with session.get(dep.url, ssl=False) as response:
                    if response.status == 200:
                        content = await response.text()
                        
                        # Basic content processing
                        processed_content = self._process_web_content(content)
                        
                        # Store in dependency
                        dep.content = processed_content[:2000]  # Limit content size
                        dep.metadata = {
                            'status_code': response.status,
                            'content_type': response.headers.get('content-type', ''),
                            'url': str(response.url),
                            'content_length': len(content)
                        }
                        
                        # Cache the result
                        self.link_cache[cache_key] = {
                            'content': dep.content,
                            'metadata': dep.metadata
                        }
                        
                        logger.info(f"✅ Successfully fetched {dep.url} ({len(content)} chars)")
                        
                    else:
                        logger.warning(f"⚠️ HTTP {response.status} for {dep.url}")
                        dep.metadata = {'status_code': response.status, 'error': f'HTTP {response.status}'}
                        
        except asyncio.TimeoutError:
            logger.warning(f"⏰ Timeout fetching {dep.url}")
            dep.metadata = {'error': 'timeout'}
            
        except Exception as e:
            logger.error(f"❌ Error fetching {dep.url}: {str(e)}")
            dep.metadata = {'error': str(e)}
            
        return dep

    def _process_web_content(self, content: str) -> str:
        """Basic processing of web content to extract useful text"""
        
        # Remove HTML tags (basic cleanup)
        clean_text = re.sub(r'<[^>]+>', ' ', content)
        
        # Remove extra whitespace
        clean_text = re.sub(r'\s+', ' ', clean_text)
        
        # Remove common web artifacts
        clean_text = re.sub(r'(javascript|css|style)\s*:', '', clean_text, flags=re.IGNORECASE)
        
        return clean_text.strip()

    def enrich_context(self, original_chunks: List[str], dependencies: List[LinkDependency]) -> List[str]:
        """
        Enrich the original context with fetched link content.
        """
        enriched_chunks = original_chunks.copy()
        
        fetched_deps = [dep for dep in dependencies if dep.content]
        
        if not fetched_deps:
            return enriched_chunks
            
        logger.info(f"📈 Enriching context with {len(fetched_deps)} fetched links")
        
        for dep in fetched_deps:
            link_context = f"""
--- FETCHED LINK CONTENT ---
URL: {dep.url}
Context: {dep.context_before} [LINK] {dep.context_after}
Content: {dep.content[:1000]}...
Status: {dep.metadata.get('status_code', 'Unknown')}
---
"""
            enriched_chunks.append(link_context)
            
        logger.info(f"✅ Context enriched: {len(original_chunks)} → {len(enriched_chunks)} chunks")
        return enriched_chunks

    async def process_dependencies(self, question: str, chunks: List[Dict[str, Any]]) -> Tuple[List[str], List[LinkDependency]]:
        """
        Main method to process dependencies: extract, classify, fetch, and enrich.
        """
        logger.info("🚀 Starting dependency-aware retrieval process")
        
        try:
            # Stage 1: Extract links from chunks
            dependencies = await self.extract_links_from_chunks(chunks)
            
            if not dependencies:
                logger.info("📋 No links found in document")
                return [chunk.get('text', '') for chunk in chunks], []
            
            # Stage 2: Classify dependencies
            dependencies = await self.classify_dependencies(question, dependencies)
            
            # Stage 3: Fetch required links
            dependencies = await self.fetch_required_links(dependencies)
            
            # Stage 4: Enrich context
            original_chunks = [chunk.get('text', '') for chunk in chunks]
            enriched_chunks = self.enrich_context(original_chunks, dependencies)
            
            logger.info("✅ Dependency-aware retrieval completed successfully")
            return enriched_chunks, dependencies
            
        except Exception as e:
            logger.error(f"❌ Error in dependency processing: {str(e)}")
            # Return original chunks on error
            return [chunk.get('text', '') for chunk in chunks], []

    def get_dependency_summary(self, dependencies: List[LinkDependency]) -> Dict[str, Any]:
        """Get a summary of dependency processing results"""
        if not dependencies:
            return {"total": 0, "required": 0, "fetched": 0, "failed": 0}
            
        total = len(dependencies)
        required = sum(1 for dep in dependencies if dep.is_required)
        fetched = sum(1 for dep in dependencies if dep.content)
        failed = sum(1 for dep in dependencies if dep.is_required and not dep.content)
        
        return {
            "total": total,
            "required": required, 
            "fetched": fetched,
            "failed": failed,
            "success_rate": (fetched / required * 100) if required > 0 else 0
        }