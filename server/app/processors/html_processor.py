# app/processors/html_processor.py
import json
import re
import time
import logging
from typing import Dict, Any
from bs4 import BeautifulSoup
import html2text
from urllib.parse import urljoin
from app.processors.base_processor import BaseProcessor
from app.utils.text_utils import words_to_numbers

logger = logging.getLogger(__name__)

class HTMLProcessor(BaseProcessor):
    """HTML document processor"""
    
    def get_supported_extensions(self) -> set:
        return {'html', 'htm'}
    
    def supports_format(self, file_format: str) -> bool:
        """Check if the processor supports the given file format"""
        return file_format.lower() in self.get_supported_extensions()
    
    async def process(self, file_content: bytes, filename: str, base_url: str = "", **kwargs) -> str:
        """Process HTML file and extract meaningful content"""
        try:
            html_content = file_content.decode('utf-8')
        except UnicodeDecodeError:
            try:
                html_content = file_content.decode('latin-1')
            except UnicodeDecodeError:
                html_content = file_content.decode('utf-8', errors='ignore')
        
        # Extract and format HTML content
        extracted_data = self._extract_html_content(html_content, base_url)
        formatted_text = self._format_html_content_for_processing(extracted_data)
        return words_to_numbers(formatted_text.strip())
    
    def _extract_html_content(self, html_content: str, base_url: str = "") -> Dict[str, Any]:
        """Extract meaningful content from HTML page"""
        logger.info("Starting HTML content extraction")
        start_time = time.time()
        
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Remove unwanted elements
            for element in soup(['script', 'style', 'nav', 'header', 'footer', 
                               'aside', 'advertisement', 'ads', 'sidebar']):
                element.decompose()
            
            extracted_data = {
                'title': '',
                'main_content': '',
                'metadata': {},
                'structured_data': {},
                'links': [],
                'images': [],
                'tables': [],
                'lists': [],
                'clean_text': ''
            }
            
            # Extract title
            title_tag = soup.find('title')
            if title_tag:
                extracted_data['title'] = title_tag.get_text().strip()
            
            # Extract metadata
            meta_tags = soup.find_all('meta')
            for meta in meta_tags:
                name = meta.get('name') or meta.get('property') or meta.get('http-equiv')
                content = meta.get('content')
                if name and content:
                    extracted_data['metadata'][name] = content
            
            # Extract main content - prioritize semantic HTML5 elements
            main_content_selectors = [
                'main', 
                'article', 
                '[role="main"]',
                '.content',
                '.main-content',
                '#content',
                '#main'
            ]
            
            main_content = None
            for selector in main_content_selectors:
                main_content = soup.select_one(selector)
                if main_content:
                    break
            
            # If no semantic main content found, use body
            if not main_content:
                main_content = soup.find('body') or soup
            
            # Extract different content types
            
            # 1. Tables with structure preservation
            tables = main_content.find_all('table')
            for i, table in enumerate(tables):
                table_data = []
                headers = []
                
                # Extract headers
                header_row = table.find('tr')
                if header_row:
                    headers = [th.get_text().strip() for th in header_row.find_all(['th', 'td'])]
                
                # Extract rows
                rows = table.find_all('tr')[1:] if headers else table.find_all('tr')
                for row in rows:
                    cells = [td.get_text().strip() for td in row.find_all(['td', 'th'])]
                    if cells:
                        table_data.append(cells)
                
                if table_data:
                    table_info = {
                        'headers': headers,
                        'rows': table_data,
                        'table_number': i + 1
                    }
                    extracted_data['tables'].append(table_info)
            
            # 2. Lists (ordered and unordered)
            lists = main_content.find_all(['ul', 'ol'])
            for i, list_elem in enumerate(lists):
                list_items = [li.get_text().strip() for li in list_elem.find_all('li')]
                if list_items:
                    list_info = {
                        'type': list_elem.name,
                        'items': list_items,
                        'list_number': i + 1
                    }
                    extracted_data['lists'].append(list_info)
            
            # 3. Links with context
            links = main_content.find_all('a', href=True)
            for link in links:
                href = link.get('href')
                text = link.get_text().strip()
                if href and text:
                    # Resolve relative URLs
                    if base_url:
                        href = urljoin(base_url, href)
                    extracted_data['links'].append({
                        'url': href,
                        'text': text
                    })
            
            # 4. Images with alt text
            images = main_content.find_all('img')
            for img in images:
                src = img.get('src')
                alt = img.get('alt', '')
                if src:
                    if base_url:
                        src = urljoin(base_url, src)
                    extracted_data['images'].append({
                        'src': src,
                        'alt': alt
                    })
            
            # 5. Structured data (JSON-LD, microdata)
            json_ld_scripts = soup.find_all('script', type='application/ld+json')
            for script in json_ld_scripts:
                try:
                    structured_data = json.loads(script.string)
                    extracted_data['structured_data']['json_ld'] = structured_data
                except:
                    pass
            
            # 6. Clean text extraction using html2text for better formatting
            h = html2text.HTML2Text()
            h.ignore_links = False
            h.ignore_images = False
            h.body_width = 0  # Don't wrap lines
            
            # Convert main content to clean text
            main_content_html = str(main_content)
            clean_text = h.handle(main_content_html)
            
            # Additional cleaning
            clean_text = re.sub(r'\n\s*\n\s*\n', '\n\n', clean_text)  # Remove excessive newlines
            clean_text = re.sub(r'[ \t]+', ' ', clean_text)  # Normalize spaces
            clean_text = clean_text.strip()
            
            extracted_data['clean_text'] = clean_text
            extracted_data['main_content'] = clean_text
            
            logger.info(f"HTML content extraction completed in {time.time() - start_time:.2f}s")
            logger.info(f"Extracted: {len(clean_text)} characters, {len(tables)} tables, {len(lists)} lists, {len(links)} links")
            
            return extracted_data
            
        except Exception as e:
            logger.error(f"HTML content extraction failed: {str(e)}")
            raise Exception(f"HTML content extraction failed: {str(e)}")
    
    def _format_html_content_for_processing(self, extracted_data: Dict[str, Any]) -> str:
        """Format extracted HTML data into a structured text format for document processing"""
        formatted_parts = []
        
        # Add title
        if extracted_data.get('title'):
            formatted_parts.append(f"=== PAGE TITLE ===\n{extracted_data['title']}\n")
        
        # Add metadata if relevant
        metadata = extracted_data.get('metadata', {})
        relevant_meta = {}
        for key, value in metadata.items():
            if key.lower() in ['description', 'keywords', 'author', 'subject']:
                relevant_meta[key] = value
        
        if relevant_meta:
            formatted_parts.append("=== PAGE METADATA ===")
            for key, value in relevant_meta.items():
                formatted_parts.append(f"{key}: {value}")
            formatted_parts.append("")
        
        # Add structured data if available
        if extracted_data.get('structured_data'):
            formatted_parts.append("=== STRUCTURED DATA ===")
            structured_data = extracted_data['structured_data']
            formatted_parts.append(json.dumps(structured_data, indent=2))
            formatted_parts.append("")
        
        # Add tables with proper formatting
        tables = extracted_data.get('tables', [])
        for table in tables:
            formatted_parts.append(f"=== TABLE {table['table_number']} ===")
            
            if table['headers']:
                formatted_parts.append("HEADERS: " + " | ".join(table['headers']))
                formatted_parts.append("-" * 50)
            
            for row in table['rows']:
                formatted_parts.append(" | ".join(row))
            
            formatted_parts.append("")
        
        # Add lists
        lists = extracted_data.get('lists', [])
        for list_info in lists:
            formatted_parts.append(f"=== {list_info['type'].upper()} LIST {list_info['list_number']} ===")
            for i, item in enumerate(list_info['items'], 1):
                if list_info['type'] == 'ol':
                    formatted_parts.append(f"{i}. {item}")
                else:
                    formatted_parts.append(f"• {item}")
            formatted_parts.append("")
        
        # Add main content
        if extracted_data.get('main_content'):
            formatted_parts.append("=== MAIN CONTENT ===")
            formatted_parts.append(extracted_data['main_content'])
            formatted_parts.append("")
        
        # Add important links if they contain useful context
        links = extracted_data.get('links', [])
        important_links = [link for link in links if len(link['text']) > 10][:10]  # Limit to 10 most substantial links
        if important_links:
            formatted_parts.append("=== IMPORTANT LINKS ===")
            for link in important_links:
                formatted_parts.append(f"{link['text']}: {link['url']}")
            formatted_parts.append("")
        
        return "\n".join(formatted_parts)