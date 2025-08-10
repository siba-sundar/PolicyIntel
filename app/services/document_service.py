# app/services/document_service.py
import httpx
import time
import logging
from fastapi import HTTPException
from app.config.settings import MAX_FILE_SIZE
from app.utils.file_utils import get_file_extension, is_supported_format
from app.utils.memory_utils import clear_memory
from app.utils.text_utils import words_to_numbers
from app.processors.pdf_processor import PDFProcessor
from app.processors.excel_processor import ExcelProcessor
from app.processors.powerpoint_processor import PowerPointProcessor
from app.processors.word_processor import WordProcessor
from app.processors.image_processor import ImageProcessor
from app.processors.html_processor import HTMLProcessor
from app.processors.text_processor import TextProcessor
from app.processors.zip_processor import ZipProcessor

logger = logging.getLogger(__name__)

class DocumentService:
    """Document processing service that handles multiple file formats"""
    
    def __init__(self):
        self.processors = {
            'pdf': PDFProcessor(),
            'xlsx': ExcelProcessor(),
            'xls': ExcelProcessor(),
            'pptx': PowerPointProcessor(),
            'ppt': PowerPointProcessor(),
            'docx': WordProcessor(),
            'doc': WordProcessor(),
            'jpg': ImageProcessor(),
            'jpeg': ImageProcessor(),
            'png': ImageProcessor(),
            'gif': ImageProcessor(),
            'bmp': ImageProcessor(),
            'tiff': ImageProcessor(),
            'webp': ImageProcessor(),
            'html': HTMLProcessor(),
            'htm': HTMLProcessor(),
            'txt': TextProcessor(),
            'zip': ZipProcessor(self),  # Pass self reference for nested processing
        }
    
    async def download_and_process_document(self, url: str) -> str:
        """Download and process document from URL"""
        start_time = time.time()
        logger.info(f"Starting enhanced document download from: {url}")
        
        try:
            # Download with optimized settings
            timeout = httpx.Timeout(30.0, connect=5.0)
            limits = httpx.Limits(max_keepalive_connections=5, max_connections=10)
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            
            async with httpx.AsyncClient(timeout=timeout, limits=limits, follow_redirects=True, headers=headers) as client:
                response = await client.get(url)
            
            response.raise_for_status()
            
            # Check content type
            content_type = response.headers.get('content-type', '').lower()
            logger.info(f"Content type: {content_type}")
            
            # Check file size
            content_length = len(response.content)
            if content_length > MAX_FILE_SIZE:
                raise HTTPException(status_code=400, detail=f"File too large: {content_length} bytes. Maximum allowed: {MAX_FILE_SIZE} bytes")
            
            logger.info(f"Document downloaded successfully. Size: {content_length} bytes")
            
            # Determine processing method based on content type or URL
            if 'text/html' in content_type or url.endswith(('.html', '.htm')) or '<!DOCTYPE html' in response.text[:100].lower():
                # Process as HTML
                logger.info("Processing as HTML page")
                
                try:
                    html_processor = HTMLProcessor()
                    document_text = await html_processor.process(response.content, url, base_url=url)
                    
                    logger.info(f"HTML processing completed. Final text length: {len(document_text)}")
                    return document_text
                    
                except Exception as e:
                    logger.error(f"HTML processing failed, trying as plain text: {str(e)}")
                    # Fallback to plain text
                    return words_to_numbers(response.text)
            
            else:
                # Process as file based on URL or content type
                filename = url.split('/')[-1].split('?')[0]  # Remove query parameters
                if not filename or '.' not in filename:
                    # Try to determine from content type
                    if 'pdf' in content_type:
                        filename = "document.pdf"
                    elif 'excel' in content_type or 'spreadsheet' in content_type:
                        filename = "document.xlsx"
                    elif 'powerpoint' in content_type or 'presentation' in content_type:
                        filename = "document.pptx"
                    elif 'word' in content_type:
                        filename = "document.docx"
                    else:
                        filename = "document.txt"
                
                # Check if format is supported
                is_supported, error_msg = is_supported_format(filename)
                if not is_supported and 'html' not in content_type:
                    raise HTTPException(status_code=400, detail=error_msg)
                
                # Process based on file type
                document_text = await self.process_file_content(response.content, filename)
                
                logger.info(f"File processing completed. Text length: {len(document_text)}")
            
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
    
    async def process_file_content(self, file_content: bytes, filename: str, depth: int = 0) -> str:
        """Process file content based on file type with depth tracking for nested ZIPs"""
        ext = get_file_extension(filename)
        
        try:
            processor = self.processors.get(ext)
            if not processor:
                raise Exception(f"No processor found for extension: {ext}")
            
            # Special handling for ZIP files that need depth parameter
            if ext == 'zip':
                return await processor.process(file_content, filename, depth=depth)
            else:
                return await processor.process(file_content, filename)
                
        except Exception as e:
            logger.error(f"Error processing {filename}: {str(e)}")
            raise Exception(f"Error processing {filename}: {str(e)}")