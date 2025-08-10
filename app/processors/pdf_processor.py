# app/processors/pdf_processor.py
import fitz  # PyMuPDF
import re
import logging
from app.processors.base_processor import SyncBaseProcessor
from app.utils.text_utils import words_to_numbers

logger = logging.getLogger(__name__)

class PDFProcessor(SyncBaseProcessor):
    """PDF document processor using PyMuPDF"""
    
    def get_supported_extensions(self) -> set:
        """Return supported file extensions"""
        return {'pdf'}
    
    def supports_format(self, file_format: str) -> bool:
        """Check if the processor supports the given file format"""
        return file_format.lower() in self.get_supported_extensions()
    
    def process_sync(self, file_content: bytes, filename: str = "", **kwargs) -> str:
        """Process PDF file and extract text synchronously"""
        start_time = self.log_processing_start(filename, "PDF text extraction")
        
        try:
            doc = fitz.open(stream=file_content, filetype="pdf")
            text_parts = []
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
            
            doc.close()
            final_text = "\n\n".join(text_parts)
            
            if not final_text.strip():
                logger.warning(f"No text extracted from PDF: {filename}")
                return "No text content found in PDF"
            
            self.log_processing_complete(start_time, len(final_text), filename, f"Processed {len(text_parts)} pages")
            return final_text
            
        except Exception as e:
            error_msg = self.handle_processing_error(e, filename, "PDF text extraction")
            raise Exception(error_msg)