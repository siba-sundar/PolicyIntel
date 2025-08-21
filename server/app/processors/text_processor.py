# app/processors/text_processor.py
import logging
from app.processors.base_processor import BaseProcessor
from app.utils.text_utils import words_to_numbers

logger = logging.getLogger(__name__)

class TextProcessor(BaseProcessor):
    """Plain text file processor"""
    
    def get_supported_extensions(self) -> set:
        return {'txt'}
    
    
    def supports_format(self, file_format: str) -> bool:
        """Check if the processor supports the given file format"""
        return file_format.lower() in self.get_supported_extensions()
    
    
    async def process(self, file_content: bytes, filename: str, **kwargs) -> str:
        """Process plain text files"""
        logger.info(f"Starting text file processing for: {filename}")
        
        try:
            # Handle text files with different encodings
            try:
                text_content = file_content.decode('utf-8')
            except UnicodeDecodeError:
                try:
                    text_content = file_content.decode('latin-1')
                except UnicodeDecodeError:
                    text_content = file_content.decode('utf-8', errors='ignore')
            
            processed_text = words_to_numbers(text_content.strip())
            
            logger.info(f"Text file processing completed. Text length: {len(processed_text)}")
            return processed_text
            
        except Exception as e:
            logger.error(f"Text file processing failed for {filename}: {str(e)}")
            raise Exception(f"Text file processing failed: {str(e)}")