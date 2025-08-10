# app/processors/image_processor.py
import io
import re
import time
import logging
import requests
import pytesseract
from PIL import Image
from app.processors.base_processor import BaseProcessor
from app.config.settings import MAX_IMAGE_SIZE, OCR_SPACE_API_KEY
from app.utils.memory_utils import cleanup_variables, clear_memory

logger = logging.getLogger(__name__)

class ImageProcessor(BaseProcessor):
    """Image processor with OCR capabilities"""
    
    def get_supported_extensions(self) -> set:
        return {'jpg', 'jpeg', 'png', 'gif', 'bmp', 'tiff', 'webp'}
    
    def supports_format(self, file_format: str) -> bool:
        """Check if the processor supports the given file format"""
        return file_format.lower() in self.get_supported_extensions()
    
    async def process(self, file_content: bytes, filename: str, **kwargs) -> str:
        """Process image with OCR, memory-optimized"""
        logger.info(f"Starting OCR processing for image: {filename}")
        start_time = time.time()
        
        try:
            # Load and optimize image
            image = Image.open(io.BytesIO(file_content))
            
            # Convert to RGB if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Resize if too large to save memory
            if image.size[0] > MAX_IMAGE_SIZE[0] or image.size[1] > MAX_IMAGE_SIZE[1]:
                image.thumbnail(MAX_IMAGE_SIZE, Image.Resampling.LANCZOS)
                logger.info(f"Resized image to {image.size} for memory efficiency")
            
            # Perform OCR with optimized settings
            ocr_config = '--oem 3 --psm 6 -l eng'  # Optimized OCR settings
            extracted_text = pytesseract.image_to_string(image, config=ocr_config)
            
            # Clean up image from memory
            image.close()
            cleanup_variables(image)
            clear_memory("OCR image processing")
            
            # Clean extracted text
            cleaned_text = re.sub(r'\s+', ' ', extracted_text.strip())
            
            logger.info(f"OCR completed in {time.time() - start_time:.2f}s. Text length: {len(cleaned_text)}")
            return cleaned_text if cleaned_text else "No text detected in image"
            
        except Exception as e:
            logger.error(f"OCR processing failed for {filename}: {str(e)}")
            return f"OCR processing failed for {filename}"

def process_image_ocr_space(image_bytes: bytes, image_name: str) -> str:
    """Process image using OCR Space API"""
    try:
        response = requests.post(
            url='https://api.ocr.space/parse/image',
            files={'filename': (image_name + '.jpg', image_bytes)},
            data={'apikey': OCR_SPACE_API_KEY, 'language': 'eng'}
        )
        result = response.json()
        return result['ParsedResults'][0]['ParsedText'] if 'ParsedResults' in result else ''
    except Exception as e:
        return f"[OCR Failed: {str(e)}]"