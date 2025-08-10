# app/processors/zip_processor.py
import io
import time
import logging
import zipfile
from typing import TYPE_CHECKING
from app.processors.base_processor import BaseProcessor
from app.config.settings import MAX_FILE_SIZE
from app.utils.file_utils import get_file_extension, is_supported_format
from app.utils.memory_utils import cleanup_variables, clear_memory

if TYPE_CHECKING:
    from app.services.document_service import DocumentService

logger = logging.getLogger(__name__)

class ZipProcessor(BaseProcessor):
    """ZIP file processor with nested ZIP handling"""
    
    def __init__(self, document_service: 'DocumentService' = None):
        self.document_service = document_service
    
    def get_supported_extensions(self) -> set:
        return {'zip'}
    
    
    def supports_format(self, file_format: str) -> bool:
        """Check if the processor supports the given file format"""
        return file_format.lower() in self.get_supported_extensions()
    
    async def process(self, file_content: bytes, filename: str, depth: int = 0, **kwargs) -> str:
        """Extract and process ZIP files - go only one level deep"""
        MAX_ZIP_DEPTH = 1  # Only go one level deep
        
        if depth >= MAX_ZIP_DEPTH:
            logger.warning(f"ZIP nesting depth limit reached ({MAX_ZIP_DEPTH}). Skipping further extraction.")
            return "No meaningful content found - ZIP nesting too deep (max 1 level allowed)"
        
        logger.info(f"Starting ZIP file processing (depth level: {depth + 1})")
        start_time = time.time()
        
        all_extracted_text = []
        processed_files = []
        nested_zips_found = []
        first_nested_zip_processed = False
        
        try:
            with zipfile.ZipFile(io.BytesIO(file_content), 'r') as zip_file:
                file_list = zip_file.namelist()
                logger.info(f"Found {len(file_list)} files in ZIP")
                
                for file_name in file_list:
                    # Skip directories and hidden files
                    if file_name.endswith('/') or file_name.startswith('.'):
                        continue
                    
                    # Check if this is a ZIP file
                    file_ext = get_file_extension(file_name)
                    if file_ext == 'zip':
                        nested_zips_found.append(file_name)
                        # Only process the first nested ZIP found
                        if first_nested_zip_processed:
                            logger.info(f"Skipping nested ZIP {file_name} - already processed first nested ZIP")
                            all_extracted_text.append(f"=== FILE: {file_name} ===\nSkipped: Multiple nested ZIPs found, only processing first one\n")
                            continue
                    
                    # Check file size before extraction
                    file_info = zip_file.getinfo(file_name)
                    if file_info.file_size > MAX_FILE_SIZE:
                        logger.warning(f"Skipping {file_name}: file too large ({file_info.file_size} bytes)")
                        all_extracted_text.append(f"=== FILE: {file_name} ===\nSkipped: File too large ({file_info.file_size} bytes)\n")
                        continue
                    
                    # Check if format is supported
                    is_supported, error_msg = is_supported_format(file_name)
                    if not is_supported:
                        logger.info(f"Skipping {file_name}: {error_msg}")
                        all_extracted_text.append(f"=== FILE: {file_name} ===\nSkipped: {error_msg}\n")
                        continue
                    
                    try:
                        # Extract file content
                        extracted_content = zip_file.read(file_name)
                        logger.info(f"Processing extracted file: {file_name} (depth: {depth + 1})")
                        
                        # Special handling for nested ZIPs
                        if file_ext == 'zip':
                            if depth >= MAX_ZIP_DEPTH - 1:  # At maximum depth
                                logger.warning(f"Cannot process nested ZIP {file_name} - at maximum depth {depth + 1}")
                                all_extracted_text.append(f"=== FILE: {file_name} ===\nSkipped: ZIP nesting limit reached (max 1 level allowed)\n")
                                continue
                            else:
                                first_nested_zip_processed = True
                                logger.info(f"Processing first nested ZIP: {file_name}")
                        
                        # Process based on file type, passing depth for nested ZIPs
                        if self.document_service:
                            file_text = await self.document_service.process_file_content(extracted_content, file_name, depth)
                        else:
                            file_text = "Document service not available"
                        
                        if file_text and file_text.strip():
                            all_extracted_text.append(f"=== FILE: {file_name} ===\n{file_text}\n")
                            processed_files.append(file_name)
                        else:
                            all_extracted_text.append(f"=== FILE: {file_name} ===\nNo content extracted\n")
                        
                        # Memory cleanup after each file
                        cleanup_variables(extracted_content)
                        clear_memory(f"ZIP file {file_name} processing")
                        
                    except Exception as e:
                        logger.error(f"Error processing {file_name} from ZIP: {str(e)}")
                        all_extracted_text.append(f"=== FILE: {file_name} ===\nError processing file: {str(e)}\n")
            
            final_text = "\n".join(all_extracted_text)
            
            # Add summary information
            summary_info = []
            summary_info.append(f"=== ZIP PROCESSING SUMMARY ===")
            summary_info.append(f"Total files found: {len(file_list)}")
            summary_info.append(f"Successfully processed: {len(processed_files)}")
            
            if nested_zips_found:
                summary_info.append(f"Nested ZIPs found: {len(nested_zips_found)} ({', '.join(nested_zips_found[:5])}{' ...' if len(nested_zips_found) > 5 else ''})")
                if len(nested_zips_found) > 1:
                    summary_info.append(f"Only processed first nested ZIP: {nested_zips_found[0]}")
                    summary_info.append(f"Skipped nested ZIPs: {', '.join(nested_zips_found[1:])}")
            
            summary_info.append(f"Processing depth: {depth + 1}/{MAX_ZIP_DEPTH + 1}")
            summary_info.append(f"="*50)
            
            final_text = "\n".join(summary_info) + "\n\n" + final_text
            
            logger.info(f"📦 ZIP processing completed (depth {depth + 1}) in {time.time() - start_time:.2f}s. Processed {len(processed_files)} files")
            if nested_zips_found:
                logger.info(f"📦 Found {len(nested_zips_found)} nested ZIP(s), processed only first: {nested_zips_found[0] if nested_zips_found else 'none'}")
            
            return final_text if final_text.strip() else "No processable content found in ZIP file"
            
        except Exception as e:
            logger.error(f"ZIP processing failed: {str(e)}")
            raise Exception(f"ZIP processing failed: {str(e)}")