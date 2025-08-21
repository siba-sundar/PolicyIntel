from abc import ABC, abstractmethod
from typing import Union, List, Tuple, Optional, Any, Dict
import logging
import time
from pathlib import Path

logger = logging.getLogger(__name__)

class BaseProcessor(ABC):
    """
    Abstract base class for all document processors.
    Defines the common interface and shared functionality for processing different file types.
    """
    
    def __init__(self):
        self.supported_extensions: List[str] = []
        self.processor_name: str = self.__class__.__name__
        
    @abstractmethod
    async def process(self, file_content: bytes, filename: str = "", **kwargs) -> str:
        """
        Process the file content and return extracted text.
        
        Args:
            file_content (bytes): The raw file content as bytes
            filename (str): The original filename (optional, used for extension detection)
            **kwargs: Additional processor-specific parameters
            
        Returns:
            str: Extracted and processed text content
            
        Raises:
            Exception: If processing fails
        """
        pass
    
    @abstractmethod
    def supports_format(self, filename: str) -> bool:
        """
        Check if this processor supports the given file format.
        
        Args:
            filename (str): The filename to check
            
        Returns:
            bool: True if the format is supported, False otherwise
        """
        pass
    
    def get_file_extension(self, filename: str) -> str:
        """Extract file extension safely"""
        return Path(filename).suffix.lower().lstrip('.')
    
    def validate_file_size(self, file_content: bytes, max_size: int) -> Tuple[bool, Optional[str]]:
        """
        Validate file size against maximum allowed size.
        
        Args:
            file_content (bytes): The file content to validate
            max_size (int): Maximum allowed size in bytes
            
        Returns:
            Tuple[bool, Optional[str]]: (is_valid, error_message)
        """
        if len(file_content) > max_size:
            return False, f"File too large: {len(file_content)} bytes. Maximum allowed: {max_size} bytes"
        return True, None
    
    def log_processing_start(self, filename: str = "", additional_info: str = "") -> float:
        """
        Log the start of processing and return start time.
        
        Args:
            filename (str): Name of the file being processed
            additional_info (str): Additional information to log
            
        Returns:
            float: Start timestamp for duration calculation
        """
        start_time = time.time()
        file_info = f" for {filename}" if filename else ""
        extra_info = f" - {additional_info}" if additional_info else ""
        logger.info(f"Starting {self.processor_name} processing{file_info}{extra_info}")
        return start_time
    
    def log_processing_complete(self, start_time: float, text_length: int, filename: str = "", additional_info: str = ""):
        """
        Log the completion of processing with duration and result info.
        
        Args:
            start_time (float): Start timestamp from log_processing_start
            text_length (int): Length of extracted text
            filename (str): Name of the file that was processed
            additional_info (str): Additional information to log
        """
        duration = time.time() - start_time
        file_info = f" for {filename}" if filename else ""
        extra_info = f" - {additional_info}" if additional_info else ""
        logger.info(f"{self.processor_name} processing completed{file_info} in {duration:.2f}s. "
                   f"Text length: {text_length} characters{extra_info}")
    
    def clean_text(self, text: str) -> str:
        """
        Basic text cleaning common to all processors.
        
        Args:
            text (str): Raw extracted text
            
        Returns:
            str: Cleaned text
        """
        if not text:
            return ""
        
        # Remove excessive whitespace
        import re
        cleaned = re.sub(r'\s+', ' ', text.strip())
        return cleaned
    
    def handle_processing_error(self, error: Exception, filename: str = "", operation: str = "") -> str:
        """
        Handle and log processing errors consistently.
        
        Args:
            error (Exception): The exception that occurred
            filename (str): Name of the file being processed
            operation (str): Description of the operation that failed
            
        Returns:
            str: Error message for the user
        """
        file_info = f" for {filename}" if filename else ""
        operation_info = f" during {operation}" if operation else ""
        error_msg = f"{self.processor_name} processing failed{file_info}{operation_info}: {str(error)}"
        logger.error(error_msg)
        return error_msg
    
    def get_supported_extensions(self) -> List[str]:
        """
        Get the list of file extensions supported by this processor.
        
        Returns:
            List[str]: List of supported file extensions (without dots)
        """
        return self.supported_extensions.copy()
    
    def get_processor_info(self) -> Dict[str, Any]:
        """
        Get information about this processor.
        
        Returns:
            Dict[str, Any]: Processor information including name and supported formats
        """
        return {
            "name": self.processor_name,
            "supported_extensions": self.get_supported_extensions(),
            "description": self.__doc__.strip().split('\n')[0] if self.__doc__ else "Document processor"
        }
    
    async def validate_and_process(self, file_content: bytes, filename: str = "", max_file_size: int = None, **kwargs) -> str:
        """
        Validate file and process it with error handling.
        
        Args:
            file_content (bytes): The raw file content
            filename (str): The original filename
            max_file_size (int): Maximum allowed file size in bytes
            **kwargs: Additional processor-specific parameters
            
        Returns:
            str: Processed text content
            
        Raises:
            Exception: If validation fails or processing errors occur
        """
        # Validate file size if specified
        if max_file_size:
            is_valid, error_msg = self.validate_file_size(file_content, max_file_size)
            if not is_valid:
                raise Exception(error_msg)
        
        # Check if format is supported
        if filename and not self.supports_format(filename):
            raise Exception(f"{self.processor_name} does not support format: {self.get_file_extension(filename)}")
        
        try:
            # Process the file
            result = await self.process(file_content, filename, **kwargs)
            
            # Validate result
            if not result or not result.strip():
                logger.warning(f"No content extracted by {self.processor_name} from {filename}")
                return "No content could be extracted from this file"
            
            return result
            
        except Exception as e:
            error_msg = self.handle_processing_error(e, filename, "content extraction")
            raise Exception(error_msg)


class SyncBaseProcessor(BaseProcessor):
    """
    Base class for processors that only have synchronous processing methods.
    Provides async wrapper around sync process method.
    """
    
    @abstractmethod
    def process_sync(self, file_content: bytes, filename: str = "", **kwargs) -> str:
        """
        Synchronous processing method to be implemented by sync processors.
        
        Args:
            file_content (bytes): The raw file content as bytes
            filename (str): The original filename
            **kwargs: Additional processor-specific parameters
            
        Returns:
            str: Extracted and processed text content
        """
        pass
    
    async def process(self, file_content: bytes, filename: str = "", **kwargs) -> str:
        """
        Async wrapper around synchronous processing method.
        """
        import asyncio
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.process_sync, file_content, filename, **kwargs)


class AsyncBaseProcessor(BaseProcessor):
    """
    Base class for processors that have native async processing capabilities.
    """
    pass