# app/utils/security_utils.py
import os
import json
import hashlib
import hmac
import base64
from typing import Any, Dict, List, Optional, Tuple, Union
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.backends import default_backend
import logging

logger = logging.getLogger(__name__)

class SecurityManager:
    """
    Comprehensive security manager for AES encryption and hash-based integrity verification
    """
    
    def __init__(self, master_key: Optional[str] = None):
        """Initialize security manager with master key"""
        self.master_key = master_key or os.environ.get('SECURITY_MASTER_KEY', self._generate_master_key())
        self.backend = default_backend()
        
    def _generate_master_key(self) -> str:
        """Generate a secure master key"""
        return base64.urlsafe_b64encode(os.urandom(32)).decode('utf-8')
    
    def derive_key(self, password: str, salt: bytes) -> bytes:
        """Derive encryption key using PBKDF2"""
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
            backend=self.backend
        )
        return kdf.derive(password.encode())
    
    def encrypt_data(self, data: Union[str, dict, list], context: str = "general") -> Dict[str, str]:
        """
        Encrypt data using AES-256-GCM
        Returns: dict with encrypted_data, salt, nonce, tag, and integrity_hash
        """
        try:
            # Convert data to JSON string if not already string
            if isinstance(data, (dict, list)):
                data_str = json.dumps(data, ensure_ascii=False, separators=(',', ':'))
            else:
                data_str = str(data)
            
            # Generate salt and derive key
            salt = os.urandom(16)
            key = self.derive_key(f"{self.master_key}:{context}", salt)
            
            # Generate nonce for GCM mode
            nonce = os.urandom(12)
            
            # Create cipher and encrypt
            cipher = Cipher(algorithms.AES(key), modes.GCM(nonce), backend=self.backend)
            encryptor = cipher.encryptor()
            
            ciphertext = encryptor.update(data_str.encode('utf-8')) + encryptor.finalize()
            
            # Get authentication tag
            tag = encryptor.tag
            
            # Create integrity hash
            integrity_data = {
                'context': context,
                'data_length': len(data_str),
                'salt': base64.b64encode(salt).decode(),
                'nonce': base64.b64encode(nonce).decode()
            }
            integrity_hash = self.create_integrity_hash(integrity_data)
            
            return {
                'encrypted_data': base64.b64encode(ciphertext).decode(),
                'salt': base64.b64encode(salt).decode(),
                'nonce': base64.b64encode(nonce).decode(),
                'tag': base64.b64encode(tag).decode(),
                'integrity_hash': integrity_hash,
                'context': context
            }
            
        except Exception as e:
            logger.error(f"Encryption failed: {str(e)}")
            raise SecurityException(f"Encryption failed: {str(e)}")
    
    def decrypt_data(self, encrypted_package: Dict[str, str]) -> Union[str, dict, list]:
        """
        Decrypt data and verify integrity
        """
        try:
            # Verify integrity first
            if not self.verify_integrity(encrypted_package):
                raise SecurityException("Integrity verification failed - data may be tampered")
            
            # Extract components
            ciphertext = base64.b64decode(encrypted_package['encrypted_data'])
            salt = base64.b64decode(encrypted_package['salt'])
            nonce = base64.b64decode(encrypted_package['nonce'])
            tag = base64.b64decode(encrypted_package['tag'])
            context = encrypted_package['context']
            
            # Derive key
            key = self.derive_key(f"{self.master_key}:{context}", salt)
            
            # Create cipher and decrypt
            cipher = Cipher(algorithms.AES(key), modes.GCM(nonce, tag), backend=self.backend)
            decryptor = cipher.decryptor()
            
            decrypted_data = decryptor.update(ciphertext) + decryptor.finalize()
            data_str = decrypted_data.decode('utf-8')
            
            # Try to parse as JSON, fallback to string
            try:
                return json.loads(data_str)
            except json.JSONDecodeError:
                return data_str
                
        except Exception as e:
            logger.error(f"Decryption failed: {str(e)}")
            raise SecurityException(f"Decryption failed: {str(e)}")
    
    def create_integrity_hash(self, data: Union[str, dict, list]) -> str:
        """
        Create HMAC-SHA256 hash for integrity verification
        """
        try:
            if isinstance(data, (dict, list)):
                data_str = json.dumps(data, sort_keys=True, ensure_ascii=False, separators=(',', ':'))
            else:
                data_str = str(data)
            
            # Create HMAC with master key
            hmac_obj = hmac.new(
                self.master_key.encode('utf-8'),
                data_str.encode('utf-8'),
                hashlib.sha256
            )
            
            return base64.b64encode(hmac_obj.digest()).decode()
            
        except Exception as e:
            logger.error(f"Hash creation failed: {str(e)}")
            raise SecurityException(f"Hash creation failed: {str(e)}")
    
    def verify_integrity(self, encrypted_package: Dict[str, str]) -> bool:
        """
        Verify integrity hash of encrypted package
        """
        try:
            stored_hash = encrypted_package.get('integrity_hash')
            if not stored_hash:
                logger.warning("No integrity hash found")
                return False
            
            # Recreate integrity data
            integrity_data = {
                'context': encrypted_package.get('context', ''),
                'data_length': len(base64.b64decode(encrypted_package['encrypted_data'])),
                'salt': encrypted_package['salt'],
                'nonce': encrypted_package['nonce']
            }
            
            expected_hash = self.create_integrity_hash(integrity_data)
            
            # Use constant-time comparison to prevent timing attacks
            return hmac.compare_digest(stored_hash, expected_hash)
            
        except Exception as e:
            logger.error(f"Integrity verification failed: {str(e)}")
            return False
    
    def encrypt_search_results(self, search_results: List[Dict]) -> Dict[str, str]:
        """Encrypt search results with metadata preservation"""
        return self.encrypt_data(search_results, context="search_results")
    
    def encrypt_questions(self, questions: List[str]) -> Dict[str, str]:
        """Encrypt questions list"""
        return self.encrypt_data(questions, context="questions")
    
    def encrypt_answers(self, answers: List[str]) -> Dict[str, str]:
        """Encrypt answers list"""
        return self.encrypt_data(answers, context="answers")
    
    def encrypt_document_content(self, content: str) -> Dict[str, str]:
        """Encrypt document content"""
        return self.encrypt_data(content, context="document")
    
    def encrypt_chunks(self, chunks: List[Dict]) -> Dict[str, str]:
        """Encrypt document chunks"""
        return self.encrypt_data(chunks, context="chunks")
    
    def secure_log_entry(self, log_data: Dict) -> str:
        """Create secure, tamper-evident log entry"""
        timestamp = str(int(time.time() * 1000))  # millisecond precision
        log_entry = {
            'timestamp': timestamp,
            'data': log_data,
            'sequence': hashlib.sha256(f"{timestamp}{json.dumps(log_data, sort_keys=True)}".encode()).hexdigest()[:16]
        }
        
        integrity_hash = self.create_integrity_hash(log_entry)
        log_entry['integrity_hash'] = integrity_hash
        
        return json.dumps(log_entry, separators=(',', ':'))


class SecureCache:
    """
    Enhanced cache with encryption and integrity verification
    """
    
    def __init__(self, security_manager: SecurityManager):
        self.security_manager = security_manager
        self._cache = {}
    
    def set(self, key: str, value: Any, context: str = "cache") -> bool:
        """Store encrypted value in cache"""
        try:
            cache_key = hashlib.sha256(key.encode()).hexdigest()
            encrypted_package = self.security_manager.encrypt_data(value, context=f"cache:{context}")
            self._cache[cache_key] = encrypted_package
            return True
        except Exception as e:
            logger.error(f"Cache set failed: {str(e)}")
            return False
    
    def get(self, key: str, context: str = "cache") -> Optional[Any]:
        """Retrieve and decrypt value from cache"""
        try:
            cache_key = hashlib.sha256(key.encode()).hexdigest()
            encrypted_package = self._cache.get(cache_key)
            
            if not encrypted_package:
                return None
            
            return self.security_manager.decrypt_data(encrypted_package)
        except Exception as e:
            logger.error(f"Cache get failed: {str(e)}")
            return None
    
    def clear(self) -> bool:
        """Clear all cache entries"""
        try:
            self._cache.clear()
            return True
        except Exception as e:
            logger.error(f"Cache clear failed: {str(e)}")
            return False


class SecurityException(Exception):
    """Custom exception for security-related errors"""
    pass


# Global security manager instance
security_manager = SecurityManager()
secure_cache = SecureCache(security_manager)


# Utility functions for easy integration
def encrypt_sensitive_data(data: Any, context: str = "general") -> Dict[str, str]:
    """Convenience function to encrypt sensitive data"""
    return security_manager.encrypt_data(data, context)

def decrypt_sensitive_data(encrypted_package: Dict[str, str]) -> Any:
    """Convenience function to decrypt sensitive data"""
    return security_manager.decrypt_data(encrypted_package)

def verify_data_integrity(data: Any) -> str:
    """Convenience function to create integrity hash"""
    return security_manager.create_integrity_hash(data)

def secure_log(log_data: Dict) -> str:
    """Convenience function to create secure log entry"""
    return security_manager.secure_log_entry(log_data)