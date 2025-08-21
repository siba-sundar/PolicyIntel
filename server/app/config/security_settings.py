# app/config/security_settings.py
import os
from typing import Optional

class SecuritySettings:
    """Security configuration settings"""
    
    # Encryption settings
    SECURITY_MASTER_KEY: str = os.environ.get('SECURITY_MASTER_KEY', '')
    ENCRYPTION_ALGORITHM: str = 'AES-256-GCM'
    KEY_DERIVATION_ITERATIONS: int = int(os.environ.get('KEY_DERIVATION_ITERATIONS', '100000'))
    
    # Integrity verification settings
    INTEGRITY_ALGORITHM: str = 'HMAC-SHA256'
    ENABLE_INTEGRITY_VERIFICATION: bool = os.environ.get('ENABLE_INTEGRITY_VERIFICATION', 'true').lower() == 'true'
    
    # Secure cache settings
    ENABLE_SECURE_CACHE: bool = os.environ.get('ENABLE_SECURE_CACHE', 'true').lower() == 'true'
    SECURE_CACHE_TTL: int = int(os.environ.get('SECURE_CACHE_TTL', '3600'))  # 1 hour
    MAX_CACHE_ENTRIES: int = int(os.environ.get('MAX_CACHE_ENTRIES', '100'))
    
    # Audit logging settings
    ENABLE_AUDIT_LOGGING: bool = os.environ.get('ENABLE_AUDIT_LOGGING', 'true').lower() == 'true'
    AUDIT_LOG_LEVEL: str = os.environ.get('AUDIT_LOG_LEVEL', 'INFO')
    SECURE_LOG_ROTATION: bool = os.environ.get('SECURE_LOG_ROTATION', 'true').lower() == 'true'
    
    # Security thresholds
    MAX_DECRYPTION_ATTEMPTS: int = int(os.environ.get('MAX_DECRYPTION_ATTEMPTS', '3'))
    INTEGRITY_FAILURE_THRESHOLD: int = int(os.environ.get('INTEGRITY_FAILURE_THRESHOLD', '5'))
    
    # Performance settings
    ENCRYPT_LARGE_DATA_THRESHOLD: int = int(os.environ.get('ENCRYPT_LARGE_DATA_THRESHOLD', '1048576'))  # 1MB
    PARALLEL_ENCRYPTION: bool = os.environ.get('PARALLEL_ENCRYPTION', 'false').lower() == 'true'
    
    # Development/Debug settings
    ENABLE_SECURITY_DEBUG: bool = os.environ.get('ENABLE_SECURITY_DEBUG', 'false').lower() == 'true'
    SKIP_ENCRYPTION_IN_DEV: bool = os.environ.get('SKIP_ENCRYPTION_IN_DEV', 'false').lower() == 'true'
    
    @classmethod
    def validate_settings(cls) -> bool:
        """Validate security settings"""
        if not cls.SECURITY_MASTER_KEY and not cls.SKIP_ENCRYPTION_IN_DEV:
            raise ValueError("SECURITY_MASTER_KEY must be set in production")
        
        if cls.KEY_DERIVATION_ITERATIONS < 50000:
            raise ValueError("KEY_DERIVATION_ITERATIONS must be at least 50,000")
        
        return True
    
    @classmethod
    def get_environment_info(cls) -> dict:
        """Get security environment information"""
        return {
            'encryption_enabled': bool(cls.SECURITY_MASTER_KEY) or not cls.SKIP_ENCRYPTION_IN_DEV,
            'integrity_verification': cls.ENABLE_INTEGRITY_VERIFICATION,
            'secure_cache': cls.ENABLE_SECURE_CACHE,
            'audit_logging': cls.ENABLE_AUDIT_LOGGING,
            'encryption_algorithm': cls.ENCRYPTION_ALGORITHM,
            'integrity_algorithm': cls.INTEGRITY_ALGORITHM,
            'debug_mode': cls.ENABLE_SECURITY_DEBUG
        }

# Initialize and validate settings
security_settings = SecuritySettings()

try:
    security_settings.validate_settings()
    print("✅ Security settings validated successfully")
except ValueError as e:
    print(f"❌ Security settings validation failed: {e}")
    if not security_settings.SKIP_ENCRYPTION_IN_DEV:
        raise