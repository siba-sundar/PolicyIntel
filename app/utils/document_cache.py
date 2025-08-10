# document_cache.py
import hashlib
import json
import os
import time
import logging
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import pickle
import threading
from concurrent.futures import ThreadPoolExecutor
import gc
import psutil

import hashlib
import pickle
import logging
import os
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import asdict
from app.processors.pdf_processor import ExtractedLink

logger = logging.getLogger(__name__)

class DocumentCache:
    """
    High-performance document caching system for large documents
    Only activates for documents with significant content (configurable threshold)
    """
    
    def __init__(self, 
                 cache_dir: str = "cache",
                 min_cache_threshold: int = 50000,  # Minimum characters to enable caching
                 max_cache_size_mb: int = 500,      # Maximum total cache size
                 cache_expiry_hours: int = 24,      # Cache expiry time
                 enable_compression: bool = True):
        
        self.cache_dir = Path(cache_dir)
        self.min_cache_threshold = min_cache_threshold
        self.max_cache_size_mb = max_cache_size_mb
        self.cache_expiry_seconds = cache_expiry_hours * 3600
        self.enable_compression = enable_compression
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Create cache directory
        self.cache_dir.mkdir(exist_ok=True)
        
        # Cache metadata file
        self.metadata_file = self.cache_dir / "cache_metadata.json"
        self.metadata = self._load_metadata()
        
        # Background cleanup
        self._last_cleanup = time.time()
        self._cleanup_interval = 3600  # 1 hour
        
        logger.info(f"📦 Document Cache initialized: {cache_dir}, threshold: {min_cache_threshold} chars, max size: {max_cache_size_mb}MB")
    
    def _load_metadata(self) -> Dict:
        """Load cache metadata"""
        try:
            if self.metadata_file.exists():
                with open(self.metadata_file, 'r') as f:
                    metadata = json.load(f)
                    logger.info(f"📊 Loaded cache metadata: {len(metadata)} entries")
                    return metadata
        except Exception as e:
            logger.warning(f"Failed to load cache metadata: {e}")
        
        return {}
    
    def _save_metadata(self):
        """Save cache metadata"""
        try:
            with open(self.metadata_file, 'w') as f:
                json.dump(self.metadata, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save cache metadata: {e}")
    
    def _generate_cache_key(self, url: str, content_hash: str = None) -> str:
        """Generate unique cache key for document"""
        if content_hash:
            # Use content hash if available (more reliable)
            key_data = f"{url}:{content_hash}"
        else:
            # Use URL only
            key_data = url
        
        return hashlib.sha256(key_data.encode()).hexdigest()[:16]
    
    def _get_content_hash(self, content: str) -> str:
        """Generate hash of document content"""
        return hashlib.md5(content.encode()).hexdigest()
    
    def _should_cache(self, content: str) -> bool:
        """Determine if document should be cached based on size"""
        content_length = len(content)
        should_cache = content_length >= self.min_cache_threshold
        
        if should_cache:
            logger.info(f"📦 Document qualifies for caching: {content_length:,} characters (threshold: {self.min_cache_threshold:,})")
        else:
            logger.info(f"📋 Document too small for caching: {content_length:,} characters (threshold: {self.min_cache_threshold:,})")
        
        return should_cache
    
    def _get_cache_file_path(self, cache_key: str, data_type: str) -> Path:
        """Get cache file path for specific data type"""
        return self.cache_dir / f"{cache_key}_{data_type}.pkl"
    
    def _serialize_data(self, data: Any) -> bytes:
        """Serialize data with optional compression"""
        serialized = pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL)
        
        if self.enable_compression:
            import gzip
            serialized = gzip.compress(serialized)
        
        return serialized
    
    def _deserialize_data(self, data: bytes) -> Any:
        """Deserialize data with optional decompression"""
        if self.enable_compression:
            import gzip
            data = gzip.decompress(data)
        
        return pickle.loads(data)
    
    def _cleanup_expired_cache(self):
        """Clean up expired cache entries"""
        current_time = time.time()
        
        # Only cleanup if enough time has passed
        if current_time - self._last_cleanup < self._cleanup_interval:
            return
        
        logger.info("🧹 Starting cache cleanup...")
        
        expired_keys = []
        total_size_before = 0
        
        for cache_key, entry in self.metadata.items():
            entry_time = entry.get('timestamp', 0)
            entry_size = entry.get('size_mb', 0)
            total_size_before += entry_size
            
            if current_time - entry_time > self.cache_expiry_seconds:
                expired_keys.append(cache_key)
        
        # Remove expired entries
        removed_size = 0
        for cache_key in expired_keys:
            removed_size += self._remove_cache_entry(cache_key)
        
        self._last_cleanup = current_time
        
        if expired_keys:
            logger.info(f"🧹 Cache cleanup completed: removed {len(expired_keys)} entries, freed {removed_size:.2f}MB")
        
        # Check total cache size
        self._enforce_cache_size_limit()
    
    def _enforce_cache_size_limit(self):
        """Enforce maximum cache size by removing oldest entries"""
        total_size = sum(entry.get('size_mb', 0) for entry in self.metadata.values())
        
        if total_size <= self.max_cache_size_mb:
            return
        
        logger.info(f"📦 Cache size ({total_size:.2f}MB) exceeds limit ({self.max_cache_size_mb}MB), removing oldest entries...")
        
        # Sort entries by timestamp (oldest first)
        sorted_entries = sorted(
            self.metadata.items(),
            key=lambda x: x[1].get('timestamp', 0)
        )
        
        removed_size = 0
        removed_count = 0
        
        for cache_key, entry in sorted_entries:
            if total_size - removed_size <= self.max_cache_size_mb * 0.8:  # Keep 80% of limit
                break
            
            entry_size = self._remove_cache_entry(cache_key)
            removed_size += entry_size
            removed_count += 1
        
        logger.info(f"📦 Cache size enforcement: removed {removed_count} entries, freed {removed_size:.2f}MB")
    
    def _remove_cache_entry(self, cache_key: str) -> float:
        """Remove a single cache entry and return freed size"""
        if cache_key not in self.metadata:
            return 0
        
        entry = self.metadata[cache_key]
        entry_size = entry.get('size_mb', 0)
        
        # Remove files
        data_types = ['document_text', 'chunks', 'embeddings']
        for data_type in data_types:
            cache_file = self._get_cache_file_path(cache_key, data_type)
            try:
                if cache_file.exists():
                    cache_file.unlink()
            except Exception as e:
                logger.warning(f"Failed to remove cache file {cache_file}: {e}")
        
        # Remove from metadata
        del self.metadata[cache_key]
        self._save_metadata()
        
        return entry_size
    
    def _calculate_data_size_mb(self, data: Any) -> float:
        """Calculate approximate size of data in MB"""
        try:
            serialized = self._serialize_data(data)
            return len(serialized) / (1024 * 1024)
        except Exception:
            return 0
    
    def is_cached(self, url: str, content_hash: str = None) -> bool:
        """Check if document is cached and valid"""
        with self._lock:
            try:
                cache_key = self._generate_cache_key(url, content_hash)
                
                if cache_key not in self.metadata:
                    return False
                
                entry = self.metadata[cache_key]
                
                # Check expiry
                current_time = time.time()
                if current_time - entry.get('timestamp', 0) > self.cache_expiry_seconds:
                    logger.info(f"🕐 Cache entry expired for key: {cache_key}")
                    self._remove_cache_entry(cache_key)
                    return False
                
                # Check if all required files exist
                data_types = ['document_text', 'chunks', 'embeddings']
                for data_type in data_types:
                    cache_file = self._get_cache_file_path(cache_key, data_type)
                    if not cache_file.exists():
                        logger.warning(f"📁 Cache file missing: {cache_file}")
                        self._remove_cache_entry(cache_key)
                        return False
                
                logger.info(f"✅ Cache hit for document: {url[:100]}...")
                return True
                
            except Exception as e:
                logger.error(f"Error checking cache: {e}")
                return False
    
    def save_to_cache(self, url: str, document_text: str, chunks: List[Dict], 
                     embeddings: List[List[float]], extracted_links: List = None,
                     fetched_content: Dict = None) -> bool:
        """Save document processing results to cache"""
        
        # Check if document qualifies for caching
        if not self._should_cache(document_text):
            return False
        
        with self._lock:
            try:
                # Cleanup before saving
                self._cleanup_expired_cache()
                
                content_hash = self._get_content_hash(document_text)
                cache_key = self._generate_cache_key(url, content_hash)
                
                logger.info(f"💾 Caching document: {url[:100]}... (key: {cache_key})")
                
                # Calculate sizes
                doc_size = self._calculate_data_size_mb(document_text)
                chunks_size = self._calculate_data_size_mb(chunks)
                embeddings_size = self._calculate_data_size_mb(embeddings)
                links_size = self._calculate_data_size_mb(extracted_links) if extracted_links else 0
                content_size = self._calculate_data_size_mb(fetched_content) if fetched_content else 0
                total_size = doc_size + chunks_size + embeddings_size + links_size + content_size
                
                # Save data files
                cache_files = {
                    'document_text': document_text,
                    'chunks': chunks,
                    'embeddings': embeddings,
                    'extracted_links': extracted_links,
                    'fetched_content': fetched_content
                }
                
                saved_files = []
                for data_type, data in cache_files.items():
                    if data is not None:  # Only save non-None data
                        cache_file = self._get_cache_file_path(cache_key, data_type)
                        try:
                            serialized_data = self._serialize_data(data)
                            with open(cache_file, 'wb') as f:
                                f.write(serialized_data)
                            saved_files.append(data_type)
                            logger.info(f"💾 Saved {data_type} to cache ({len(serialized_data) / 1024 / 1024:.2f}MB)")
                        except Exception as e:
                            logger.error(f"Failed to save {data_type} to cache: {e}")
                            # Clean up partially saved files
                            for saved_type in saved_files:
                                try:
                                    self._get_cache_file_path(cache_key, saved_type).unlink()
                                except:
                                    pass
                            return False
                
                # Update metadata
                self.metadata[cache_key] = {
                    'url': url,
                    'content_hash': content_hash,
                    'timestamp': time.time(),
                    'size_mb': total_size,
                    'doc_length': len(document_text),
                    'chunk_count': len(chunks),
                    'embedding_count': len(embeddings)
                }
                
                self._save_metadata()
                
                logger.info(f"✅ Document cached successfully: {total_size:.2f}MB, {len(chunks)} chunks, {len(embeddings)} embeddings")
                return True
                
            except Exception as e:
                logger.error(f"Failed to cache document: {e}")
                return False
    
    def load_from_cache(self, url: str, content_hash: str = None) -> Optional[Tuple[str, List[Dict], List[List[float]]]]:
        """Load document processing results from cache"""
        with self._lock:
            try:
                cache_key = self._generate_cache_key(url, content_hash)
                
                if not self.is_cached(url, content_hash):
                    return None
                
                logger.info(f"📦 Loading from cache: {url[:100]}... (key: {cache_key})")
                
                # Load data files
                data_types = ['document_text', 'chunks', 'embeddings']
                loaded_data = {}
                
                for data_type in data_types:
                    cache_file = self._get_cache_file_path(cache_key, data_type)
                    try:
                        with open(cache_file, 'rb') as f:
                            serialized_data = f.read()
                        loaded_data[data_type] = self._deserialize_data(serialized_data)
                        logger.info(f"📦 Loaded {data_type} from cache ({len(serialized_data) / 1024 / 1024:.2f}MB)")
                    except Exception as e:
                        logger.error(f"Failed to load {data_type} from cache: {e}")
                        return None
                
                # Update access time
                self.metadata[cache_key]['last_accessed'] = time.time()
                self._save_metadata()
                
                document_text = loaded_data['document_text']
                chunks = loaded_data['chunks']
                embeddings = loaded_data['embeddings']
                
                logger.info(f"✅ Cache load successful: {len(document_text):,} chars, {len(chunks)} chunks, {len(embeddings)} embeddings")
                
                return document_text, chunks, embeddings
                
            except Exception as e:
                logger.error(f"Failed to load from cache: {e}")
                return None
    
    def get_cache_stats(self) -> Dict:
        """Get cache statistics"""
        with self._lock:
            total_entries = len(self.metadata)
            total_size_mb = sum(entry.get('size_mb', 0) for entry in self.metadata.values())
            
            # Count by data types
            active_entries = 0
            expired_entries = 0
            current_time = time.time()
            
            for entry in self.metadata.values():
                if current_time - entry.get('timestamp', 0) > self.cache_expiry_seconds:
                    expired_entries += 1
                else:
                    active_entries += 1
            
            return {
                'total_entries': total_entries,
                'active_entries': active_entries,
                'expired_entries': expired_entries,
                'total_size_mb': round(total_size_mb, 2),
                'max_size_mb': self.max_cache_size_mb,
                'cache_utilization_percent': round((total_size_mb / self.max_cache_size_mb) * 100, 1),
                'min_cache_threshold_chars': self.min_cache_threshold,
                'expiry_hours': self.cache_expiry_seconds / 3600,
                'compression_enabled': self.enable_compression,
                'cache_directory': str(self.cache_dir)
            }
    
    def clear_cache(self, confirm: bool = False) -> Dict:
        """Clear all cache entries"""
        if not confirm:
            return {'error': 'Must set confirm=True to clear cache'}
        
        with self._lock:
            logger.info("🗑️ Clearing all cache entries...")
            
            removed_count = 0
            removed_size = 0
            
            for cache_key in list(self.metadata.keys()):
                removed_size += self._remove_cache_entry(cache_key)
                removed_count += 1
            
            # Clear metadata file
            self.metadata = {}
            self._save_metadata()
            
            logger.info(f"🗑️ Cache cleared: removed {removed_count} entries, freed {removed_size:.2f}MB")
            
            return {
                'cleared': True,
                'removed_entries': removed_count,
                'freed_size_mb': round(removed_size, 2)
            }
    
    def optimize_cache(self) -> Dict:
        """Optimize cache by removing expired entries and defragmenting"""
        with self._lock:
            logger.info("⚙️ Starting cache optimization...")
            
            # Force cleanup
            original_cleanup_interval = self._cleanup_interval
            self._cleanup_interval = 0  # Force cleanup
            self._cleanup_expired_cache()
            self._cleanup_interval = original_cleanup_interval
            
            # Force garbage collection
            collected = gc.collect()
            
            # Get final stats
            stats = self.get_cache_stats()
            
            logger.info(f"⚙️ Cache optimization completed: {stats['active_entries']} active entries, {stats['total_size_mb']}MB")
            
            return {
                'optimized': True,
                'garbage_collected': collected,
                'final_stats': stats
            }

# Factory function for easy integration
def create_document_cache(
    cache_dir: str = "cache",
    min_cache_threshold: int = 50000,  # 50K characters (roughly 25-30 pages)
    max_cache_size_mb: int = 500,
    cache_expiry_hours: int = 24,
    enable_compression: bool = True
) -> DocumentCache:
    """
    Factory function to create DocumentCache instance
    
    Args:
        cache_dir: Directory to store cache files
        min_cache_threshold: Minimum characters to enable caching (50K ≈ 25-30 pages)
        max_cache_size_mb: Maximum total cache size in MB
        cache_expiry_hours: How long to keep cache entries
        enable_compression: Whether to compress cache files
    """
    return DocumentCache(
        cache_dir=cache_dir,
        min_cache_threshold=min_cache_threshold,
        max_cache_size_mb=max_cache_size_mb,
        cache_expiry_hours=cache_expiry_hours,
        enable_compression=enable_compression
    )
    
    
    
    

class EnhancedDocumentCache:
    """Enhanced document cache that supports link and fetched content storage"""
    
    def __init__(self, cache_dir: str = "/tmp/enhanced_doc_cache"):
        self.cache_dir = cache_dir
        self.ensure_cache_directory()
        
    def ensure_cache_directory(self):
        """Ensure cache directory exists"""
        try:
            os.makedirs(self.cache_dir, exist_ok=True)
        except Exception as e:
            logger.warning(f"Could not create cache directory {self.cache_dir}: {e}")
    
    def _generate_cache_key(self, documents: List[Dict]) -> str:
        """Generate a unique cache key for document set"""
        doc_signatures = []
        for doc in documents:
            url = doc.get('url', '')
            filename = doc.get('filename', '')
            # Create a signature from URL and filename
            signature = f"{url}:{filename}"
            doc_signatures.append(signature)
        
        # Sort to ensure consistent key regardless of document order
        doc_signatures.sort()
        combined = "|".join(doc_signatures)
        
        return hashlib.md5(combined.encode()).hexdigest()
    
    def _get_cache_path(self, cache_key: str) -> str:
        """Get full path for cache file"""
        return os.path.join(self.cache_dir, f"{cache_key}.enhanced_cache")
    
    def save_to_cache(self, documents: List[Dict], document_text: str, 
                     chunks: List[Dict], chunk_embeddings: List, 
                     extracted_links: List[ExtractedLink] = None,
                     fetched_content: Dict[str, Any] = None) -> bool:
        """
        Save document processing results to cache including link data
        
        Args:
            documents: List of document configurations
            document_text: Processed document text
            chunks: Document chunks
            chunk_embeddings: Chunk embeddings
            extracted_links: Links extracted from the document
            fetched_content: Content fetched from links
        """
        try:
            cache_key = self._generate_cache_key(documents)
            cache_path = self._get_cache_path(cache_key)
            
            # Prepare data for caching
            cache_data = {
                'document_text': document_text,
                'chunks': chunks,
                'chunk_embeddings': chunk_embeddings,
                'extracted_links': [asdict(link) for link in extracted_links] if extracted_links else [],
                'fetched_content': fetched_content or {},
                'documents_signature': documents,
                'cache_version': '2.0'  # Enhanced version
            }
            
            # Save to cache
            with open(cache_path, 'wb') as f:
                pickle.dump(cache_data, f)
                
            logger.info(f"✅ Saved enhanced cache: {cache_key}")
            logger.info(f"   - Document text: {len(document_text)} chars")
            logger.info(f"   - Chunks: {len(chunks)}")
            logger.info(f"   - Links: {len(extracted_links) if extracted_links else 0}")
            logger.info(f"   - Fetched content: {len(fetched_content) if fetched_content else 0} items")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to save enhanced cache: {str(e)}")
            return False
    
    def load_from_cache(self, documents: List[Dict]) -> Optional[Tuple[str, List[Dict], List, List[ExtractedLink], Dict[str, Any]]]:
        """
        Load document processing results from cache including link data
        
        Returns:
            Tuple of (document_text, chunks, chunk_embeddings, extracted_links, fetched_content) or None
        """
        try:
            cache_key = self._generate_cache_key(documents)
            cache_path = self._get_cache_path(cache_key)
            
            if not os.path.exists(cache_path):
                return None
                
            with open(cache_path, 'rb') as f:
                cache_data = pickle.load(f)
            
            # Validate cache version and structure
            if cache_data.get('cache_version') != '2.0':
                logger.warning("Cache version mismatch, invalidating cache")
                return None
            
            # Reconstruct ExtractedLink objects
            extracted_links = []
            if cache_data.get('extracted_links'):
                for link_dict in cache_data['extracted_links']:
                    try:
                        link = ExtractedLink(**link_dict)
                        extracted_links.append(link)
                    except Exception as e:
                        logger.warning(f"Failed to reconstruct link object: {e}")
                        continue
            
            logger.info(f"✅ Loaded enhanced cache: {cache_key}")
            logger.info(f"   - Document text: {len(cache_data['document_text'])} chars")
            logger.info(f"   - Chunks: {len(cache_data['chunks'])}")
            logger.info(f"   - Links: {len(extracted_links)}")
            logger.info(f"   - Fetched content: {len(cache_data.get('fetched_content', {}))} items")
            
            return (
                cache_data['document_text'],
                cache_data['chunks'], 
                cache_data['chunk_embeddings'],
                extracted_links,
                cache_data.get('fetched_content', {})
            )
            
        except Exception as e:
            logger.error(f"Failed to load enhanced cache: {str(e)}")
            return None
    
    def clear_cache(self, confirm: bool = False) -> bool:
        """Clear all cached data"""
        if not confirm:
            logger.warning("Cache clear called without confirmation")
            return False
            
        try:
            if not os.path.exists(self.cache_dir):
                return True
                
            cache_files = [f for f in os.listdir(self.cache_dir) if f.endswith('.enhanced_cache')]
            
            for cache_file in cache_files:
                file_path = os.path.join(self.cache_dir, cache_file)
                try:
                    os.remove(file_path)
                except Exception as e:
                    logger.warning(f"Failed to remove cache file {cache_file}: {e}")
                    
            logger.info(f"🗑️ Cleared {len(cache_files)} enhanced cache files")
            return True
            
        except Exception as e:
            logger.error(f"Failed to clear enhanced cache: {str(e)}")
            return False
    
    def get_cache_info(self) -> Dict[str, Any]:
        """Get information about cached data"""
        try:
            if not os.path.exists(self.cache_dir):
                return {"cache_files": 0, "total_size": 0}
                
            cache_files = [f for f in os.listdir(self.cache_dir) if f.endswith('.enhanced_cache')]
            total_size = 0
            
            for cache_file in cache_files:
                file_path = os.path.join(self.cache_dir, cache_file)
                try:
                    total_size += os.path.getsize(file_path)
                except Exception:
                    continue
            
            return {
                "cache_files": len(cache_files),
                "total_size": total_size,
                "total_size_mb": round(total_size / (1024 * 1024), 2)
            }
            
        except Exception as e:
            logger.error(f"Failed to get cache info: {str(e)}")
            return {"error": str(e)}


# Global enhanced cache instance
enhanced_document_cache = EnhancedDocumentCache()


# Backwards compatibility wrapper for existing code
class DocumentCacheWrapper:
    """Wrapper to maintain backwards compatibility with existing cache code"""
    
    def __init__(self, enhanced_cache: EnhancedDocumentCache):
        self.enhanced_cache = enhanced_cache
    
    def save_to_cache(self, url: str, document_text: str, 
                     chunks: List[Dict], chunk_embeddings: List,
                     extracted_links: List = None, 
                     fetched_content: Dict = None) -> bool:
        """Backwards compatible save method"""
        # Convert single URL to document format expected by enhanced cache
        documents = [{'url': url}]
        return self.enhanced_cache.save_to_cache(
            documents, document_text, chunks, chunk_embeddings,
            extracted_links, fetched_content
        )
    
    def load_from_cache(self, url: str):
        """Backwards compatible load method"""
        # Convert single URL to document format expected by enhanced cache
        documents = [{'url': url}]
        result = self.enhanced_cache.load_from_cache(documents)
        if result:
            # Return only the first 3 elements for backwards compatibility
            return result[:3]
        return None
    
    def clear_cache(self, confirm: bool = False) -> bool:
        """Backwards compatible clear method"""
        return self.enhanced_cache.clear_cache(confirm)


# Create wrapper instance for backwards compatibility
document_cache = DocumentCacheWrapper(enhanced_document_cache)