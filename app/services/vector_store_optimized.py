"""
Optimized Vector Store dengan Caching
=====================================

Priority 2: Vector Search Result Caching

Features:
- Cache ChromaDB query results di Redis
- TTL 10 menit untuk query results
- Cache hit: ~5ms vs 100-300ms cache miss
"""

import asyncio
import hashlib
from typing import List, Dict, Any, Optional
from datetime import datetime

from app.core.config import settings
from app.core.logging import get_logger
from app.core.redis_cache import get_redis_cache, CACHE_TTL
from app.services.vector_store import VectorStoreService

logger = get_logger(__name__)


class OptimizedVectorStore(VectorStoreService):
    """Vector store dengan Redis caching layer."""
    
    def __init__(self):
        super().__init__()
        self._cache_initialized = False
    
    async def _ensure_cache(self):
        """Ensure Redis cache is ready."""
        if not self._cache_initialized:
            await get_redis_cache()
            self._cache_initialized = True
    
    def _generate_cache_key(self, query_text: str, collection_name: str, n_results: int) -> str:
        """Generate cache key untuk query."""
        key_data = f"{query_text}:{collection_name}:{n_results}"
        hash_value = hashlib.sha256(key_data.encode()).hexdigest()[:16]
        return f"vector:{collection_name}:{hash_value}"
    
    async def query(
        self,
        query_text: str,
        collection_name: str,
        n_results: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Query dengan caching.
        
        Cache hit: ~5ms
        Cache miss: 100-300ms
        """
        await self._ensure_cache()
        
        # Generate cache key
        cache_key = self._generate_cache_key(query_text, collection_name, n_results)
        
        # Try cache first
        try:
            redis_cache = await get_redis_cache()
            cached = await redis_cache.get(cache_key)
            
            if cached:
                logger.debug(
                    "vector_cache_hit",
                    collection=collection_name,
                    query=query_text[:30]
                )
                return cached
        except Exception as e:
            logger.warning("vector_cache_read_error", error=str(e))
        
        # Cache miss - query ChromaDB
        start_time = datetime.now()
        results = await super().query(query_text, collection_name, n_results)
        elapsed_ms = (datetime.now() - start_time).total_seconds() * 1000
        
        # Cache results
        if results:
            try:
                redis_cache = await get_redis_cache()
                await redis_cache.set(
                    cache_key,
                    results,
                    ttl=CACHE_TTL["vector_search"]
                )
                logger.debug(
                    "vector_cache_set",
                    collection=collection_name,
                    query=query_text[:30],
                    time_ms=elapsed_ms
                )
            except Exception as e:
                logger.warning("vector_cache_write_error", error=str(e))
        
        logger.info(
            "vector_query_completed",
            collection=collection_name,
            results=len(results),
            time_ms=elapsed_ms,
            cached=False
        )
        
        return results
    
    async def get_cache_stats(self, collection_name: str) -> Dict[str, Any]:
        """Get cache statistics untuk collection."""
        try:
            redis_cache = await get_redis_cache()
            # Get all keys matching pattern
            pattern = f"vector:{collection_name}:*"
            keys = await redis_cache._redis.keys(pattern)
            
            return {
                "collection": collection_name,
                "cached_queries": len(keys),
                "ttl_seconds": CACHE_TTL["vector_search"]
            }
        except Exception as e:
            return {"error": str(e)}
    
    async def clear_collection_cache(self, collection_name: str) -> int:
        """Clear cache untuk collection."""
        try:
            redis_cache = await get_redis_cache()
            pattern = f"vector:{collection_name}:*"
            return await redis_cache.clear_pattern(pattern)
        except Exception as e:
            logger.error("vector_cache_clear_error", error=str(e))
            return 0


# Global instance
_optimized_vector_store = None


async def get_optimized_vector_store() -> OptimizedVectorStore:
    """Get singleton instance."""
    global _optimized_vector_store
    if _optimized_vector_store is None:
        _optimized_vector_store = OptimizedVectorStore()
        await _optimized_vector_store.initialize()
    return _optimized_vector_store
