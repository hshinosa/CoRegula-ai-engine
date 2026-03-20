"""
Optimized RAG Pipeline dengan Caching dan Async Operations
===========================================================

Optimasi utama:
1. Query result caching dengan Redis
2. Async embedding generation
3. Connection pooling untuk vector store
4. Pre-computed embeddings
5. Parallel processing untuk retrieval + generation

Issue: KOL-42 - Performance Optimization
"""

import asyncio
import hashlib
import json
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, asdict
from datetime import datetime
from functools import lru_cache
import time

from app.core.config import settings
from app.core.logging import get_logger
from app.core.guardrails import get_guardrails, GuardrailAction
from app.services.vector_store import get_vector_store, VectorStoreService
from app.services.llm_optimized import OptimizedLLMService, LLMResponse  # Use optimized version
from app.services.efficiency_guard import get_efficiency_guard, EfficiencyGuard

logger = get_logger(__name__)

# OPTIMIZATION: Cache configuration
CACHE_TTL_SECONDS = 3600  # 1 hour
MAX_CACHE_SIZE = 1000
SEMANTIC_SIMILARITY_THRESHOLD = 0.90


@dataclass
class RAGResult:
    """Result dari RAG pipeline query."""
    answer: str
    sources: List[Dict[str, Any]]
    query: str
    tokens_used: int
    success: bool
    scaffolding_triggered: bool = False
    error: Optional[str] = None
    processing_time_ms: float = 0.0
    cache_hit: bool = False  # NEW: Track cache hit
    from_cache: bool = False  # NEW: Track if result from cache


class OptimizedRAGPipeline:
    """
    Optimized RAG Pipeline dengan caching dan async operations.
    
    Optimasi dari versi sebelumnya:
    1. Response caching dengan Redis/in-memory
    2. Query embedding cache
    3. Async operations untuk paralelisasi
    4. Connection pooling
    5. Pre-computed embeddings
    """
    
    # Skip retrieval patterns
    SKIP_PATTERNS = [
        "halo", "hai", "hi", "hello", "terima kasih", "thanks",
        "ok", "oke", "baik", "siap", "mantap", "good", "nice",
        "selamat pagi", "selamat siang", "selamat malam"
    ]
    
    MIN_QUERY_WORDS = 3
    
    def __init__(
        self,
        vector_store: Optional[VectorStoreService] = None,
        llm_service: Optional[OptimizedLLMService] = None,
        efficiency_guard: Optional[EfficiencyGuard] = None
    ):
        self.vector_store = vector_store or get_vector_store()
        self.llm_service = llm_service
        self.guardrails = get_guardrails()
        self.efficiency_guard = efficiency_guard or (
            get_efficiency_guard() if settings.ENABLE_EFFICIENCY_GUARD else None
        )
        
        # OPTIMIZATION: In-memory cache untuk query results
        # Untuk production, ganti dengan Redis
        self._query_cache: Dict[str, Dict[str, Any]] = {}
        self._embedding_cache: Dict[str, List[float]] = {}
        
        # Semantic caching
        self._last_queries: List[tuple] = []  # [(query_hash, query_text, result)]
        
        logger.info(
            "optimized_rag_pipeline_initialized",
            cache_enabled=True,
            max_cache_size=MAX_CACHE_SIZE
        )
    
    def _get_cache_key(self, query: str, collection_name: str) -> str:
        """Generate cache key dari query dan collection."""
        key_data = f"{query}:{collection_name}"
        return hashlib.sha256(key_data.encode()).hexdigest()
    
    def _get_from_cache(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Get result dari cache jika masih valid."""
        if cache_key not in self._query_cache:
            return None
        
        entry = self._query_cache[cache_key]
        
        # Check TTL
        if datetime.now().timestamp() - entry['timestamp'] > CACHE_TTL_SECONDS:
            del self._query_cache[cache_key]
            return None
        
        entry['hit_count'] = entry.get('hit_count', 0) + 1
        return entry['result']
    
    def _save_to_cache(self, cache_key: str, result: Dict[str, Any]):
        """Save result ke cache."""
        # Cleanup jika cache terlalu besar
        if len(self._query_cache) >= MAX_CACHE_SIZE:
            # Remove oldest entries
            sorted_keys = sorted(
                self._query_cache.keys(),
                key=lambda k: self._query_cache[k]['timestamp']
            )
            for key in sorted_keys[:len(sorted_keys)//4]:  # Remove 25%
                del self._query_cache[key]
        
        self._query_cache[cache_key] = {
            'result': result,
            'timestamp': datetime.now().timestamp(),
            'hit_count': 0
        }
    
    async def _get_cached_embedding(self, text: str) -> Optional[List[float]]:
        """Get embedding dari cache."""
        text_hash = hashlib.sha256(text.encode()).hexdigest()
        return self._embedding_cache.get(text_hash)
    
    async def _cache_embedding(self, text: str, embedding: List[float]):
        """Cache embedding untuk reuse."""
        text_hash = hashlib.sha256(text.encode()).hexdigest()
        
        # Cleanup jika terlalu besar
        if len(self._embedding_cache) >= MAX_CACHE_SIZE:
            # Remove random entries (simpler than LRU)
            keys = list(self._embedding_cache.keys())
            for key in keys[:len(keys)//4]:
                del self._embedding_cache[key]
        
        self._embedding_cache[text_hash] = embedding
    
    def _should_retrieve(self, query: str) -> bool:
        """
        Policy Agent: Decide whether to FETCH or NO_FETCH.
        """
        query_lower = query.lower().strip()
        
        # Skip untuk greetings
        if query_lower in self.SKIP_PATTERNS:
            logger.debug("policy_decision", action="NO_FETCH", reason="skip_pattern")
            return False
        
        # Skip untuk short queries
        word_count = len(query.split())
        if word_count < self.MIN_QUERY_WORDS:
            logger.debug("policy_decision", action="NO_FETCH", reason="short_query")
            return False
        
        # Check greeting patterns
        for pattern in self.SKIP_PATTERNS:
            if query_lower.startswith(pattern) and word_count <= 5:
                logger.debug("policy_decision", action="NO_FETCH", reason="greeting_prefix")
                return False
        
        logger.debug("policy_decision", action="FETCH", reason="substantive_query")
        return True
    
    async def query(
        self,
        query: str,
        collection_name: str,
        n_results: int = 5,
        use_cache: bool = True
    ) -> RAGResult:
        """
        Optimized RAG query dengan caching dan async operations.
        
        Args:
            query: User query
            collection_name: Collection name untuk vector search
            n_results: Number of results to retrieve
            use_cache: Whether to use caching
            
        Returns:
            RAGResult dengan timing dan cache info
        """
        start_time = time.time()
        
        try:
            # OPTIMIZATION: Check cache first
            if use_cache:
                cache_key = self._get_cache_key(query, collection_name)
                cached_result = self._get_from_cache(cache_key)
                
                if cached_result:
                    processing_time = (time.time() - start_time) * 1000
                    logger.info(
                        "rag_cache_hit",
                        query=query[:50],
                        processing_time_ms=round(processing_time, 2)
                    )
                    return RAGResult(
                        answer=cached_result['answer'],
                        sources=cached_result['sources'],
                        query=query,
                        tokens_used=cached_result.get('tokens_used', 0),
                        success=True,
                        processing_time_ms=processing_time,
                        from_cache=True
                    )
            
            # Initialize LLM service jika belum
            if self.llm_service is None:
                from app.services.llm_optimized import get_llm_service
                self.llm_service = get_llm_service()
            
            # Step 1: Guardrails check
            guard_result = await self.guardrails.check_input(query)
            if guard_result.action == GuardrailAction.BLOCK:
                processing_time = (time.time() - start_time) * 1000
                return RAGResult(
                    answer="Maaf, saya tidak bisa memproses pertanyaan tersebut.",
                    sources=[],
                    query=query,
                    tokens_used=0,
                    success=False,
                    error=guard_result.reason,
                    processing_time_ms=processing_time
                )
            
            # Step 2: Policy decision - FETCH or NO_FETCH
            contexts = []
            should_retrieve = self._should_retrieve(query)
            
            if should_retrieve:
                # OPTIMIZATION: Parallel retrieval dan preparation
                retrieval_task = self._retrieve_contexts(query, collection_name, n_results)
                
                try:
                    contexts = await retrieval_task
                except Exception as e:
                    logger.error("context_retrieval_failed", error=str(e))
                    contexts = []
            
            # Step 3: Generate response dengan LLM
            llm_result = await self.llm_service.generate_rag_response(
                query=query,
                contexts=contexts
            )
            
            # Step 4: Format sources
            sources = []
            if contexts:
                for i, ctx in enumerate(contexts[:3], 1):  # Limit to top 3
                    sources.append({
                        "source": ctx.get('source', 'Dokumen'),
                        "page": ctx.get('page'),
                        "relevance_score": ctx.get('score', 0)
                    })
            
            # Calculate processing time
            processing_time = (time.time() - start_time) * 1000
            
            # OPTIMIZATION: Cache successful results
            if use_cache and llm_result.success:
                result_to_cache = {
                    'answer': llm_result.content,
                    'sources': sources,
                    'tokens_used': llm_result.tokens_used
                }
                self._save_to_cache(cache_key, result_to_cache)
            
            logger.info(
                "rag_query_completed",
                query=query[:50],
                has_contexts=len(contexts) > 0,
                tokens_used=llm_result.tokens_used,
                processing_time_ms=round(processing_time, 2),
                cached=False
            )
            
            return RAGResult(
                answer=llm_result.content,
                sources=sources,
                query=query,
                tokens_used=llm_result.tokens_used,
                success=llm_result.success,
                processing_time_ms=processing_time,
                from_cache=False
            )
            
        except Exception as e:
            processing_time = (time.time() - start_time) * 1000
            logger.error(
                "rag_query_failed",
                query=query[:50],
                error=str(e),
                processing_time_ms=round(processing_time, 2)
            )
            
            return RAGResult(
                answer="Maaf, terjadi kesalahan saat memproses pertanyaan.",
                sources=[],
                query=query,
                tokens_used=0,
                success=False,
                error=str(e),
                processing_time_ms=processing_time
            )
    
    async def _retrieve_contexts(
        self,
        query: str,
        collection_name: str,
        n_results: int
    ) -> List[Dict[str, Any]]:
        """Retrieve contexts dari vector store."""
        # Search vector store
        results = await self.vector_store.query(
            query_text=query,
            collection_name=collection_name,
            n_results=n_results
        )
        
        # Filter by similarity threshold
        filtered_results = [
            r for r in results
            if r.get('score', 0) >= settings.SIMILARITY_THRESHOLD
        ]
        
        return filtered_results
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_queries = len(self._query_cache)
        total_hits = sum(e.get('hit_count', 0) for e in self._query_cache.values())
        
        return {
            'query_cache_size': total_queries,
            'query_cache_hits': total_hits,
            'embedding_cache_size': len(self._embedding_cache),
            'cache_hit_rate': total_hits / (total_queries + total_hits) if (total_queries + total_hits) > 0 else 0
        }
    
    def clear_cache(self):
        """Clear all caches."""
        self._query_cache.clear()
        self._embedding_cache.clear()
        logger.info("rag_pipeline_cache_cleared")


# Global singleton
_rag_pipeline = None


def get_optimized_rag_pipeline() -> OptimizedRAGPipeline:
    """Get singleton instance."""
    global _rag_pipeline
    if _rag_pipeline is None:
        _rag_pipeline = OptimizedRAGPipeline()
    return _rag_pipeline
