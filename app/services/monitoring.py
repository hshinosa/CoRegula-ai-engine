"""
Performance Monitoring Service
==============================
Prometheus metrics export and performance tracking.

KOL-139: Performance Monitoring with Prometheus Metrics
"""

from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
from datetime import datetime
from typing import Dict, Any
import time
import asyncio

from app.core.logging import get_logger

logger = get_logger(__name__)

# ============================================================================
# Metrics Definitions
# ============================================================================

# Request metrics
REQUEST_COUNT = Counter(
    'ai_engine_requests_total',
    'Total HTTP requests',
    ['method', 'endpoint', 'status']
)

REQUEST_LATENCY = Histogram(
    'ai_engine_request_latency_seconds',
    'HTTP request latency',
    ['method', 'endpoint'],
    buckets=[0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0, 30.0]
)

# LLM metrics
LLM_CALL_COUNT = Counter(
    'ai_engine_llm_calls_total',
    'Total LLM API calls',
    ['model', 'status']
)

LLM_CALL_LATENCY = Histogram(
    'ai_engine_llm_call_latency_seconds',
    'LLM API call latency',
    ['model'],
    buckets=[0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0]
)

# RAG metrics
RAG_QUERY_COUNT = Counter(
    'ai_engine_rag_queries_total',
    'Total RAG queries',
    ['collection', 'action']  # FETCH or NO_FETCH
)

RAG_QUERY_LATENCY = Histogram(
    'ai_engine_rag_query_latency_seconds',
    'RAG query latency',
    ['action'],
    buckets=[0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0]
)

# Cache metrics
CACHE_HIT_COUNT = Counter(
    'ai_engine_cache_hits_total',
    'Cache hits',
    ['cache_type']  # response_cache, embedding_cache, vector_cache
)

CACHE_MISS_COUNT = Counter(
    'ai_engine_cache_misses_total',
    'Cache misses',
    ['cache_type']
)

# Connection metrics
ACTIVE_CONNECTIONS = Gauge(
    'ai_engine_active_connections',
    'Active WebSocket connections'
)

MONGO_POOL_SIZE = Gauge(
    'ai_engine_mongo_pool_size',
    'MongoDB connection pool size'
)

# Circuit Breaker metrics
CIRCUIT_BREAKER_STATE = Gauge(
    'ai_engine_circuit_breaker_state',
    'Circuit breaker state (0=closed, 1=open, 2=half_open)',
    ['service']
)

# Error metrics
ERROR_COUNT = Counter(
    'ai_engine_errors_total',
    'Total errors',
    ['type', 'endpoint']
)

# ============================================================================
# Performance Monitor
# ============================================================================

class PerformanceMonitor:
    """
    Performance monitoring service with Prometheus metrics.
    
    Features:
    - Request/response tracking
    - LLM call monitoring
    - RAG query tracking
    - Cache hit/miss tracking
    - Circuit breaker state monitoring
    - Error tracking
    """
    
    def __init__(self):
        self.start_time = datetime.now()
        self._lock = asyncio.Lock()
        
        logger.info("performance_monitor_initialized")
    
    def track_request(self, method: str, endpoint: str):
        """
        Context manager for tracking HTTP request latency.
        
        Usage:
            with monitor.track_request('POST', '/api/rag/query'):
                # process request
        """
        return RequestTracker(method, endpoint)
    
    def track_llm_call(self, model: str):
        """
        Context manager for tracking LLM API call latency.
        
        Usage:
            with monitor.track_llm_call('gpt-4.7'):
                # call LLM API
        """
        return LLMCallTracker(model)
    
    def track_rag_query(self, action: str):
        """
        Context manager for tracking RAG query latency.
        
        Usage:
            with monitor.track_rag_query('FETCH'):
                # execute RAG query
        """
        return RAGQueryTracker(action)
    
    def record_cache_hit(self, cache_type: str):
        """Record cache hit."""
        CACHE_HIT_COUNT.labels(cache_type=cache_type).inc()
    
    def record_cache_miss(self, cache_type: str):
        """Record cache miss."""
        CACHE_MISS_COUNT.labels(cache_type=cache_type).inc()
    
    def record_rag_query(self, collection: str, action: str):
        """Record RAG query."""
        RAG_QUERY_COUNT.labels(collection=collection, action=action).inc()
    
    def record_error(self, error_type: str, endpoint: str):
        """Record error."""
        ERROR_COUNT.labels(type=error_type, endpoint=endpoint).inc()
    
    def update_circuit_breaker_state(self, service: str, state: int):
        """
        Update circuit breaker state gauge.
        
        Args:
            service: Service name (e.g., 'llm_service')
            state: 0=closed, 1=open, 2=half_open
        """
        CIRCUIT_BREAKER_STATE.labels(service=service).set(state)
    
    def update_active_connections(self, count: int):
        """Update active connection count."""
        ACTIVE_CONNECTIONS.set(count)
    
    def get_metrics(self) -> str:
        """Get Prometheus metrics in text format."""
        return generate_latest()
    
    def get_content_type(self) -> str:
        """Get Prometheus metrics content type."""
        return CONTENT_TYPE_LATEST
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get data for dashboard/API response."""
        uptime = (datetime.now() - self.start_time).total_seconds()
        
        return {
            'uptime_seconds': uptime,
            'uptime_human': self._format_uptime(uptime),
            'metrics_available': True,
            'metrics_endpoint': '/metrics'
        }
    
    def _format_uptime(self, seconds: float) -> str:
        """Format uptime in human-readable format."""
        days = int(seconds // 86400)
        hours = int((seconds % 86400) // 3600)
        minutes = int((seconds % 3600) // 60)
        
        parts = []
        if days > 0:
            parts.append(f"{days}d")
        if hours > 0:
            parts.append(f"{hours}h")
        if minutes > 0:
            parts.append(f"{minutes}m")
        
        return ' '.join(parts) if parts else '< 1m'


# ============================================================================
# Context Managers
# ============================================================================

class RequestTracker:
    """Context manager for HTTP request tracking."""
    
    def __init__(self, method: str, endpoint: str):
        self.method = method
        self.endpoint = endpoint
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = time.time() - self.start_time
        status = 'error' if exc_type else 'success'
        
        REQUEST_COUNT.labels(
            method=self.method,
            endpoint=self.endpoint,
            status=status
        ).inc()
        
        REQUEST_LATENCY.labels(
            method=self.method,
            endpoint=self.endpoint
        ).observe(duration)


class LLMCallTracker:
    """Context manager for LLM API call tracking."""
    
    def __init__(self, model: str):
        self.model = model
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = time.time() - self.start_time
        status = 'error' if exc_type else 'success'
        
        LLM_CALL_COUNT.labels(
            model=self.model,
            status=status
        ).inc()
        
        LLM_CALL_LATENCY.labels(model=self.model).observe(duration)


class RAGQueryTracker:
    """Context manager for RAG query tracking."""
    
    def __init__(self, action: str):
        self.action = action
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = time.time() - self.start_time
        
        RAG_QUERY_LATENCY.labels(action=self.action).observe(duration)


# ============================================================================
# Singleton
# ============================================================================

_monitor: PerformanceMonitor = None

def get_monitor() -> PerformanceMonitor:
    """Get PerformanceMonitor singleton."""
    global _monitor
    if _monitor is None:
        _monitor = PerformanceMonitor()
    return _monitor
