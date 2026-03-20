"""
Embedding Service
=================
Generates vector embeddings using OpenAI Compatible API (GLM-4.7).
Supports single text, batch texts, and optimized query embedding.
"""

from typing import List
from openai import AsyncOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)


class OpenAIEmbeddingService:
    """Service for generating embeddings using OpenAI Compatible API."""
    
    def __init__(self):
        """Initialize OpenAI embedding service."""
        self._initialized = False
        self._client = None
        self._model = None
    
    def initialize(self) -> None:
        """Initialize the OpenAI API client."""
        if self._initialized:
            return
            
        if not settings.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY is required")
        
        self._client = AsyncOpenAI(
            api_key=settings.OPENAI_API_KEY,
            base_url=settings.OPENAI_BASE_URL
        )
        self._model = settings.OPENAI_EMBEDDING_MODEL
        self._initialized = True
        logger.info("OpenAI embedding service initialized", model=self._model, base_url=settings.OPENAI_BASE_URL)
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
    )
    async def embed_text(self, text: str) -> List[float]:
        """
        Generate embedding for a single text.
        
        Args:
            text: Text to embed
            
        Returns:
            List of embedding values
        """
        if not self._initialized:
            self.initialize()
        
        response = await self._client.embeddings.create(
            model=self._model,
            input=text
        )
        return response.data[0].embedding
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
    )
    async def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        if not self._initialized:
            self.initialize()
        
        response = await self._client.embeddings.create(
            model=self._model,
            input=texts
        )
        return [item.embedding for item in response.data]
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
    )
    async def embed_query(self, query: str) -> List[float]:
        """
        Generate embedding for a query (optimized for retrieval).
        
        Args:
            query: Query text to embed
            
        Returns:
            Query embedding vector
        """
        if not self._initialized:
            self.initialize()
        
        response = await self._client.embeddings.create(
            model=self._model,
            input=query
        )
        return response.data[0].embedding

    async def get_embedding(self, text: str) -> List[float]:
        """Alias for embed_text for compatibility with existing code."""
        return await self.embed_text(text)


# Singleton instance
_embedding_service: OpenAIEmbeddingService = None


def get_embedding_service() -> OpenAIEmbeddingService:
    """Get or create the embedding service singleton."""
    global _embedding_service
    if _embedding_service is None:
        _embedding_service = OpenAIEmbeddingService()
    return _embedding_service
