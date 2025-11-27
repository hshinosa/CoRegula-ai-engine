"""
Gemini Embedding Service
========================
Embedding generation using Google Gemini API.
"""

from typing import List
import google.generativeai as genai
from tenacity import retry, stop_after_attempt, wait_exponential

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)


class GeminiEmbeddingService:
    """Service for generating embeddings using Google Gemini."""
    
    def __init__(self):
        """Initialize Gemini embedding service."""
        self._initialized = False
        self._model = None
    
    def initialize(self) -> None:
        """Initialize the Gemini API client."""
        if self._initialized:
            return
            
        if not settings.GOOGLE_API_KEY:
            raise ValueError("GOOGLE_API_KEY is required")
        
        genai.configure(api_key=settings.GOOGLE_API_KEY)
        self._model = settings.GEMINI_EMBEDDING_MODEL
        self._initialized = True
        logger.info("Gemini embedding service initialized", model=self._model)
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
    )
    def embed_text(self, text: str) -> List[float]:
        """
        Generate embedding for a single text.
        
        Args:
            text: Text to embed
            
        Returns:
            List of embedding values
        """
        if not self._initialized:
            self.initialize()
        
        result = genai.embed_content(
            model=self._model,
            content=text,
            task_type="retrieval_document",
        )
        return result["embedding"]
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
    )
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        if not self._initialized:
            self.initialize()
        
        embeddings = []
        for text in texts:
            result = genai.embed_content(
                model=self._model,
                content=text,
                task_type="retrieval_document",
            )
            embeddings.append(result["embedding"])
        
        return embeddings
    
    def embed_query(self, query: str) -> List[float]:
        """
        Generate embedding for a query (optimized for retrieval).
        
        Args:
            query: Query text to embed
            
        Returns:
            Query embedding vector
        """
        if not self._initialized:
            self.initialize()
        
        result = genai.embed_content(
            model=self._model,
            content=query,
            task_type="retrieval_query",
        )
        return result["embedding"]


# Singleton instance
_embedding_service: GeminiEmbeddingService = None


def get_embedding_service() -> GeminiEmbeddingService:
    """Get or create the embedding service singleton."""
    global _embedding_service
    if _embedding_service is None:
        _embedding_service = GeminiEmbeddingService()
    return _embedding_service
