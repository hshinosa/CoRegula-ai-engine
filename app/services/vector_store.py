"""
Vector Store Service
====================
Manages ChromaDB collections with persistent storage. 
Handles document indexing, similarity search, and collection lifecycle.
"""

import asyncio
from typing import List, Dict, Any, Optional
from pathlib import Path
import chromadb
from chromadb.config import Settings as ChromaSettings

# Disable PostHog telemetry to avoid noisy errors in offline environments
try:
    import posthog  # type: ignore

    posthog.disabled = True

    def _noop_capture(*args, **kwargs):  # type: ignore
        return None

    posthog.capture = _noop_capture  # type: ignore
except ImportError:
    posthog = None  # type: ignore

from app.core.config import settings
from app.core.logging import get_logger
from app.services.embeddings import get_embedding_service

logger = get_logger(__name__)


class OpenAIEmbeddingFunction:
    """ChromaDB compatible embedding function using OpenAI Compatible API."""
    
    def __call__(self, input: List[str]) -> List[List[float]]:
        """Generate embeddings for ChromaDB."""
        import asyncio
        embedding_service = get_embedding_service()
        
        # Run async method in event loop
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(embedding_service.embed_texts(input))


class VectorStoreService:
    """Service for managing ChromaDB vector store."""
    
    def __init__(self):
        """Initialize vector store service."""
        self._client: Optional[chromadb.ClientAPI] = None
        self._embedding_function = OpenAIEmbeddingFunction()
        self._collections: Dict[str, chromadb.Collection] = {}
        self._initialized = False
    
    async def initialize(self) -> None:
        """Initialize ChromaDB client and embedding service."""
        if self._initialized:
            return
        
        # Ensure persist directory exists
        persist_dir = Path(settings.CHROMA_PERSIST_DIR)
        persist_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize embedding service first
        embedding_service = get_embedding_service()
        embedding_service.initialize()
        
        # Initialize ChromaDB with persistence
        self._client = chromadb.PersistentClient(
            path=str(persist_dir),
            settings=ChromaSettings(
                anonymized_telemetry=False,
                allow_reset=True,
            ),
        )
        
        self._initialized = True
        logger.info("Vector store initialized", persist_dir=str(persist_dir))
    
    async def _ensure_collection(self, collection_name: str) -> chromadb.Collection:
        """
        Ensure a collection exists and return it.
        
        Args:
            collection_name: Name of the collection
            
        Returns:
            ChromaDB collection
        """
        if not self._client:
            await self.initialize()
        
        if collection_name in self._collections:
            return self._collections[collection_name]
        
        collection = self._client.get_or_create_collection(
            name=collection_name,
            embedding_function=self._embedding_function,
        )
        
        self._collections[collection_name] = collection
        logger.debug("Collection accessed", collection=collection_name)
        return collection
    
    def _get_collection_name(self, course_id: str) -> str:
        """Get collection name for a course."""
        return f"{settings.CHROMA_COLLECTION_PREFIX}_{course_id}"
    
    async def get_or_create_collection(self, course_id: str) -> chromadb.Collection:
        """
        Get or create a collection for a course.
        
        Args:
            course_id: Course identifier
            
        Returns:
            ChromaDB collection
        """
        collection_name = self._get_collection_name(course_id)
        return await self._ensure_collection(collection_name)
    
    async def add_documents(
        self,
        documents: List[str],
        metadatas: List[Dict[str, Any]],
        ids: List[str],
        collection_name: Optional[str] = None,
        course_id: Optional[str] = None,
    ) -> None:
        """
        Add documents to a collection.
        
        Args:
            documents: List of document texts
            metadatas: List of metadata dicts for each document
            ids: List of unique IDs for each document
            collection_name: Optional collection name
            course_id: Optional course ID (to derive collection name)
        """
        target_collection = collection_name or (
            self._get_collection_name(course_id) if course_id else "default"
        )
        
        collection = await self._ensure_collection(target_collection)
        
        # Run in thread to avoid blocking
        await asyncio.to_thread(
            collection.add,
            documents=documents,
            metadatas=metadatas,
            ids=ids,
        )
        
        logger.info(
            "Documents added to collection",
            collection=target_collection,
            count=len(documents),
        )
    
    async def search(
        self,
        query: str,
        collection_name: Optional[str] = None,
        n_results: int = 5,
        where: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search for similar documents.
        
        Args:
            query: Query string
            collection_name: Collection to search in
            n_results: Number of results to return
            where: Optional metadata filter
            
        Returns:
            List of results with content, metadata, and scores
        """
        target_collection = collection_name or "default"
        
        try:
            collection = await self._ensure_collection(target_collection)
        except Exception as e:
            logger.warning(
                "Collection not found for search",
                collection=target_collection,
                error=str(e)
            )
            return []
        
        # Build query kwargs
        query_kwargs = {
            "query_texts": [query],
            "n_results": n_results,
            "include": ["documents", "metadatas", "distances"],
        }
        
        if where:
            query_kwargs["where"] = where
        
        # Execute query in thread
        results = await asyncio.to_thread(
            collection.query,
            **query_kwargs
        )
        
        # Format results
        formatted = []
        if results and results.get("documents"):
            docs = results["documents"][0] if results["documents"] else []
            metas = results["metadatas"][0] if results.get("metadatas") else []
            distances = results["distances"][0] if results.get("distances") else []
            
            for i, doc in enumerate(docs):
                # Convert distance to similarity score (lower distance = higher similarity)
                score = 1 - (distances[i] if i < len(distances) else 0)
                formatted.append({
                    "content": doc,
                    "metadata": metas[i] if i < len(metas) else {},
                    "score": score,
                })
        
        logger.debug(
            "Search executed",
            collection=target_collection,
            query_length=len(query),
            results_count=len(formatted),
        )
        
        return formatted
    
    async def query(
        self,
        course_id: str,
        query_text: str,
        n_results: int = None,
    ) -> Dict[str, Any]:
        """
        Query documents in a course collection (legacy method).
        
        Args:
            course_id: Course identifier
            query_text: Query string
            n_results: Number of results to return
            
        Returns:
            Query results with documents, metadatas, and distances
        """
        if n_results is None:
            n_results = settings.TOP_K_RESULTS
        
        collection_name = self._get_collection_name(course_id)
        collection = await self._ensure_collection(collection_name)
        
        results = await asyncio.to_thread(
            collection.query,
            query_texts=[query_text],
            n_results=n_results,
            include=["documents", "metadatas", "distances"],
        )
        
        logger.debug(
            "Query executed",
            course_id=course_id,
            query_length=len(query_text),
            results_count=len(results.get("documents", [[]])[0]),
        )
        
        return results
    
    async def delete_documents(
        self,
        ids: Optional[List[str]] = None,
        collection_name: Optional[str] = None,
        where: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Delete documents from a collection.
        
        Args:
            ids: List of document IDs to delete
            collection_name: Collection name
            where: Optional metadata filter for deletion
        """
        target_collection = collection_name or "default"
        
        try:
            collection = await self._ensure_collection(target_collection)
            
            delete_kwargs = {}
            if ids:
                delete_kwargs["ids"] = ids
            if where:
                delete_kwargs["where"] = where
            
            if delete_kwargs:
                await asyncio.to_thread(
                    collection.delete,
                    **delete_kwargs
                )
                
            logger.info(
                "Documents deleted from collection",
                collection=target_collection,
                ids_count=len(ids) if ids else 0,
            )
        except Exception as e:
            logger.error(
                "Failed to delete documents",
                collection=target_collection,
                error=str(e)
            )
            raise
    
    async def delete_collection(self, collection_name: str) -> bool:
        """
        Delete a collection.
        
        Args:
            collection_name: Collection name
            
        Returns:
            True if deleted, False if not found
        """
        if not self._client:
            await self.initialize()
        
        try:
            await asyncio.to_thread(
                self._client.delete_collection,
                collection_name
            )
            
            # Remove from cache
            if collection_name in self._collections:
                del self._collections[collection_name]
            
            logger.info("Collection deleted", collection=collection_name)
            return True
        except Exception as e:
            logger.warning(
                "Collection not found for deletion",
                collection=collection_name,
                error=str(e)
            )
            return False
    
    async def list_collections(self) -> List[Dict[str, Any]]:
        """
        List all collections.
        
        Returns:
            List of collection info dicts
        """
        if not self._client:
            await self.initialize()
        
        collections = await asyncio.to_thread(
            self._client.list_collections
        )
        
        result = []
        for col in collections:
            count = await asyncio.to_thread(col.count)
            result.append({
                "name": col.name,
                "metadata": col.metadata,
                "count": count,
            })
        
        return result
    
    async def get_collection_stats(self, course_id: str) -> Dict[str, Any]:
        """
        Get statistics for a course collection.
        
        Args:
            course_id: Course identifier
            
        Returns:
            Collection statistics
        """
        collection = await self.get_or_create_collection(course_id)
        count = await asyncio.to_thread(collection.count)
        
        return {
            "course_id": course_id,
            "collection_name": self._get_collection_name(course_id),
            "document_count": count,
        }


# Singleton instance
_vector_store: Optional[VectorStoreService] = None


def get_vector_store() -> VectorStoreService:
    """Get or create the vector store singleton."""
    global _vector_store
    if _vector_store is None:
        _vector_store = VectorStoreService()
    return _vector_store
