"""
Services Package - MVP Phase 1-1.5
CoRegula AI Engine

All AI services for document processing, RAG, and chat intervention.
"""

from app.services.embeddings import GeminiEmbeddingService, get_embedding_service
from app.services.vector_store import VectorStoreService, get_vector_store
from app.services.pdf_processor import PDFProcessingService, get_pdf_processor
from app.services.document_processor import (
    DocumentProcessor,
    get_document_processor,
    ProcessedDocument,
    ProcessedChunk,
    BatchProcessResult,
)
from app.services.llm import GeminiLLMService, get_llm_service, ChatMessage, LLMResponse
from app.services.rag import RAGPipeline, get_rag_pipeline, RAGResult
from app.services.intervention import (
    ChatInterventionService,
    get_intervention_service,
    InterventionType,
    InterventionResult,
)

__all__ = [
    # Embedding
    "GeminiEmbeddingService",
    "get_embedding_service",
    # Vector Store
    "VectorStoreService",
    "get_vector_store",
    # PDF Processor (legacy)
    "PDFProcessingService",
    "get_pdf_processor",
    # Document Processor (comprehensive)
    "DocumentProcessor",
    "get_document_processor",
    "ProcessedDocument",
    "ProcessedChunk",
    "BatchProcessResult",
    # LLM
    "GeminiLLMService",
    "get_llm_service",
    "ChatMessage",
    "LLMResponse",
    # RAG
    "RAGPipeline",
    "get_rag_pipeline",
    "RAGResult",
    # Intervention
    "ChatInterventionService",
    "get_intervention_service",
    "InterventionType",
    "InterventionResult",
]