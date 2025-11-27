"""
PDF Processing Service
======================
PDF ingestion, text extraction, chunking, and embedding pipeline.
"""

from typing import List, Dict, Any, Optional
from pathlib import Path
import uuid
import hashlib
from io import BytesIO

from pypdf import PdfReader

from app.core.config import settings
from app.core.logging import get_logger
from app.services.vector_store import get_vector_store

logger = get_logger(__name__)


class PDFProcessingService:
    """Service for processing PDF documents."""
    
    def __init__(self):
        """Initialize PDF processing service."""
        self.chunk_size = settings.CHUNK_SIZE
        self.chunk_overlap = settings.CHUNK_OVERLAP
        self.max_file_size_bytes = settings.MAX_FILE_SIZE_MB * 1024 * 1024
    
    def validate_file(self, file_content: bytes, filename: str) -> Dict[str, Any]:
        """
        Validate uploaded file.
        
        Args:
            file_content: Raw file bytes
            filename: Original filename
            
        Returns:
            Validation result dict
            
        Raises:
            ValueError: If validation fails
        """
        # Check file size
        if len(file_content) > self.max_file_size_bytes:
            raise ValueError(
                f"File size exceeds maximum allowed size of {settings.MAX_FILE_SIZE_MB}MB"
            )
        
        # Check file extension
        if not filename.lower().endswith(".pdf"):
            raise ValueError("Only PDF files are supported")
        
        # Try to parse PDF
        try:
            reader = PdfReader(BytesIO(file_content))
            page_count = len(reader.pages)
        except Exception as e:
            raise ValueError(f"Invalid PDF file: {str(e)}")
        
        return {
            "valid": True,
            "filename": filename,
            "size_bytes": len(file_content),
            "page_count": page_count,
        }
    
    def extract_text_from_pdf(self, file_content: bytes) -> List[Dict[str, Any]]:
        """
        Extract text from PDF file by page.
        
        Args:
            file_content: Raw PDF bytes
            
        Returns:
            List of page text with metadata
        """
        reader = PdfReader(BytesIO(file_content))
        pages = []
        
        for page_num, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            text = text.strip()
            
            if text:  # Only include pages with content
                pages.append({
                    "page_number": page_num,
                    "text": text,
                    "char_count": len(text),
                })
        
        logger.debug(
            "PDF text extracted",
            total_pages=len(reader.pages),
            pages_with_text=len(pages),
        )
        
        return pages
    
    def chunk_text(
        self,
        text: str,
        metadata: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """
        Split text into overlapping chunks.
        
        Args:
            text: Text to chunk
            metadata: Base metadata to include with each chunk
            
        Returns:
            List of chunks with metadata
        """
        chunks = []
        text_length = len(text)
        
        if text_length <= self.chunk_size:
            # Text fits in single chunk
            chunks.append({
                "text": text,
                "metadata": {
                    **metadata,
                    "chunk_index": 0,
                    "chunk_count": 1,
                    "char_start": 0,
                    "char_end": text_length,
                },
            })
            return chunks
        
        # Split into overlapping chunks
        start = 0
        chunk_index = 0
        
        while start < text_length:
            end = min(start + self.chunk_size, text_length)
            
            # Try to break at sentence or word boundary
            if end < text_length:
                # Look for sentence boundary
                for sep in [". ", ".\n", "? ", "! ", "\n\n"]:
                    last_sep = text.rfind(sep, start, end)
                    if last_sep > start + self.chunk_size // 2:
                        end = last_sep + len(sep)
                        break
                else:
                    # Look for word boundary
                    last_space = text.rfind(" ", start, end)
                    if last_space > start + self.chunk_size // 2:
                        end = last_space + 1
            
            chunk_text = text[start:end].strip()
            
            if chunk_text:
                chunks.append({
                    "text": chunk_text,
                    "metadata": {
                        **metadata,
                        "chunk_index": chunk_index,
                        "char_start": start,
                        "char_end": end,
                    },
                })
                chunk_index += 1
            
            # Move start position with overlap
            start = end - self.chunk_overlap
            if start >= text_length:
                break
        
        # Update chunk count in all metadata
        for chunk in chunks:
            chunk["metadata"]["chunk_count"] = len(chunks)
        
        return chunks
    
    def generate_document_id(
        self,
        course_id: str,
        filename: str,
        page_number: int,
        chunk_index: int,
    ) -> str:
        """Generate unique document ID."""
        content = f"{course_id}:{filename}:{page_number}:{chunk_index}"
        hash_part = hashlib.md5(content.encode()).hexdigest()[:8]
        return f"doc_{hash_part}_{chunk_index}"
    
    async def process_pdf(
        self,
        pdf_content: bytes,
        filename: str,
        document_id: str,
        collection_name: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Process PDF and ingest into vector store.
        
        Args:
            pdf_content: Raw PDF bytes
            filename: Original filename
            document_id: Unique document identifier
            collection_name: Target collection name
            metadata: Additional metadata to include
            
        Returns:
            Processing result with statistics
        """
        # Validate file
        validation = self.validate_file(pdf_content, filename)
        
        # Extract text from PDF
        pages = self.extract_text_from_pdf(pdf_content)
        
        if not pages:
            raise ValueError("No text content found in PDF")
        
        # Process all pages into chunks
        all_chunks = []
        base_metadata = metadata or {}
        
        for page in pages:
            page_metadata = {
                **base_metadata,
                "document_id": document_id,
                "source": filename,
                "filename": filename,
                "page": page["page_number"],
            }
            chunks = self.chunk_text(page["text"], page_metadata)
            all_chunks.extend(chunks)
        
        # Prepare data for vector store
        documents = []
        metadatas = []
        ids = []
        
        for i, chunk in enumerate(all_chunks):
            chunk_id = f"{document_id}_{i}"
            documents.append(chunk["text"])
            metadatas.append(chunk["metadata"])
            ids.append(chunk_id)
        
        # Add to vector store
        vector_store = get_vector_store()
        await vector_store.add_documents(
            documents=documents,
            metadatas=metadatas,
            ids=ids,
            collection_name=collection_name,
        )
        
        result = {
            "success": True,
            "document_id": document_id,
            "filename": filename,
            "page_count": validation["page_count"],
            "pages_processed": len(pages),
            "chunks_count": len(all_chunks),
            "file_size_bytes": validation["size_bytes"],
        }
        
        logger.info("PDF processed and ingested", **result)
        
        return result

    async def process_and_ingest(
        self,
        course_id: str,
        file_content: bytes,
        filename: str,
    ) -> Dict[str, Any]:
        """
        Process PDF and ingest into vector store (legacy method).
        
        Args:
            course_id: Course identifier
            file_content: Raw PDF bytes
            filename: Original filename
            
        Returns:
            Ingestion result with statistics
        """
        document_id = str(uuid.uuid4())
        collection_name = f"course_{course_id}"
        
        return await self.process_pdf(
            pdf_content=file_content,
            filename=filename,
            document_id=document_id,
            collection_name=collection_name,
            metadata={"course_id": course_id}
        )


# Singleton instance
_pdf_processor: Optional[PDFProcessingService] = None


def get_pdf_processor() -> PDFProcessingService:
    """Get or create the PDF processor singleton."""
    global _pdf_processor
    if _pdf_processor is None:
        _pdf_processor = PDFProcessingService()
    return _pdf_processor
