"""
Unit Tests for Document Processor Service
==========================================

Tests for DocumentProcessor including:
- File type routing
- Text cleaning and normalization
- Text chunking logic with overlap
- Zip archive extraction and processing
- Image extraction (Multimodal/Vision)
"""

import pytest
import io
import zipfile
from unittest.mock import AsyncMock, Mock, patch, MagicMock
from app.services.document_processor import DocumentProcessor, ProcessedDocument, ProcessedChunk

# ==============================================================================
# FIXTURES
# ==============================================================================

@pytest.fixture
def doc_processor():
    """Create DocumentProcessor with mocked dependencies."""
    with patch('app.services.document_processor.get_vector_store'), \
         patch('app.services.document_processor.PaddleOCR'):
        
        processor = DocumentProcessor()
        processor.ocr_available = False # Simplify for unit tests
        processor.vision_available = False
        
        return processor


# ==============================================================================
# TESTS: Text Cleaning & Chunking
# ==============================================================================

def test_clean_text(doc_processor):
    """Test text normalization and cleaning."""
    dirty_text = "Hello   world!\n\n\nNew line\twith tab."
    clean = doc_processor._clean_text(dirty_text)
    
    assert "  " not in clean
    assert "\t" not in clean
    assert clean == "Hello world! New line with tab."


def test_create_chunks_basic(doc_processor):
    """Test splitting text into chunks."""
    doc_processor.chunk_size = 20
    doc_processor.chunk_overlap = 5
    text = "Ini adalah teks yang cukup panjang untuk dipecah."
    
    chunks = doc_processor._create_chunks(
        text=text,
        document_id="doc1",
        filename="test.txt",
        page_number=1
    )
    
    assert len(chunks) > 1
    assert chunks[0].text.startswith("Ini adalah")
    assert chunks[0].metadata["document_id"] == "doc1"


def test_create_chunks_with_overlap(doc_processor):
    """Test that chunks have overlap."""
    doc_processor.chunk_size = 20
    doc_processor.chunk_overlap = 10
    text = "012345678901234567890123456789"
    
    chunks = doc_processor._create_chunks(text, "d", "f", 1)
    
    # Check if some content from end of chunk 0 is at start of chunk 1
    overlap_area = chunks[0].text[-5:]
    assert overlap_area in chunks[1].text


# ==============================================================================
# TESTS: File Processing
# ==============================================================================

@pytest.mark.asyncio
async def test_process_text_file(doc_processor):
    """Test processing a plain text file."""
    content = b"Konten teks sederhana."
    
    # Mock store_chunks to avoid DB call
    doc_processor._store_chunks = AsyncMock()
    
    result = await doc_processor.process_file(
        file_content=content,
        filename="test.txt",
        document_id="id123",
        collection_name="col1"
    )
    
    assert result.success is True
    assert result.file_type == "text"
    assert len(result.chunks) > 0
    assert result.total_characters == len(content.decode())


@pytest.mark.asyncio
async def test_process_unsupported_file(doc_processor):
    """Test handling of unsupported file extensions."""
    result = await doc_processor.process_file(
        file_content=b"...",
        filename="test.exe",
        document_id="id",
        collection_name="col"
    )
    
    assert result.success is False
    assert "Unsupported" in result.error


# ==============================================================================
# TESTS: Zip Processing
# ==============================================================================

@pytest.mark.asyncio
async def test_process_zip_file(doc_processor):
    """Test extracting and processing a ZIP archive."""
    # Create a mock zip in memory
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w") as zf:
        zf.writestr("file1.txt", "Content 1")
        zf.writestr("file2.txt", "Content 2")
    
    zip_content = zip_buffer.getvalue()
    
    # Mock single file processing
    doc_processor.process_file = AsyncMock(return_value=ProcessedDocument(
        filename="f", file_type="text", chunks=[], page_count=1, 
        image_count=0, total_characters=10, processing_time_ms=1, success=True
    ))
    
    batch_result = await doc_processor.process_zip(
        zip_content=zip_content,
        base_document_id="zip_id",
        collection_name="col"
    )
    
    assert batch_result.total_files == 2
    assert batch_result.successful_files == 2


# ==============================================================================
# TESTS: PDF Processing (Mocked)
# ==============================================================================

@pytest.mark.asyncio
async def test_process_pdf_mocked(doc_processor):
    """Test PDF routing logic."""
    content = b"fake pdf"
    
    with patch.object(doc_processor, '_process_pdf', new_callable=AsyncMock) as mock_pdf:
        mock_pdf.return_value = ProcessedDocument(
            filename="test.pdf", file_type="pdf", chunks=[], page_count=5, 
            image_count=0, total_characters=100, processing_time_ms=10, success=True
        )
        doc_processor._store_chunks = AsyncMock()
        
        result = await doc_processor.process_file(content, "test.pdf", "id", "col")
        
        assert result.file_type == "pdf"
        assert result.page_count == 5
        mock_pdf.assert_called_once()
