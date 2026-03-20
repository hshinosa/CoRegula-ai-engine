import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from app.services.document_processor import DocumentProcessor, ProcessedChunk

@pytest.fixture
def processor():
    with patch('app.services.document_processor.get_vector_store'):
        return DocumentProcessor()

@pytest.mark.unit
@pytest.mark.asyncio
async def test_process_docx_full(processor):
    mock_doc = MagicMock()
    mock_doc.paragraphs = [MagicMock(text='Para 1')]
    mock_doc.tables = []
    
    with patch('app.services.document_processor.DocxDocument', return_value=mock_doc),          patch.object(processor, '_extract_images_from_docx', return_value=iter([])),          patch.object(processor, '_store_chunks', new_callable=AsyncMock):
        res = await processor.process_file(b'fake', 'test.docx', 'id1', metadata={})
        assert res.success is True
        assert res.file_type == 'docx'

@pytest.mark.unit
@pytest.mark.asyncio
async def test_process_pptx_full(processor):
    mock_pres = MagicMock()
    mock_pres.slides = [MagicMock(shapes=[])]
    
    with patch('app.services.document_processor.Presentation', return_value=mock_pres),          patch.object(processor, '_store_chunks', new_callable=AsyncMock):
        res = await processor.process_file(b'fake', 'test.pptx', 'id2', metadata={})
        assert res.success is True
        assert res.file_type == 'pptx'

@pytest.mark.unit
@pytest.mark.asyncio
async def test_process_pdf_ocr_trigger(processor):
    mock_pdf = MagicMock()
    mock_page = MagicMock()
    mock_page.get_text.return_value = 'low text'
    mock_page.get_images.return_value = []
    mock_pdf.__iter__.return_value = [mock_page]
    
    with patch('app.services.document_processor.fitz.open', return_value=mock_pdf),          patch.object(processor, '_run_page_ocr', new_callable=AsyncMock, return_value='OCR text'),          patch.object(processor, '_store_chunks', new_callable=AsyncMock):
        res = await processor._process_pdf(b'fake', 'test.pdf', 'id3', metadata={})
        assert res.success is True