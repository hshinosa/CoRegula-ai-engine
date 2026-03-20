import sys
from unittest.mock import MagicMock, AsyncMock, patch

# Mock module dependencies BEFORE importing the actual application code
sys.modules['chromadb'] = MagicMock()
sys.modules['chromadb.config'] = MagicMock()
sys.modules['chromadb.utils'] = MagicMock()
sys.modules['hnswlib'] = MagicMock()
sys.modules['pypdf'] = MagicMock()
sys.modules['fitz'] = MagicMock()
sys.modules['docx'] = MagicMock()
sys.modules['pptx'] = MagicMock()
sys.modules['openpyxl'] = MagicMock()
sys.modules['pandas'] = MagicMock()
sys.modules['numpy'] = MagicMock()
sys.modules['PIL'] = MagicMock()
sys.modules['motor'] = MagicMock()
sys.modules['motor.motor_asyncio'] = MagicMock()

import pytest
import asyncio
from app.services.vector_store import VectorStoreService, OpenAIEmbeddingFunction

@pytest.fixture
def mock_get_embedding_service():
    with patch('app.services.vector_store.get_embedding_service') as mock:
        yield mock

@pytest.mark.asyncio
async def test_vector_store_initialization():
    with patch('app.services.vector_store.chromadb.PersistentClient') as mock_client:
        service = VectorStoreService()
        await service.initialize()
        assert service._initialized is True
        mock_client.assert_called()

@pytest.mark.asyncio
async def test_ensure_collection():
    service = VectorStoreService()
    # Mock the client
    service._client = MagicMock()
    mock_collection = MagicMock()
    service._client.get_or_create_collection.return_value = mock_collection
    
    collection = await service._ensure_collection("test_col")
    assert collection == mock_collection
    assert "test_col" in service._collections

def test_openai_embedding_function(mock_get_embedding_service):
    func = OpenAIEmbeddingFunction()
    mock_service = MagicMock()
    # Mock embed_texts to be async
    mock_service.embed_texts = AsyncMock(return_value=[[0.1, 0.2]])
    mock_get_embedding_service.return_value = mock_service
    
    result = func(["test text"])
    assert result == [[0.1, 0.2]]

