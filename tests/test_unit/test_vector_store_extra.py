import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from app.services.vector_store import VectorStoreService, OpenAIEmbeddingFunction, get_vector_store

@pytest.fixture
def mock_chroma():
    with patch('chromadb.PersistentClient') as mock_client:
        client_inst = MagicMock()
        mock_client.return_value = client_inst
        yield client_inst

@pytest.fixture
def vector_store(mock_chroma):
    with patch('app.services.vector_store.get_embedding_service'):
        service = VectorStoreService()
        with patch('os.makedirs'):
            service._client = mock_chroma
            service._initialized = True
        return service

@pytest.mark.unit
@pytest.mark.asyncio
async def test_query_legacy_full(vector_store, mock_chroma):
    mock_coll = MagicMock()
    mock_chroma.get_or_create_collection.return_value = mock_coll
    mock_coll.query.return_value = {'ids': [['id1']]}
    
    res = await vector_store.query('c1', 'q1')
    assert res['ids'] == [['id1']]

@pytest.mark.unit
@pytest.mark.asyncio
async def test_delete_documents_with_where(vector_store, mock_chroma):
    mock_coll = MagicMock()
    mock_chroma.get_or_create_collection.return_value = mock_coll
    
    await vector_store.delete_documents(ids=['id1'], where={'s': 1}, collection_name='test')
    mock_coll.delete.assert_called_with(ids=['id1'], where={'s': 1})

@pytest.mark.unit
@pytest.mark.asyncio
async def test_delete_collection_not_found(vector_store, mock_chroma):
    mock_chroma.delete_collection.side_effect = Exception('Not found')
    res = await vector_store.delete_collection('nonexistent')
    assert res is False

@pytest.mark.unit
@pytest.mark.asyncio
async def test_initialize_twice(mock_chroma):
    service = VectorStoreService()
    with patch('os.makedirs'):
        await service.initialize()
        assert service._initialized is True
        await service.initialize() # Should return early