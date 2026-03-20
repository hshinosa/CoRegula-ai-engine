import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from app.services.embeddings import OpenAIEmbeddingService, get_embedding_service

@pytest.fixture
def mock_openai():
    with patch('app.services.embeddings.AsyncOpenAI') as mock:
        client = MagicMock()
        mock.return_value = client
        yield client

@pytest.fixture
def service(mock_openai):
    s = OpenAIEmbeddingService()
    s.initialize()
    return s

@pytest.mark.unit
@pytest.mark.asyncio
async def test_embed_text_full(service, mock_openai):
    mock_resp = MagicMock()
    mock_resp.data = [MagicMock(embedding=[0.1, 0.2])]
    mock_openai.embeddings.create = AsyncMock(return_value=mock_resp)
    
    res = await service.embed_text('test')
    assert res == [0.1, 0.2]

@pytest.mark.unit
@pytest.mark.asyncio
async def test_embed_texts_full(service, mock_openai):
    mock_resp = MagicMock()
    mock_resp.data = [MagicMock(embedding=[0.1]), MagicMock(embedding=[0.2])]
    mock_openai.embeddings.create = AsyncMock(return_value=mock_resp)
    
    res = await service.embed_texts(['t1', 't2'])
    assert res == [[0.1], [0.2]]

@pytest.mark.unit
@pytest.mark.asyncio
async def test_initialize_twice(mock_openai):
    s = OpenAIEmbeddingService()
    s.initialize()
    first_client = s._client
    s.initialize() # Should return early
    assert s._client is first_client

@pytest.mark.unit
def test_singleton():
    s1 = get_embedding_service()
    s2 = get_embedding_service()
    assert s1 is s2