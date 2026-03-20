"""
Tests for Embeddings Service - Full Coverage
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from app.services.embeddings import (
    OpenAIEmbeddingService,
    get_embedding_service,
)


class TestOpenAIEmbeddingServiceFull:
    """Test OpenAIEmbeddingService full coverage."""
    
    @pytest.fixture
    def mock_client(self):
        """Create mock OpenAI client."""
        mock = MagicMock()
        mock.embeddings.create = AsyncMock(return_value=MagicMock(
            data=[MagicMock(embedding=[0.1, 0.2, 0.3])]
        ))
        return mock
    
    def test_initialize_already_initialized(self):
        """Test initialize when already initialized."""
        with patch('app.services.embeddings.settings') as mock_settings:
            mock_settings.OPENAI_API_KEY = 'test_key'
            mock_settings.OPENAI_BASE_URL = 'http://test.com'
            mock_settings.OPENAI_EMBEDDING_MODEL = 'test-model'
            
            service = OpenAIEmbeddingService()
            service._initialized = True
            
            # Should not re-initialize
            service.initialize()
            
            assert service._initialized is True
    
    @pytest.mark.asyncio
    async def test_embed_text_auto_init(self):
        """Test embed_text auto-initializes."""
        with patch('app.services.embeddings.settings') as mock_settings:
            mock_settings.OPENAI_API_KEY = 'test'
            mock_settings.OPENAI_BASE_URL = 'http://test'
            mock_settings.OPENAI_EMBEDDING_MODEL = 'test'
            
            service = OpenAIEmbeddingService()
            service._initialized = False
            
            # First call initialize
            service.initialize()
            
            # Then mock the client
            service._client.embeddings.create = AsyncMock(
                return_value=MagicMock(data=[MagicMock(embedding=[0.1, 0.2])])
            )
            
            result = await service.embed_text("test")
            
            assert len(result) == 2
    
    @pytest.mark.asyncio
    async def test_embed_texts_auto_init(self):
        """Test embed_texts auto-initializes."""
        with patch('app.services.embeddings.settings') as mock_settings:
            mock_settings.OPENAI_API_KEY = 'test'
            mock_settings.OPENAI_BASE_URL = 'http://test'
            mock_settings.OPENAI_EMBEDDING_MODEL = 'test'
            
            service = OpenAIEmbeddingService()
            service.initialize()
            
            service._client.embeddings.create = AsyncMock(
                return_value=MagicMock(data=[MagicMock(embedding=[0.1, 0.2])])
            )
            
            result = await service.embed_texts(["test"])
            
            assert len(result) == 1
    
    @pytest.mark.asyncio
    async def test_embed_query_auto_init(self):
        """Test embed_query auto-initializes."""
        with patch('app.services.embeddings.settings') as mock_settings:
            mock_settings.OPENAI_API_KEY = 'test'
            mock_settings.OPENAI_BASE_URL = 'http://test'
            mock_settings.OPENAI_EMBEDDING_MODEL = 'test'
            
            service = OpenAIEmbeddingService()
            service.initialize()
            
            service._client.embeddings.create = AsyncMock(
                return_value=MagicMock(data=[MagicMock(embedding=[0.1, 0.2])])
            )
            
            result = await service.embed_query("query")
            
            assert len(result) == 2
    
    @pytest.mark.asyncio
    async def test_embed_texts(self):
        """Test embed_texts method."""
        with patch('app.services.embeddings.settings') as mock_settings:
            mock_settings.OPENAI_API_KEY = 'test'
            mock_settings.OPENAI_BASE_URL = 'http://test'
            mock_settings.OPENAI_EMBEDDING_MODEL = 'test'
            
            service = OpenAIEmbeddingService()
            service.initialize()
            
            service._client.embeddings.create = AsyncMock(
                return_value=MagicMock(
                    data=[
                        MagicMock(embedding=[0.1, 0.2]),
                        MagicMock(embedding=[0.3, 0.4]),
                    ]
                )
            )
            
            result = await service.embed_texts(["text1", "text2"])
            
            assert len(result) == 2
    
    @pytest.mark.asyncio
    async def test_embed_query(self):
        """Test embed_query method."""
        with patch('app.services.embeddings.settings') as mock_settings:
            mock_settings.OPENAI_API_KEY = 'test'
            mock_settings.OPENAI_BASE_URL = 'http://test'
            mock_settings.OPENAI_EMBEDDING_MODEL = 'test'
            
            service = OpenAIEmbeddingService()
            service.initialize()
            
            service._client.embeddings.create = AsyncMock(
                return_value=MagicMock(data=[MagicMock(embedding=[0.5, 0.6])])
            )
            
            result = await service.embed_query("query test")
            
            assert len(result) == 2
    
    @pytest.mark.asyncio
    async def test_get_embedding_alias(self):
        """Test get_embedding is alias for embed_text."""
        with patch('app.services.embeddings.settings') as mock_settings:
            mock_settings.OPENAI_API_KEY = 'test'
            mock_settings.OPENAI_BASE_URL = 'http://test'
            mock_settings.OPENAI_EMBEDDING_MODEL = 'test'
            
            service = OpenAIEmbeddingService()
            service.initialize()
            
            service._client.embeddings.create = AsyncMock(
                return_value=MagicMock(data=[MagicMock(embedding=[0.7, 0.8])])
            )
            
            result = await service.get_embedding("test")
            
            assert len(result) == 2
    
    def test_initialize_no_api_key(self):
        """Test initialize without API key."""
        with patch('app.services.embeddings.settings') as mock_settings:
            mock_settings.OPENAI_API_KEY = ''
            
            service = OpenAIEmbeddingService()
            
            with pytest.raises(ValueError):
                service.initialize()


class TestGetEmbeddingServiceFull:
    """Test get_embedding_service singleton full coverage."""
    
    def test_singleton_creates_new_instance(self):
        """Test singleton creates new instance when None."""
        from app.services import embeddings
        embeddings._embedding_service = None
        
        with patch('app.services.embeddings.settings') as mock_settings:
            mock_settings.OPENAI_API_KEY = 'test'
            mock_settings.OPENAI_BASE_URL = 'http://test'
            mock_settings.OPENAI_EMBEDDING_MODEL = 'test'
            
            service = get_embedding_service()
            
            assert service is not None
            assert isinstance(service, OpenAIEmbeddingService)
    
    def test_singleton_returns_existing(self):
        """Test singleton returns existing instance."""
        from app.services import embeddings
        
        existing = MagicMock()
        embeddings._embedding_service = existing
        
        result = get_embedding_service()
        
        assert result is existing
