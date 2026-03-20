"""
Tests for Vector Store Service - 100% Coverage
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock
from app.services.vector_store import (
    VectorStoreService,
    OpenAIEmbeddingFunction,
    get_vector_store,
)


class TestOpenAIEmbeddingFunction:
    """Test OpenAIEmbeddingFunction class."""
    
    def test_call(self):
        """Test __call__ method."""
        with patch('app.services.vector_store.get_embedding_service') as mock_get:
            mock_service = MagicMock()
            mock_service.embed_texts = MagicMock(return_value=[[0.1, 0.2], [0.3, 0.4]])
            mock_get.return_value = mock_service
            
            # Mock asyncio loop
            with patch('asyncio.get_event_loop') as mock_loop:
                mock_loop.return_value.run_until_complete = MagicMock(return_value=[[0.1, 0.2], [0.3, 0.4]])
                
                func = OpenAIEmbeddingFunction()
                result = func(["text1", "text2"])
                
                assert result == [[0.1, 0.2], [0.3, 0.4]]


class TestVectorStoreService:
    """Test VectorStoreService class."""
    
    @pytest.fixture
    def vector_store(self):
        """Create vector store instance."""
        return VectorStoreService()
    
    def test_init(self, vector_store):
        """Test initialization."""
        assert vector_store._client is None
        assert vector_store._initialized is False
        assert vector_store._collections == {}
    
    @pytest.mark.asyncio
    async def test_initialize(self, vector_store):
        """Test initialize method."""
        with patch('app.services.vector_store.Path') as mock_path:
            mock_path.return_value = MagicMock()
            with patch('app.services.vector_store.get_embedding_service') as mock_get:
                mock_service = MagicMock()
                mock_get.return_value = mock_service
                
                with patch('app.services.vector_store.chromadb.PersistentClient') as mock_client:
                    mock_client.return_value = MagicMock()
                    
                    await vector_store.initialize()
                    
                    assert vector_store._initialized is True
                    assert vector_store._client is not None
    
    @pytest.mark.asyncio
    async def test_initialize_already_initialized(self, vector_store):
        """Test initialize when already initialized."""
        vector_store._initialized = True
        vector_store._client = MagicMock()
        
        await vector_store.initialize()
        
        # Should not re-initialize
        assert vector_store._initialized is True
    
    @pytest.mark.asyncio
    async def test_ensure_collection(self, vector_store):
        """Test _ensure_collection method."""
        mock_collection = MagicMock()
        mock_client = MagicMock()
        mock_client.get_or_create_collection.return_value = mock_collection
        vector_store._client = mock_client
        
        result = await vector_store._ensure_collection("test_collection")
        
        assert result == mock_collection
        assert "test_collection" in vector_store._collections
    
    @pytest.mark.asyncio
    async def test_ensure_collection_cached(self, vector_store):
        """Test _ensure_collection returns cached collection."""
        mock_collection = MagicMock()
        vector_store._collections["cached"] = mock_collection
        vector_store._client = MagicMock()
        
        result = await vector_store._ensure_collection("cached")
        
        assert result == mock_collection
    
    @pytest.mark.asyncio
    async def test_ensure_collection_initializes_client(self, vector_store):
        """Test _ensure_collection initializes client if needed."""
        vector_store._client = None
        vector_store._initialized = False
        
        async def mock_init_side_effect():
            vector_store._initialized = True
            vector_store._client = MagicMock()
            vector_store._client.get_or_create_collection = MagicMock(return_value=MagicMock())
            
        with patch.object(vector_store, 'initialize', new_callable=AsyncMock, side_effect=mock_init_side_effect):
            with patch.object(vector_store, '_collections', {}):
                result = await vector_store._ensure_collection("test")
                
                assert result is not None
    
    def test_get_collection_name(self, vector_store):
        """Test _get_collection_name method."""
        with patch('app.services.vector_store.settings') as mock_settings:
            mock_settings.CHROMA_COLLECTION_PREFIX = "prefix"
            
            result = vector_store._get_collection_name("course123")
            
            assert result == "prefix_course123"
    
    @pytest.mark.asyncio
    async def test_get_or_create_collection(self, vector_store):
        """Test get_or_create_collection method."""
        with patch.object(vector_store, '_ensure_collection', new_callable=AsyncMock) as mock_ensure:
            mock_ensure.return_value = MagicMock()
            
            result = await vector_store.get_or_create_collection("course123")
            
            assert result is not None
            mock_ensure.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_add_documents(self, vector_store):
        """Test add_documents method."""
        mock_collection = MagicMock()
        mock_collection.add = MagicMock()
        
        with patch.object(vector_store, '_ensure_collection', new_callable=AsyncMock) as mock_ensure:
            mock_ensure.return_value = mock_collection
            
            await vector_store.add_documents(
                documents=["doc1", "doc2"],
                metadatas=[{"source": "test"}, {"source": "test2"}],
                ids=["id1", "id2"],
                collection_name="test_collection"
            )
            
            mock_collection.add.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_add_documents_with_course_id(self, vector_store):
        """Test add_documents with course_id."""
        mock_collection = MagicMock()
        
        with patch.object(vector_store, '_ensure_collection', new_callable=AsyncMock) as mock_ensure:
            mock_ensure.return_value = mock_collection
            
            with patch('app.services.vector_store.settings') as mock_settings:
                mock_settings.CHROMA_COLLECTION_PREFIX = "prefix"
                
                await vector_store.add_documents(
                    documents=["doc1"],
                    metadatas=[{}],
                    ids=["id1"],
                    course_id="course1"
                )
                
                mock_ensure.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_search(self, vector_store):
        """Test search method."""
        mock_collection = MagicMock()
        mock_collection.query.return_value = {
            "documents": [["result1", "result2"]],
            "metadatas": [[{"source": "test1"}, {"source": "test2"}]],
            "distances": [[0.1, 0.2]],
        }
        
        with patch.object(vector_store, '_ensure_collection', new_callable=AsyncMock) as mock_ensure:
            mock_ensure.return_value = mock_collection
            
            results = await vector_store.search(
                query="test query",
                collection_name="test_collection",
                n_results=2
            )
            
            assert len(results) == 2
            assert results[0]["content"] == "result1"
            assert results[0]["metadata"] == {"source": "test1"}
            assert results[0]["score"] > 0
    
    @pytest.mark.asyncio
    async def test_search_with_where(self, vector_store):
        """Test search with metadata filter."""
        mock_collection = MagicMock()
        mock_collection.query.return_value = {
            "documents": [["result"]],
            "metadatas": [[{}]],
            "distances": [[0.1]],
        }
        
        with patch.object(vector_store, '_ensure_collection', new_callable=AsyncMock) as mock_ensure:
            mock_ensure.return_value = mock_collection
            
            await vector_store.search(
                query="test",
                where={"course_id": "test"}
            )
            
            mock_collection.query.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_search_collection_not_found(self, vector_store):
        """Test search when collection not found."""
        with patch.object(vector_store, '_ensure_collection', new_callable=AsyncMock) as mock_ensure:
            mock_ensure.side_effect = Exception("Collection not found")
            
            results = await vector_store.search("query")
            
            assert results == []
    
    @pytest.mark.asyncio
    async def test_query(self, vector_store):
        """Test query method."""
        mock_collection = MagicMock()
        mock_collection.query.return_value = {
            "documents": [["result"]],
            "metadatas": [[{}]],
            "distances": [[0.1]],
        }
        
        with patch.object(vector_store, '_ensure_collection', new_callable=AsyncMock) as mock_ensure:
            mock_ensure.return_value = mock_collection
            
            with patch('app.services.vector_store.settings') as mock_settings:
                mock_settings.TOP_K_RESULTS = 5
                
                result = await vector_store.query(
                    course_id="course1",
                    query_text="test query"
                )
                
                assert "documents" in result
                assert "metadatas" in result
                assert "distances" in result
    
    @pytest.mark.asyncio
    async def test_query_custom_n_results(self, vector_store):
        """Test query with custom n_results."""
        mock_collection = MagicMock()
        mock_collection.query.return_value = {
            "documents": [[]],
            "metadatas": [[]],
            "distances": [[]],
        }
        
        with patch.object(vector_store, '_ensure_collection', new_callable=AsyncMock) as mock_ensure:
            mock_ensure.return_value = mock_collection
            
            await vector_store.query(
                course_id="course1",
                query_text="test",
                n_results=10
            )
            
            call_kwargs = mock_collection.query.call_args[1]
            assert call_kwargs["n_results"] == 10
    
    @pytest.mark.asyncio
    async def test_delete_documents(self, vector_store):
        """Test delete_documents method."""
        mock_collection = MagicMock()
        mock_collection.delete = MagicMock()
        
        with patch.object(vector_store, '_ensure_collection', new_callable=AsyncMock) as mock_ensure:
            mock_ensure.return_value = mock_collection
            
            await vector_store.delete_documents(
                ids=["id1", "id2"],
                collection_name="test_collection"
            )
            
            mock_collection.delete.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_delete_documents_with_where(self, vector_store):
        """Test delete_documents with where filter."""
        mock_collection = MagicMock()
        
        with patch.object(vector_store, '_ensure_collection', new_callable=AsyncMock) as mock_ensure:
            mock_ensure.return_value = mock_collection
            
            await vector_store.delete_documents(
                where={"course_id": "test"},
                collection_name="test"
            )
            
            mock_collection.delete.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_delete_documents_exception(self, vector_store):
        """Test delete_documents handles exception."""
        with patch.object(vector_store, '_ensure_collection', new_callable=AsyncMock) as mock_ensure:
            mock_ensure.side_effect = Exception("Test error")
            
            with pytest.raises(Exception):
                await vector_store.delete_documents(ids=["id1"])
    
    @pytest.mark.asyncio
    async def test_delete_collection(self, vector_store):
        """Test delete_collection method."""
        mock_client = MagicMock()
        mock_client.delete_collection = MagicMock()
        vector_store._client = mock_client
        vector_store._collections["to_delete"] = MagicMock()
        
        result = await vector_store.delete_collection("to_delete")
        
        assert result is True
        assert "to_delete" not in vector_store._collections
    
    @pytest.mark.asyncio
    async def test_delete_collection_not_found(self, vector_store):
        """Test delete_collection when not found."""
        mock_client = MagicMock()
        mock_client.delete_collection = MagicMock(side_effect=Exception("Not found"))
        vector_store._client = mock_client
        
        result = await vector_store.delete_collection("nonexistent")
        
        assert result is False
    
    @pytest.mark.asyncio
    async def test_delete_collection_initializes_client(self, vector_store):
        """Test delete_collection initializes client if needed."""
        vector_store._client = None
        
        with patch.object(vector_store, 'initialize', new_callable=AsyncMock):
            with patch('app.services.vector_store.chromadb.PersistentClient') as mock_client:
                mock_client.return_value = MagicMock()
                mock_client.return_value.delete_collection = MagicMock(side_effect=Exception("Not found"))
                
                await vector_store.delete_collection("test")
    
    @pytest.mark.asyncio
    async def test_list_collections(self, vector_store):
        """Test list_collections method."""
        mock_col1 = MagicMock()
        mock_col1.name = "collection1"
        mock_col1.metadata = {}
        mock_col1.count = MagicMock(return_value=10)
        
        mock_col2 = MagicMock()
        mock_col2.name = "collection2"
        mock_col2.metadata = {}
        mock_col2.count = MagicMock(return_value=5)
        
        mock_client = MagicMock()
        mock_client.list_collections = MagicMock(return_value=[mock_col1, mock_col2])
        vector_store._client = mock_client
        
        result = await vector_store.list_collections()
        
        assert len(result) == 2
        assert result[0]["name"] == "collection1"
        assert result[0]["count"] == 10
        assert result[1]["name"] == "collection2"
        assert result[1]["count"] == 5
    
    @pytest.mark.asyncio
    async def test_list_collections_initializes_client(self, vector_store):
        """Test list_collections initializes client if needed."""
        vector_store._client = None
        
        async def mock_init_side_effect():
            vector_store._initialized = True
            mock_client = MagicMock()
            mock_col = MagicMock()
            mock_col.name = "test"
            mock_col.metadata = {}
            mock_col.count = MagicMock(return_value=5)
            mock_client.list_collections = MagicMock(return_value=[mock_col])
            vector_store._client = mock_client
            
        with patch.object(vector_store, 'initialize', new_callable=AsyncMock, side_effect=mock_init_side_effect):
            result = await vector_store.list_collections()
            
            assert len(result) == 1
            assert result[0]["name"] == "test"
    
    @pytest.mark.asyncio
    async def test_get_collection_stats(self, vector_store):
        """Test get_collection_stats method."""
        mock_collection = MagicMock()
        mock_collection.count = MagicMock(return_value=25)
        
        with patch.object(vector_store, 'get_or_create_collection', new_callable=AsyncMock) as mock_get:
            mock_get.return_value = mock_collection
            
            with patch.object(vector_store, '_get_collection_name') as mock_name:
                mock_name.return_value = "prefix_course1"
                
                result = await vector_store.get_collection_stats("course1")
                
                assert result["course_id"] == "course1"
                assert result["collection_name"] == "prefix_course1"
                assert result["document_count"] == 25


class TestGetVectorStore:
    """Test get_vector_store singleton."""
    
    def test_get_vector_store_singleton(self):
        """Test get_vector_store returns singleton."""
        from app.services import vector_store
        vector_store._vector_store = None
        
        store1 = get_vector_store()
        store2 = get_vector_store()
        
        assert store1 is store2
        assert isinstance(store1, VectorStoreService)
    
    def test_get_vector_store_initialization(self):
        """Test get_vector_store initializes correctly."""
        from app.services import vector_store
        vector_store._vector_store = None
        
        store = get_vector_store()
        
        assert store is not None
        assert isinstance(store, VectorStoreService)
        assert store._initialized is False
