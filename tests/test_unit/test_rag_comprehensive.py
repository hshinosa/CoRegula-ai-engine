"""
Tests for RAG Pipeline Service - 100% Coverage
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from app.services.rag import (
    RAGPipeline,
    RAGResult,
    get_rag_pipeline,
)


class TestRAGResult:
    """Test RAGResult dataclass."""
    
    def test_rag_result_success(self):
        """Test RAGResult with success."""
        result = RAGResult(
            answer="Test answer",
            sources=[{"source": "doc.pdf", "page": 1}],
            query="Test query",
            tokens_used=100,
            success=True,
            processing_time_ms=50.0
        )
        
        assert result.answer == "Test answer"
        assert len(result.sources) == 1
        assert result.query == "Test query"
        assert result.tokens_used == 100
        assert result.success is True
        assert result.processing_time_ms == 50.0
    
    def test_rag_result_with_error(self):
        """Test RAGResult with error."""
        result = RAGResult(
            answer="",
            sources=[],
            query="Test query",
            tokens_used=0,
            success=False,
            error="Test error"
        )
        
        assert result.success is False
        assert result.error == "Test error"
    
    def test_rag_result_scaffolding(self):
        """Test RAGResult with scaffolding."""
        result = RAGResult(
            answer="Test answer",
            sources=[],
            query="Test query",
            tokens_used=50,
            success=True,
            scaffolding_triggered=True
        )
        
        assert result.scaffolding_triggered is True


class TestRAGPipeline:
    """Test RAGPipeline class."""
    
    @pytest.fixture
    def mock_vector_store(self):
        """Create mock vector store."""
        mock = MagicMock()
        mock.search = AsyncMock(return_value=[
            {
                "content": "Test content",
                "metadata": {"source": "test.pdf", "page": 1},
                "score": 0.85
            }
        ])
        return mock
    
    @pytest.fixture
    def mock_llm(self):
        """Create mock LLM service."""
        mock = MagicMock()
        mock.generate = AsyncMock(return_value=MagicMock(
            content="Test response",
            tokens_used=50,
            success=True
        ))
        mock.generate_rag_response = AsyncMock(return_value=MagicMock(
            content="RAG response",
            tokens_used=100,
            success=True
        ))
        mock.reframe_to_socratic = AsyncMock(return_value="Reframed response")
        return mock
    
    @pytest.fixture
    def mock_guardrails(self):
        """Create mock guardrails."""
        mock = MagicMock()
        mock.check_input = MagicMock(return_value=MagicMock(
            action=MagicMock(value="allow"),
            message=None,
            sanitized_input=None
        ))
        mock.check_output = MagicMock(return_value=MagicMock(
            action=MagicMock(value="allow"),
            message=None,
            sanitized_input=None
        ))
        return mock
    
    @pytest.fixture
    def mock_efficiency_guard(self):
        """Create mock efficiency guard."""
        mock = MagicMock()
        mock.execute_with_caching = AsyncMock(return_value={
            "answer": "Test answer",
            "sources": [],
            "query": "Test query",
            "tokens_used": 50,
            "success": True,
            "processing_time_ms": 10.0
        })
        return mock
    
    @pytest.fixture
    def rag_pipeline(self, mock_vector_store, mock_llm, mock_guardrails):
        """Create RAG pipeline with mocks."""
        with patch('app.services.rag.get_vector_store', return_value=mock_vector_store):
            with patch('app.services.rag.get_llm_service', return_value=mock_llm):
                with patch('app.services.rag.get_guardrails', return_value=mock_guardrails):
                    pipeline = RAGPipeline(
                        vector_store=mock_vector_store,
                        llm_service=mock_llm,
                        efficiency_guard=None
                    )
                    return pipeline
    
    def test_init(self, mock_vector_store, mock_llm):
        """Test initialization."""
        with patch('app.services.rag.get_guardrails') as mock_guards:
            mock_guards.return_value = MagicMock()
            
            pipeline = RAGPipeline(
                vector_store=mock_vector_store,
                llm_service=mock_llm
            )
            
            assert pipeline.vector_store == mock_vector_store
            assert pipeline.llm_service == mock_llm
    
    def test_should_retrieve_skip_pattern(self, rag_pipeline):
        """Test _should_retrieve with skip pattern."""
        result = rag_pipeline._should_retrieve("halo")
        assert result is False
    
    def test_should_retrieve_short_query(self, rag_pipeline):
        """Test _should_retrieve with short query."""
        result = rag_pipeline._should_retrieve("test")
        assert result is False
    
    def test_should_retrieve_greeting(self, rag_pipeline):
        """Test _should_retrieve with greeting."""
        result = rag_pipeline._should_retrieve("halo apa kabar")
        assert result is False
    
    def test_should_retrieve_substantive(self, rag_pipeline):
        """Test _should_retrieve with substantive query."""
        result = rag_pipeline._should_retrieve("Jelaskan tentang machine learning dan deep learning")
        assert result is True
    
    def test_format_search_results(self, rag_pipeline):
        """Test _format_search_results."""
        results = [
            {"content": "Doc 1", "metadata": {"source": "test.pdf"}, "score": 0.9},
            {"content": "Doc 2", "metadata": {}, "score": 0.7},
        ]
        
        contexts = rag_pipeline._format_search_results(results)
        
        assert len(contexts) == 2
        assert contexts[0]["content"] == "Doc 1"
        assert contexts[0]["score"] == 0.9
    
    def test_format_search_results_empty(self, rag_pipeline):
        """Test _format_search_results with empty results."""
        contexts = rag_pipeline._format_search_results([])
        assert contexts == []
    
    def test_extract_sources(self, rag_pipeline):
        """Test _extract_sources."""
        results = [
            {
                "content": "Test",
                "metadata": {"source": "doc1.pdf", "page": 1, "chunk_index": 0},
                "score": 0.9
            },
            {
                "content": "Test 2",
                "metadata": {"source": "doc2.pdf", "page": 2},
                "score": 0.8
            },
        ]
        
        sources = rag_pipeline._extract_sources(results)
        
        assert len(sources) == 2
        assert sources[0]["source"] == "doc1.pdf"
        assert sources[0]["page"] == 1
    
    def test_extract_sources_deduplicates(self, rag_pipeline):
        """Test _extract_sources deduplicates."""
        results = [
            {"content": "Test", "metadata": {"source": "same.pdf"}, "score": 0.9},
            {"content": "Test 2", "metadata": {"source": "same.pdf"}, "score": 0.8},
        ]
        
        sources = rag_pipeline._extract_sources(results)
        
        assert len(sources) == 1
    
    def test_extract_sources_unknown(self, rag_pipeline):
        """Test _extract_sources with unknown source."""
        results = [
            {"content": "Test", "metadata": {}, "score": 0.9},
        ]
        
        sources = rag_pipeline._extract_sources(results)
        
        assert sources[0]["source"] == "Unknown"
    
    @pytest.mark.asyncio
    async def test_query_no_fetch_greeting(self, rag_pipeline, mock_guardrails):
        """Test query with greeting (NO_FETCH)."""
        mock_guardrails.check_input.return_value = MagicMock(
            action=MagicMock(value="allow"),
            sanitized_input="halo"
        )
        
        result = await rag_pipeline.query("halo")
        
        assert result.success is True
        assert result.answer == "Test response"
    
    @pytest.mark.asyncio
    async def test_query_blocked(self, rag_pipeline, mock_guardrails):
        """Test query blocked by guardrails."""
        mock_guardrails.check_input.return_value = MagicMock(
            action=MagicMock(value="block"),
            message="Blocked message",
            sanitized_input=None
        )
        
        result = await rag_pipeline.query("Kerjakan tugas saya")
        
        assert result.success is True  # Handled successfully
        assert "Blocked" in result.answer
    
    @pytest.mark.asyncio
    async def test_query_fetch(self, rag_pipeline, mock_guardrails):
        """Test query with FETCH policy."""
        mock_guardrails.check_input.return_value = MagicMock(
            action=MagicMock(value="allow"),
            sanitized_input="Jelaskan machine learning"
        )
        
        result = await rag_pipeline.query("Jelaskan machine learning dan deep learning")
        
        assert result.success is True
        assert len(result.sources) > 0
    
    @pytest.mark.asyncio
    async def test_query_no_results(self, rag_pipeline, mock_guardrails, mock_vector_store):
        """Test query with no search results."""
        mock_guardrails.check_input.return_value = MagicMock(
            action=MagicMock(value="allow"),
            sanitized_input="Jelaskan quantum physics"
        )
        mock_vector_store.search = AsyncMock(return_value=[])
        
        result = await rag_pipeline.query("Jelaskan quantum physics yang tidak ada di dokumen")
        
        assert result.success is True
        assert result.answer == "Test response"
    
    @pytest.mark.asyncio
    async def test_query_output_blocked(self, rag_pipeline, mock_guardrails):
        """Test query with output blocked by guardrails."""
        mock_guardrails.check_input.return_value = MagicMock(
            action=MagicMock(value="allow"),
            sanitized_input="test query"
        )
        mock_guardrails.check_output.return_value = MagicMock(
            action=MagicMock(value="block"),
            message="Output blocked"
        )
        
        result = await rag_pipeline.query("test query yang panjang ini")
        
        assert result.scaffolding_triggered is True
    
    @pytest.mark.asyncio
    async def test_query_output_redirect(self, rag_pipeline, mock_guardrails, mock_llm):
        """Test query with output redirected (Socratic)."""
        mock_guardrails.check_input.return_value = MagicMock(
            action=MagicMock(value="allow"),
            sanitized_input="test query"
        )
        mock_guardrails.check_output.return_value = MagicMock(
            action=MagicMock(value="redirect"),
            message=None
        )
        
        result = await rag_pipeline.query("test query yang panjang ini")
        
        assert result.scaffolding_triggered is True
        assert result.answer == "Reframed response"
    
    @pytest.mark.asyncio
    async def test_query_output_sanitize(self, rag_pipeline, mock_guardrails):
        """Test query with output sanitized."""
        mock_guardrails.check_input.return_value = MagicMock(
            action=MagicMock(value="allow"),
            sanitized_input="test"
        )
        mock_guardrails.check_output.return_value = MagicMock(
            action=MagicMock(value="sanitize"),
            sanitized_input="Sanitized content"
        )
        
        result = await rag_pipeline.query("test query yang panjang")
        
        assert result.answer == "Sanitized content"
    
    @pytest.mark.asyncio
    async def test_query_exception(self, rag_pipeline, mock_guardrails, mock_vector_store):
        """Test query with exception."""
        mock_guardrails.check_input.return_value = MagicMock(
            action=MagicMock(value="allow"),
            sanitized_input="test"
        )
        mock_vector_store.search = AsyncMock(side_effect=Exception("Test error"))
        
        result = await rag_pipeline.query("test query yang panjang")
        
        assert result.success is False
        assert result.error is not None
    
    @pytest.mark.asyncio
    async def test_query_with_efficiency_guard(
        self, mock_vector_store, mock_llm, mock_guardrails, mock_efficiency_guard
    ):
        """Test query with efficiency guard enabled."""
        with patch('app.services.rag.settings') as mock_settings:
            mock_settings.ENABLE_EFFICIENCY_GUARD = True
            mock_settings.CACHE_TTL_SECONDS = 3600
            
            pipeline = RAGPipeline(
                vector_store=mock_vector_store,
                llm_service=mock_llm,
                efficiency_guard=mock_efficiency_guard
            )
            
            mock_guardrails.check_input.return_value = MagicMock(
                action=MagicMock(value="allow"),
                sanitized_input="test"
            )
            
            result = await pipeline.query("test query")
            
            assert result.success is True
    
    @pytest.mark.asyncio
    async def test_is_semantically_identical_true(self, rag_pipeline):
        """Test _is_semantically_identical returns True."""
        rag_pipeline._last_query = "Test query similar"
        rag_pipeline._last_contexts = [{"content": "context"}]
        
        with patch('app.services.rag.get_embedding_service') as mock_get:
            mock_embedder = MagicMock()
            mock_embedder.get_embedding = AsyncMock(side_effect=[
                [0.9, 0.1],  # query
                [0.89, 0.11]  # last_query
            ])
            mock_get.return_value = mock_embedder
            
            result = await rag_pipeline._is_semantically_identical("Test query similar")
            
            assert result is True
    
    @pytest.mark.asyncio
    async def test_is_semantically_identical_false(self, rag_pipeline):
        """Test _is_semantically_identical returns False."""
        rag_pipeline._last_query = "Different topic"
        rag_pipeline._last_contexts = []
        
        result = await rag_pipeline._is_semantically_identical("Test query")
        
        assert result is False
    
    @pytest.mark.asyncio
    async def test_is_semantically_identical_no_last_query(self, rag_pipeline):
        """Test _is_semantically_identical with no last query."""
        rag_pipeline._last_query = None
        
        result = await rag_pipeline._is_semantically_identical("Test query")
        
        assert result is False
    
    @pytest.mark.asyncio
    async def test_is_semantically_identical_exception(self, rag_pipeline):
        """Test _is_semantically_identical with exception."""
        rag_pipeline._last_query = "Test"
        rag_pipeline._last_contexts = [{}]
        
        with patch('app.services.rag.get_embedding_service') as mock_get:
            mock_get.side_effect = Exception("Embedding error")
            
            result = await rag_pipeline._is_semantically_identical("Test")
            
            assert result is False
    
    @pytest.mark.asyncio
    async def test_query_with_course_context(self, rag_pipeline, mock_guardrails):
        """Test query_with_course_context."""
        mock_guardrails.check_input.return_value = MagicMock(
            action=MagicMock(value="allow"),
            sanitized_input="test"
        )
        
        result = await rag_pipeline.query_with_course_context(
            query="Test question",
            course_id="course123"
        )
        
        assert result.success is True
    
    @pytest.mark.asyncio
    async def test_get_similar_questions(self, rag_pipeline, mock_vector_store):
        """Test get_similar_questions."""
        mock_vector_store.search = AsyncMock(return_value=[
            {
                "content": "Similar question 1",
                "metadata": {"answer": "Answer 1"},
                "score": 0.9
            }
        ])
        
        result = await rag_pipeline.get_similar_questions("Test query")
        
        assert len(result) == 1
        assert result[0]["question"] == "Similar question 1"
        assert result[0]["answer"] == "Answer 1"
    
    @pytest.mark.asyncio
    async def test_get_similar_questions_exception(self, rag_pipeline, mock_vector_store):
        """Test get_similar_questions with exception."""
        mock_vector_store.search = AsyncMock(side_effect=Exception("Error"))
        
        result = await rag_pipeline.get_similar_questions("Test query")
        
        assert result == []


class TestGetRAGPipeline:
    """Test get_rag_pipeline singleton."""
    
    def test_get_rag_pipeline_singleton(self):
        """Test get_rag_pipeline returns singleton."""
        from app.services import rag
        rag._rag_pipeline = None
        
        with patch('app.services.rag.get_vector_store') as mock_vs:
            mock_vs.return_value = MagicMock()
            with patch('app.services.rag.get_llm_service') as mock_llm:
                mock_llm.return_value = MagicMock()
                with patch('app.services.rag.get_guardrails') as mock_guards:
                    mock_guards.return_value = MagicMock()
                    
                    pipeline1 = get_rag_pipeline()
                    pipeline2 = get_rag_pipeline()
                    
                    assert pipeline1 is pipeline2
                    assert isinstance(pipeline1, RAGPipeline)
    
    def test_get_rag_pipeline_initialization(self):
        """Test get_rag_pipeline initializes correctly."""
        from app.services import rag
        rag._rag_pipeline = None
        
        with patch('app.services.rag.get_vector_store') as mock_vs:
            mock_vs.return_value = MagicMock()
            with patch('app.services.rag.get_llm_service') as mock_llm:
                mock_llm.return_value = MagicMock()
                with patch('app.services.rag.get_guardrails') as mock_guards:
                    mock_guards.return_value = MagicMock()
                    
                    pipeline = get_rag_pipeline()
                    
                    assert pipeline is not None
                    assert isinstance(pipeline, RAGPipeline)
