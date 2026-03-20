"""
Tests for LLM Service - 100% Coverage
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from app.services.llm import (
    OpenAILLMService,
    LLMResponse,
    ChatMessage,
    get_llm_service,
)


class TestLLMResponse:
    """Test LLMResponse dataclass."""
    
    def test_llm_response_success(self):
        """Test LLMResponse with success."""
        response = LLMResponse(
            content="Test content",
            tokens_used=50,
            model="test-model",
            success=True
        )
        assert response.content == "Test content"
        assert response.tokens_used == 50
        assert response.model == "test-model"
        assert response.success is True
        assert response.error is None
    
    def test_llm_response_with_error(self):
        """Test LLMResponse with error."""
        response = LLMResponse(
            content="",
            tokens_used=0,
            model="test-model",
            success=False,
            error="Test error"
        )
        assert response.success is False
        assert response.error == "Test error"


class TestChatMessage:
    """Test ChatMessage dataclass."""
    
    def test_chat_message_user(self):
        """Test ChatMessage with user role."""
        msg = ChatMessage(role="user", content="Hello")
        assert msg.role == "user"
        assert msg.content == "Hello"
    
    def test_chat_message_assistant(self):
        """Test ChatMessage with assistant role."""
        msg = ChatMessage(role="assistant", content="Hi there")
        assert msg.role == "assistant"
        assert msg.content == "Hi there"
    
    def test_chat_message_system(self):
        """Test ChatMessage with system role."""
        msg = ChatMessage(role="system", content="System message")
        assert msg.role == "system"


class TestOpenAILLMService:
    """Test OpenAILLMService class."""
    
    @pytest.fixture
    def mock_openai_client(self):
        """Create mock OpenAI client."""
        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(return_value=MagicMock(
            choices=[MagicMock(message=MagicMock(content="Test response"))],
            usage=MagicMock(total_tokens=50)
        ))
        return mock_client
    
    @pytest.fixture
    def llm_service(self, mock_openai_client):
        """Create LLM service with mocked client."""
        with patch('app.services.llm.AsyncOpenAI', return_value=mock_openai_client):
            with patch('app.services.llm.settings.OPENAI_API_KEY', 'test_key'):
                with patch('app.services.llm.settings.OPENAI_BASE_URL', 'http://test.com'):
                    with patch('app.services.llm.settings.OPENAI_MODEL', 'test-model'):
                        service = OpenAILLMService()
                        return service
    
    def test_init(self, mock_openai_client):
        """Test OpenAILLMService initialization."""
        with patch('app.services.llm.AsyncOpenAI', return_value=mock_openai_client):
            with patch('app.services.llm.settings.OPENAI_API_KEY', 'test_key'):
                with patch('app.services.llm.settings.OPENAI_BASE_URL', 'http://test.com'):
                    with patch('app.services.llm.settings.OPENAI_MODEL', 'test-model'):
                        service = OpenAILLMService()
                        
                        assert service.client is not None
                        assert service.model == 'test-model'
                        assert service.temperature == 0.0
                        assert service.max_tokens == 1000
    
    def test_init_no_api_key(self):
        """Test OpenAILLMService initialization without API key."""
        with patch('app.services.llm.settings.OPENAI_API_KEY', ''):
            with pytest.raises(ValueError, match="OPENAI_API_KEY is required"):
                OpenAILLMService()
    
    def test_system_prompts(self):
        """Test SYSTEM_PROMPTS dictionary."""
        assert 'default' in OpenAILLMService.SYSTEM_PROMPTS
        assert 'rag' in OpenAILLMService.SYSTEM_PROMPTS
        assert 'intervention' in OpenAILLMService.SYSTEM_PROMPTS
        assert 'summary' in OpenAILLMService.SYSTEM_PROMPTS
    
    @pytest.mark.asyncio
    async def test_generate_success(self, llm_service, mock_openai_client):
        """Test generate method success."""
        response = await llm_service.generate("Test prompt")
        
        assert response.success is True
        assert response.content == "Test response"
        assert response.tokens_used == 50
        assert response.model == "test-model"
    
    @pytest.mark.asyncio
    async def test_generate_with_system_prompt(self, llm_service, mock_openai_client):
        """Test generate with custom system prompt."""
        response = await llm_service.generate(
            "Test prompt",
            system_prompt="Custom system prompt"
        )
        
        assert response.success is True
    
    @pytest.mark.asyncio
    async def test_generate_with_context(self, llm_service, mock_openai_client):
        """Test generate with context."""
        context = "This is context"
        response = await llm_service.generate(
            "Test prompt",
            context=context
        )
        
        assert response.success is True
    
    @pytest.mark.asyncio
    async def test_generate_with_temperature(self, llm_service, mock_openai_client):
        """Test generate with custom temperature."""
        response = await llm_service.generate(
            "Test prompt",
            temperature=0.7
        )
        
        assert response.success is True
    
    @pytest.mark.asyncio
    async def test_generate_with_max_tokens(self, llm_service, mock_openai_client):
        """Test generate with custom max_tokens."""
        response = await llm_service.generate(
            "Test prompt",
            max_tokens=2000
        )
        
        assert response.success is True
    
    @pytest.mark.asyncio
    async def test_generate_api_error(self, llm_service, mock_openai_client):
        """Test generate with API error."""
        from openai import APIError
        
        mock_openai_client.chat.completions.create = AsyncMock(
            side_effect=APIError("Test error", response=MagicMock(), body=None)
        )
        
        with pytest.raises(APIError):
            await llm_service.generate("Test prompt")
    
    @pytest.mark.asyncio
    async def test_generate_empty_content(self, llm_service, mock_openai_client):
        """Test generate with empty content response."""
        mock_openai_client.chat.completions.create = AsyncMock(return_value=MagicMock(
            choices=[MagicMock(message=MagicMock(content=None))],
            usage=MagicMock(total_tokens=0)
        ))
        
        response = await llm_service.generate("Test prompt")
        
        assert response.success is True
        assert response.content == ""
    
    @pytest.mark.asyncio
    async def test_generate_no_usage(self, llm_service, mock_openai_client):
        """Test generate with no usage info."""
        mock_openai_client.chat.completions.create = AsyncMock(return_value=MagicMock(
            choices=[MagicMock(message=MagicMock(content="Response"))],
            usage=None
        ))
        
        response = await llm_service.generate("Test prompt")
        
        assert response.success is True
        assert response.tokens_used == 0
    
    @pytest.mark.asyncio
    async def test_generate_rag_response(self, llm_service, mock_openai_client):
        """Test generate_rag_response method."""
        contexts = [
            {"content": "Context 1"},
            {"content": "Context 2"}
        ]
        
        response = await llm_service.generate_rag_response(
            query="Test query",
            contexts=contexts
        )
        
        assert response.success is True
    
    @pytest.mark.asyncio
    async def test_generate_rag_response_with_history(self, llm_service, mock_openai_client):
        """Test generate_rag_response with chat history."""
        contexts = [{"content": "Context"}]
        history = [
            ChatMessage(role="user", content="Hello"),
            ChatMessage(role="assistant", content="Hi"),
        ]
        
        response = await llm_service.generate_rag_response(
            query="Test",
            contexts=contexts,
            chat_history=history
        )
        
        assert response.success is True
    
    @pytest.mark.asyncio
    async def test_get_scaffolding_instruction_full(self, llm_service):
        """Test _get_scaffolding_instruction with full scaffolding."""
        # Low fading level = full scaffolding
        instruction = llm_service._get_scaffolding_instruction(0.1)
        assert instruction == "Detail."
    
    @pytest.mark.asyncio
    async def test_get_scaffolding_instruction_socratic(self, llm_service):
        """Test _get_scaffolding_instruction with Socratic scaffolding."""
        # High fading level = Socratic
        instruction = llm_service._get_scaffolding_instruction(0.8)
        assert instruction == "Socratic."
    
    @pytest.mark.asyncio
    async def test_generate_intervention(self, llm_service, mock_openai_client):
        """Test generate_intervention method."""
        messages = [
            {"sender": "User1", "content": "Message 1"},
            {"sender": "User2", "content": "Message 2"},
        ]
        
        response = await llm_service.generate_intervention(
            messages=messages,
            intervention_type="redirect"
        )
        
        assert response.success is True
    
    @pytest.mark.asyncio
    async def test_generate_intervention_with_topic(self, llm_service, mock_openai_client):
        """Test generate_intervention with topic."""
        messages = [{"sender": "User1", "content": "Test"}]
        
        response = await llm_service.generate_intervention(
            messages=messages,
            intervention_type="prompt",
            topic="AI"
        )
        
        assert response.success is True
    
    @pytest.mark.asyncio
    async def test_generate_summary(self, llm_service, mock_openai_client):
        """Test generate_summary method."""
        messages = [
            {"sender": "User1", "content": "Hello"},
            {"sender": "User2", "content": "Hi"},
        ]
        
        response = await llm_service.generate_summary(messages)
        
        assert response.success is True
    
    @pytest.mark.asyncio
    async def test_generate_summary_with_action_items(self, llm_service, mock_openai_client):
        """Test generate_summary with action items."""
        messages = [{"sender": "User1", "content": "Test"}]
        
        response = await llm_service.generate_summary(
            messages,
            include_action_items=True
        )
        
        assert response.success is True
    
    @pytest.mark.asyncio
    async def test_reframe_to_socratic_success(self, llm_service, mock_openai_client):
        """Test reframe_to_socratic success."""
        response = await llm_service.reframe_to_socratic("Direct answer")
        
        assert response == "Test response"
    
    @pytest.mark.asyncio
    async def test_reframe_to_socratic_failure(self, llm_service, mock_openai_client):
        """Test reframe_to_socratic with failure."""
        mock_openai_client.chat.completions.create = AsyncMock(
            side_effect=Exception("API Error")
        )
        
        original = "Direct answer"
        response = await llm_service.reframe_to_socratic(original)
        
        # Should return original on failure
        assert response == original
    
    @pytest.mark.asyncio
    async def test_get_goal_refinement_suggestion(self, llm_service, mock_openai_client):
        """Test get_goal_refinement_suggestion method."""
        response = await llm_service.get_goal_refinement_suggestion(
            current_goal="Learn Python",
            missing_criteria=["Specific", "Time-bound"]
        )
        
        assert response.success is True
    
    def test_format_contexts(self, llm_service):
        """Test _format_contexts method."""
        contexts = [
            {"content": "First context"},
            {"content": "Second context"},
        ]
        
        result = llm_service._format_contexts(contexts)
        
        assert "[1] First context" in result
        assert "[2] Second context" in result
    
    def test_format_contexts_empty(self, llm_service):
        """Test _format_contexts with empty contexts."""
        result = llm_service._format_contexts([])
        assert result == ""
    
    def test_format_contexts_no_content(self, llm_service):
        """Test _format_contexts with no content."""
        contexts = [{"other": "data"}]
        result = llm_service._format_contexts(contexts)
        assert "[1] " in result
    
    def test_format_chat_history(self, llm_service):
        """Test _format_chat_history method."""
        history = [
            ChatMessage(role="user", content="Hello"),
            ChatMessage(role="assistant", content="Hi there"),
        ]
        
        result = llm_service._format_chat_history(history)
        
        assert "User: Hello" in result
        assert "AI: Hi there" in result
    
    def test_format_chat_history_truncated(self, llm_service):
        """Test _format_chat_history truncates to last 5."""
        history = [ChatMessage(role="user", content=f"Msg {i}") for i in range(10)]
        
        result = llm_service._format_chat_history(history)
        
        # Should only include last 5
        assert "Msg 5" in result
        assert "Msg 0" not in result
    
    def test_format_chat_history_empty(self, llm_service):
        """Test _format_chat_history with empty history."""
        result = llm_service._format_chat_history([])
        assert result == ""


class TestGetLLMService:
    """Test get_llm_service singleton."""
    
    def test_get_llm_service_singleton(self):
        """Test get_llm_service returns singleton."""
        from app.services import llm
        llm._llm_service = None
        
        with patch('app.services.llm.settings.OPENAI_API_KEY', 'test_key'):
            with patch('app.services.llm.settings.OPENAI_BASE_URL', 'http://test.com'):
                with patch('app.services.llm.settings.OPENAI_MODEL', 'test-model'):
                    service1 = get_llm_service()
                    service2 = get_llm_service()
                    
                    assert service1 is service2
                    assert isinstance(service1, OpenAILLMService)
    
    def test_get_llm_service_initialization(self):
        """Test get_llm_service initializes correctly."""
        from app.services import llm
        llm._llm_service = None
        
        with patch('app.services.llm.settings.OPENAI_API_KEY', 'test_key'):
            with patch('app.services.llm.settings.OPENAI_BASE_URL', 'http://test.com'):
                with patch('app.services.llm.settings.OPENAI_MODEL', 'test-model'):
                    service = get_llm_service()
                    
                    assert service is not None
                    assert service.model is not None
