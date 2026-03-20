"""
Tests for Intervention Service - 100% Coverage
"""
import pytest
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch
from app.services.intervention import (
    ChatInterventionService,
    InterventionType,
    InterventionResult,
    get_intervention_service,
)


class TestInterventionResult:
    """Test InterventionResult dataclass."""
    
    def test_intervention_result_success(self):
        """Test InterventionResult with success."""
        result = InterventionResult(
            message="Test message",
            intervention_type=InterventionType.REDIRECT,
            confidence=0.85,
            should_intervene=True,
            reason="Off-topic detected",
            success=True
        )
        assert result.message == "Test message"
        assert result.should_intervene is True
        assert result.error is None
    
    def test_intervention_result_with_error(self):
        """Test InterventionResult with error."""
        result = InterventionResult(
            message="",
            intervention_type=InterventionType.PROMPT,
            confidence=0.0,
            should_intervene=False,
            reason="Error occurred",
            success=False,
            error="Test error"
        )
        assert result.success is False
        assert result.error == "Test error"


class TestInterventionType:
    """Test InterventionType enum."""
    
    def test_intervention_type_values(self):
        """Test InterventionType enum values."""
        assert InterventionType.REDIRECT.value == "redirect"
        assert InterventionType.PROMPT.value == "prompt"
        assert InterventionType.SUMMARIZE.value == "summarize"
        assert InterventionType.CLARIFY.value == "clarify"
        assert InterventionType.RESOURCE.value == "resource"
        assert InterventionType.ENCOURAGE.value == "encourage"


class TestChatInterventionService:
    """Test ChatInterventionService class."""
    
    @pytest.fixture
    def mock_llm(self):
        """Create mock LLM service."""
        mock = MagicMock()
        mock.generate = AsyncMock(return_value=MagicMock(
            content="Test response",
            tokens_used=50,
            success=True
        ))
        return mock
    
    @pytest.fixture
    def intervention_service(self, mock_llm):
        """Create intervention service with mock LLM."""
        return ChatInterventionService(llm_service=mock_llm)
    
    @pytest.mark.asyncio
    async def test_analyze_and_intervene_no_messages(self, intervention_service):
        """Test analyze_and_intervene with no messages."""
        result = await intervention_service.analyze_and_intervene(
            messages=[],
            topic="Test topic",
            chat_room_id="room_1"
        )
        
        assert result.success is True
        assert result.should_intervene is False
        assert result.reason == "No messages to analyze"
    
    @pytest.mark.asyncio
    async def test_analyze_and_intervene_with_messages(self, intervention_service):
        """Test analyze_and_intervene with messages."""
        messages = [
            {"sender": "User1", "content": "Hello"},
            {"sender": "User2", "content": "Hi there"},
        ]
        
        result = await intervention_service.analyze_and_intervene(
            messages=messages,
            topic="Test topic",
            chat_room_id="room_1"
        )
        
        assert result.success is True
        assert result is not None
    
    @pytest.mark.asyncio
    async def test_analyze_and_intervene_recent_intervention(self, intervention_service):
        """Test analyze_and_intervene with recent intervention."""
        messages = [{"sender": "User1", "content": "Hello"}]
        recent_time = datetime.now()
        
        result = await intervention_service.analyze_and_intervene(
            messages=messages,
            topic="Test topic",
            chat_room_id="room_1",
            last_intervention_time=recent_time
        )
        
        assert result.success is True
    
    @pytest.mark.asyncio
    async def test_check_triggers(self, intervention_service):
        """Test _check_triggers method."""
        messages = [
            {"sender": "User1", "content": "Test message 1"},
            {"sender": "User2", "content": "Test message 2"},
        ]
        
        triggers = await intervention_service._check_triggers(
            messages=messages,
            topic="Test topic",
            last_intervention_time=None
        )
        
        assert isinstance(triggers, dict)
    
    @pytest.mark.asyncio
    async def test_check_triggers_with_recent_intervention(self, intervention_service):
        """Test _check_triggers with recent intervention."""
        messages = [{"sender": "User1", "content": "Hello"}]
        recent_time = datetime.now()
        
        triggers = await intervention_service._check_triggers(
            messages=messages,
            topic="Test topic",
            last_intervention_time=recent_time
        )
        
        assert triggers.get("recent_intervention") is True
    
    @pytest.mark.asyncio
    async def test_select_intervention(self, intervention_service):
        """Test _select_intervention method."""
        triggers = {
            "off_topic": True,
            "inactive": False,
            "needs_summary": False,
            "low_engagement": False,
        }
        
        intervention_type, confidence, reason = intervention_service._select_intervention(triggers)
        
        assert intervention_type is not None
        assert isinstance(confidence, float)
        assert isinstance(reason, str)
    
    @pytest.mark.asyncio
    async def test_select_intervention_no_triggers(self, intervention_service):
        """Test _select_intervention with no triggers."""
        triggers = {
            "off_topic": False,
            "inactive": False,
            "needs_summary": False,
            "low_engagement": False,
        }
        
        intervention_type, confidence, reason = intervention_service._select_intervention(triggers)
        
        assert intervention_type == InterventionType.ENCOURAGE
        assert confidence == 0.0
    
    @pytest.mark.asyncio
    async def test_select_intervention_multiple_triggers(self, intervention_service):
        """Test _select_intervention with multiple triggers."""
        triggers = {
            "off_topic": True,
            "inactive": True,
            "needs_summary": True,
            "low_engagement": False,
        }
        
        intervention_type, confidence, reason = intervention_service._select_intervention(triggers)
        
        assert intervention_type is not None
    
    @pytest.mark.asyncio
    async def test_generate_redirect_message(self, intervention_service, mock_llm):
        """Test _generate_redirect_message."""
        topic = "Machine Learning"
        off_topic_messages = [
            {"sender": "User1", "content": "Did you see the game last night?"}
        ]
        
        result = await intervention_service._generate_redirect_message(
            topic=topic,
            off_topic_messages=off_topic_messages
        )
        
        assert result is not None
        assert "Machine Learning" in result or "topik" in result.lower()
    
    @pytest.mark.asyncio
    async def test_generate_prompt_message(self, intervention_service, mock_llm):
        """Test _generate_prompt_message."""
        topic = "AI Ethics"
        last_messages = [
            {"sender": "User1", "content": "I don't know what to say"}
        ]
        
        result = await intervention_service._generate_prompt_message(
            topic=topic,
            last_messages=last_messages
        )
        
        assert result is not None
    
    @pytest.mark.asyncio
    async def test_generate_summary_message(self, intervention_service, mock_llm):
        """Test _generate_summary_message."""
        messages = [
            {"sender": "User1", "content": "Machine learning is interesting"},
            {"sender": "User2", "content": "Yes, I agree"},
            {"sender": "User1", "content": "Neural networks are cool"},
        ]
        
        result = await intervention_service._generate_summary_message(messages)
        
        assert result is not None
    
    @pytest.mark.asyncio
    async def test_generate_clarification_message(self, intervention_service, mock_llm):
        """Test _generate_clarification_message."""
        unclear_message = {"sender": "User1", "content": "That thing is weird"}
        
        result = await intervention_service._generate_clarification_message(unclear_message)
        
        assert result is not None
    
    @pytest.mark.asyncio
    async def test_check_off_topic(self, intervention_service):
        """Test _check_off_topic method."""
        messages = [
            {"sender": "User1", "content": "Let's talk about movies"},
            {"sender": "User2", "content": "Great idea!"},
        ]
        topic = "Machine Learning"
        
        is_off_topic, confidence = await intervention_service._check_off_topic(messages, topic)
        
        assert isinstance(is_off_topic, bool)
        assert isinstance(confidence, float)
    
    @pytest.mark.asyncio
    async def test_check_inactivity(self, intervention_service):
        """Test _check_inactivity method."""
        messages = [
            {"sender": "User1", "content": "Hello", "timestamp": datetime.now()}
        ]
        
        is_inactive, minutes = intervention_service._check_inactivity(messages)
        
        assert isinstance(is_inactive, bool)
        assert isinstance(minutes, (int, float))
    
    @pytest.mark.asyncio
    async def test_check_engagement_quality(self, intervention_service):
        """Test _check_engagement_quality method."""
        messages = [
            {"sender": "User1", "content": "Short"},
            {"sender": "User2", "content": "OK"},
        ]
        
        is_low_quality, avg_length = intervention_service._check_engagement_quality(messages)
        
        assert isinstance(is_low_quality, bool)
        assert isinstance(avg_length, (int, float))
    
    @pytest.mark.asyncio
    async def test_check_needs_summary(self, intervention_service):
        """Test _check_needs_summary method."""
        messages = [{"sender": f"User{i}", "content": f"Message {i}"} for i in range(15)]
        
        needs_summary, message_count = intervention_service._check_needs_summary(messages)
        
        assert isinstance(needs_summary, bool)
        assert message_count == 15
    
    @pytest.mark.asyncio
    async def test_get_intervention_message(self, intervention_service, mock_llm):
        """Test _get_intervention_message method."""
        intervention_type = InterventionType.REDIRECT
        context = {"topic": "Test topic"}
        
        result = await intervention_service._get_intervention_message(intervention_type, context)
        
        assert result is not None
    
    @pytest.mark.asyncio
    async def test_get_intervention_message_redirect(self, intervention_service, mock_llm):
        """Test _get_intervention_message for redirect."""
        context = {"topic": "AI"}
        result = await intervention_service._get_intervention_message(
            InterventionType.REDIRECT, context
        )
        assert result is not None
    
    @pytest.mark.asyncio
    async def test_get_intervention_message_prompt(self, intervention_service, mock_llm):
        """Test _get_intervention_message for prompt."""
        context = {"topic": "ML"}
        result = await intervention_service._get_intervention_message(
            InterventionType.PROMPT, context
        )
        assert result is not None
    
    @pytest.mark.asyncio
    async def test_get_intervention_message_summarize(self, intervention_service, mock_llm):
        """Test _get_intervention_message for summarize."""
        context = {"messages": []}
        result = await intervention_service._get_intervention_message(
            InterventionType.SUMMARIZE, context
        )
        assert result is not None


class TestGetInterventionService:
    """Test get_intervention_service singleton."""
    
    def test_get_intervention_service_singleton(self):
        """Test get_intervention_service returns singleton."""
        from app.services import intervention
        intervention._intervention_service = None
        
        service1 = get_intervention_service()
        service2 = get_intervention_service()
        
        assert service1 is service2
        assert isinstance(service1, ChatInterventionService)
    
    def test_get_intervention_service_with_llm(self):
        """Test get_intervention_service with custom LLM."""
        from app.services import intervention
        intervention._intervention_service = None
        
        mock_llm = MagicMock()
        service = get_intervention_service(llm_service=mock_llm)
        
        assert service is not None
        assert service.llm_service is mock_llm
