import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from app.services.rag import RAGPipeline, RAGResult

@pytest.fixture
def mock_deps():
    return {
        'vs': AsyncMock(),
        'llm': AsyncMock(),
        'eg': AsyncMock()
    }

@pytest.mark.asyncio
async def test_query_direct_execution(mock_deps):
    with patch('app.services.rag.get_guardrails') as mock_gr:
        mock_gr.return_value.check_input.return_value = MagicMock(action='allow', sanitized_input=None)
        pipe = RAGPipeline(vector_store=mock_deps['vs'], llm_service=mock_deps['llm'], efficiency_guard=None)
        
        mock_deps['llm'].generate.return_value = MagicMock(content='Direct Answer', success=True)
        
        # Test NO_FETCH path
        res = await pipe.query('halo')
        assert res.answer == 'Direct Answer'