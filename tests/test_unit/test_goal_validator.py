import pytest
import json
from unittest.mock import AsyncMock, patch, MagicMock
from app.services.goal_validator import (
    GoalValidator,
    SMARTCriterion,
    SMARTValidationResult,
    get_goal_validator,
)

@pytest.fixture
def validator():
    return GoalValidator()

@pytest.mark.unit
@pytest.mark.parametrize("text,expected", [
    ("Saya ingin membuat aplikasi", True),
    ("Target kita adalah mendesain database", True),
    ("Halo semua", False),
])
def test_is_goal_statement(validator, text, expected):
    assert validator.is_goal_statement(text) == expected

@pytest.mark.unit
def test_validate_goal_valid(validator, sample_valid_goal):
    result = validator.validate_goal(sample_valid_goal)
    assert result.is_valid is True
    assert result.score == 1.0

@pytest.mark.unit
def test_validate_goal_invalid(validator, sample_invalid_goal):
    result = validator.validate_goal(sample_invalid_goal)
    assert result.is_valid is False
    assert result.score < 1.0

@pytest.mark.unit
@pytest.mark.asyncio
async def test_refine_goal_success(validator, mock_llm_service):
    mock_json = {"refined_goal": "goal", "explanation": "exp"}
    mock_llm_service.generate.return_value.content = json.dumps(mock_json)
    with patch("app.services.llm.get_llm_service", return_value=mock_llm_service):
        result = await validator.refine_goal("old", ["specific"])
        assert result["success"] is True
        assert result["refined_goal"] == "goal"

@pytest.mark.unit
def test_get_goal_validator_singleton():
    v1 = get_goal_validator()
    v2 = get_goal_validator()
    assert v1 is v2