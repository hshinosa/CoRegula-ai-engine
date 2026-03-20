import pytest
import os
from unittest.mock import patch
from app.core.config import Settings

@pytest.mark.unit
def test_settings_overrides():
    # Test loading from environment
    with patch.dict(os.environ, {
        'OPENAI_API_KEY': 'env_key',
        'OPENAI_MODEL': 'env_model'
    }):
        settings = Settings()
        assert settings.OPENAI_API_KEY == 'env_key'
        assert settings.OPENAI_MODEL == 'env_model'

@pytest.mark.unit
def test_gemini_api_key_fallback():
    # Test validation logic for fallback
    with patch.dict(os.environ, {
        'GOOGLE_API_KEY': 'google_val',
        'GEMINI_API_KEY': ''
    }):
        settings = Settings()
        assert settings.GEMINI_API_KEY == 'google_val'