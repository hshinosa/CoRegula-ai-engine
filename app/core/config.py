"""
Application Configuration
=========================
Pydantic settings for environment variable management.
"""

import os
from pathlib import Path
from typing import Literal, Optional
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import model_validator
from functools import lru_cache


# Get the ai-engine directory path
AI_ENGINE_DIR = Path(__file__).resolve().parent.parent.parent


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # Application
    VERSION: str = "1.0.0"
    
    # Server Configuration
    HOST: str = "0.0.0.0"
    PORT: int = 8001
    ENV: Literal["development", "production", "testing"] = "development"
    DEBUG: bool = True
    
    # Google Gemini API
    GOOGLE_API_KEY: str = ""
    GEMINI_API_KEY: str = ""  # Alias for GOOGLE_API_KEY
    GEMINI_MODEL: str = "gemini-2.0-flash"
    GEMINI_EMBEDDING_MODEL: str = "models/text-embedding-004"
    GEMINI_TEMPERATURE: float = 0.7
    GEMINI_TOP_P: float = 0.95
    GEMINI_MAX_OUTPUT_TOKENS: int = 2048
    
    # Vector Database (ChromaDB)
    CHROMA_PERSIST_DIR: str = "./data/chroma"
    CHROMA_COLLECTION_PREFIX: str = "coregula"
    
    # Document Processing
    MAX_FILE_SIZE_MB: int = 10
    MAX_UPLOAD_SIZE_MB: int = 10
    MAX_ZIP_SIZE_MB: int = 50  # Max size for ZIP files
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    ENABLE_OCR: bool = False
    
    # RAG Configuration
    TOP_K_RESULTS: int = 5
    SIMILARITY_THRESHOLD: float = 0.7
    RAG_MIN_QUERY_WORDS: int = 3  # Minimum words for FETCH policy
    
    # NLP Analytics (SSRL Metrics)
    NLP_LOW_LEXICAL_THRESHOLD: float = 0.3  # Below this = shallow discussion
    NLP_HOT_TARGET_PERCENTAGE: float = 40.0  # Target % of HOT messages
    NLP_QUALITY_ALERT_THRESHOLD: float = 30.0  # Notify teacher below this
    
    # Orchestration (Teacher-AI Complementarity)
    INTERVENTION_COOLDOWN_MINUTES: int = 5  # Min time between interventions
    INTERVENTION_MIN_MESSAGES: int = 5  # Min messages before quality check
    NOTIFY_TEACHER_ON_LOW_QUALITY: bool = True
    
    # Core-API Integration
    CORE_API_URL: str = "http://localhost:3000"
    CORE_API_SECRET: str = ""
    
    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: Literal["json", "console"] = "json"
    
    @model_validator(mode='after')
    def set_gemini_api_key(self) -> 'Settings':
        """Use GOOGLE_API_KEY as fallback for GEMINI_API_KEY."""
        if not self.GEMINI_API_KEY and self.GOOGLE_API_KEY:
            self.GEMINI_API_KEY = self.GOOGLE_API_KEY
        return self
    
    model_config = SettingsConfigDict(
        env_file=str(AI_ENGINE_DIR / ".env"),
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",  # Ignore extra environment variables (from Laravel, etc.)
    )


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


# Global settings instance
settings = get_settings()
