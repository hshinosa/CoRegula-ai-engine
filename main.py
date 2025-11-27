"""
CoRegula AI-Engine
===================
FastAPI backend for AI computation, RAG pipeline, and LLM integration.
Uses Google Gemini as the LLM provider.
"""

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from app.core.config import settings
from app.core.logging import setup_logging, get_logger
from app.api.routes import router as api_router
from app.services.vector_store import get_vector_store

# Setup logging
setup_logging()
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler for startup/shutdown events."""
    # Startup
    logger.info("Starting CoRegula AI-Engine", version="1.0.0", env=settings.ENV)
    
    # Initialize vector store
    vector_store = get_vector_store()
    await vector_store.initialize()
    logger.info("Vector store initialized", persist_dir=settings.CHROMA_PERSIST_DIR)
    
    yield
    
    # Shutdown
    logger.info("Shutting down CoRegula AI-Engine")


# Create FastAPI application
app = FastAPI(
    title="CoRegula AI-Engine",
    description="AI computation service for collaborative learning platform",
    version="1.0.0",
    docs_url="/docs" if settings.DEBUG else None,
    redoc_url="/redoc" if settings.DEBUG else None,
    lifespan=lifespan,
)

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        settings.CORE_API_URL,
        "http://localhost:3000",
        "http://localhost:8000",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:8000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(api_router, prefix="/api")


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "service": "CoRegula AI-Engine",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs" if settings.DEBUG else "disabled",
    }


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level=settings.LOG_LEVEL.lower(),
    )
