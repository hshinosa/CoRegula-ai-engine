# CoRegula AI-Engine

AI computation service for the CoRegula collaborative learning platform. Built with FastAPI and Google Gemini.

## Features

- **RAG (Retrieval-Augmented Generation)**: Query course materials with context-aware AI responses
- **PDF Processing**: Upload and process PDF documents for knowledge base
- **Selective OCR**: PaddleOCR-powered text extraction for scanned documents (optional)
- **Chat Intervention**: AI-driven interventions for productive discussions
- **Vector Search**: Semantic search using ChromaDB and Gemini embeddings

## Tech Stack

- **Framework**: FastAPI 0.115+
- **LLM**: Google Gemini (gemini-2.0-flash)
- **Embeddings**: Gemini text-embedding-004
- **Vector DB**: ChromaDB 0.5+
- **PDF Processing**: pypdf 5.1+
- **Python**: 3.11+

## Quick Start

### 1. Prerequisites

- Python 3.11 or higher
- Google AI Studio API key ([Get one here](https://makersuite.google.com/app/apikey))
- (Optional) [PaddlePaddle](https://www.paddlepaddle.org.cn/) runtime if you plan to enable OCR. CPU wheels are available via `pip install paddlepaddle==2.6.1 -i https://mirror.baidu.com/pypi/simple` on Windows.

### 2. Setup

```bash
# Navigate to ai-engine directory
cd ai-engine

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Copy environment file
cp .env.example .env

# Edit .env and add your Google API key
```

### 3. Run

```bash
# Development mode with auto-reload
python main.py

# Or using uvicorn directly
uvicorn main:app --reload --host 0.0.0.0 --port 8001
```

### 4. Access

- **API Docs**: http://localhost:8001/docs
- **Health Check**: http://localhost:8001/api/health

## API Endpoints

### Health
- `GET /api/health` - Service health check

### Documents
- `POST /api/documents/upload` - Upload and process PDF
- `DELETE /api/documents/{document_id}` - Delete a document

### RAG Query
- `POST /api/query` - Query knowledge base with RAG

### Chat Intervention
- `POST /api/intervention/analyze` - Analyze chat for intervention
- `POST /api/intervention/summary` - Generate discussion summary
- `POST /api/intervention/prompt` - Generate discussion prompt

### Collections
- `GET /api/collections` - List all collections
- `POST /api/collections` - Create new collection
- `DELETE /api/collections/{name}` - Delete collection

## Project Structure

```
ai-engine/
├── main.py                 # FastAPI application entry
├── requirements.txt        # Python dependencies
├── .env.example           # Environment template
├── app/
│   ├── __init__.py
│   ├── api/
│   │   ├── __init__.py
│   │   ├── routes.py      # API endpoints
│   │   └── schemas.py     # Request/Response models
│   ├── core/
│   │   ├── __init__.py
│   │   ├── config.py      # Settings management
│   │   └── logging.py     # Structured logging
│   └── services/
│       ├── __init__.py
│       ├── embeddings.py  # Gemini embeddings
│       ├── vector_store.py # ChromaDB management
│       ├── pdf_processor.py # PDF ingestion
│       ├── llm.py         # Gemini LLM service
│       ├── rag.py         # RAG pipeline
│       └── intervention.py # Chat intervention
└── data/
    └── chroma/            # Vector database storage
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `GOOGLE_API_KEY` | Google AI API key | Required |
| `GEMINI_MODEL` | LLM model name | gemini-2.0-flash |
| `GEMINI_EMBEDDING_MODEL` | Embedding model | text-embedding-004 |
| `CHROMA_PERSIST_DIR` | ChromaDB storage path | ./data/chroma |
| `MAX_UPLOAD_SIZE_MB` | Max PDF upload size | 10 |
| `CHUNK_SIZE` | Text chunk size | 1000 |
| `TOP_K_RESULTS` | RAG retrieval count | 5 |

## Development

```bash
# Format code
black .

# Lint
ruff check .

# Run tests
pytest
```

## Integration with Core-API

The AI-Engine integrates with the Core-API (Node.js backend) through HTTP endpoints. Configure `CORE_API_URL` and `CORE_API_SECRET` for secure communication.

## License

MIT License - CoRegula Project
