# CoRegula AI-Engine

AI computation service for the CoRegula collaborative learning platform. Built with FastAPI, Python 3.11, and Google Gemini 2.0 Flash.

## 🎯 Purpose

AI-Engine provides intelligent support for collaborative learning:
- **RAG (Retrieval-Augmented Generation)**: Answer questions using course materials as context
- **Document Processing**: Extract text and images from PDF, DOCX, PPTX files
- **Vector Search**: Semantic similarity search using ChromaDB + Gemini embeddings
- **NLP Analytics**: Measure engagement, detect Higher Order Thinking (HOT), analyze lexical variety
- **Chat Intervention**: AI-driven prompts and guidance for productive discussions
- **OCR Support**: Optional PaddleOCR for scanned documents (gracefully disabled if unavailable)

## 🛠️ Tech Stack

- **Framework**: FastAPI 0.115+ (async Python web framework)
- **LLM**: Google Generative AI (Gemini 2.0 Flash)
- **Embeddings**: Gemini text-embedding-004
- **Vector Database**: ChromaDB 0.5+ (in-memory + persistent)
- **Document Processing**:
  - PyPDF 5.1+ (PDF text extraction)
  - python-docx (Word documents)
  - python-pptx (PowerPoint presentations)
  - Pillow (Image processing)
  - PaddleOCR 3.3.2 (Optional - scanned document text extraction)
- **NLP**: NLTK (tokenization, analysis)
- **Machine Learning**: scikit-learn (engagement metrics)
- **Data**: Pydantic v2 (validation), Python 3.11+
- **Process Logging**: Structured EPM-format event logs to MongoDB

## 📋 Prerequisites

- Python 3.11+ (tested with 3.11.9)
- Google AI Studio API key ([Get free key](https://makersuite.google.com/app/apikey))
- Pip or Conda for package management
- (Optional) MongoDB for persistent chat logs
- (Optional) PaddleOCR for scanned document support

## 🚀 Quick Start

### 1. Setup Virtual Environment

```bash
cd ai-engine

# Create virtual environment
python -m venv venv

# Activate
# Windows:
venv\Scripts\activate

# Linux/Mac:
source venv/bin/activate
```

### 2. Install Dependencies

```bash
# Install with binary wheels (avoids paddle build issues)
pip install --prefer-binary -r requirements.txt
```

**Note on PaddleOCR**: The optional `paddlepaddle` dependency gracefully degrades. If installation fails, FastAPI still runs with OCR disabled (single warning log instead of repeated errors).

### 3. Configure Environment

```bash
# Copy template
cp .env.example .env

# Edit and add your Google API key
# nano .env
```

**Essential .env variables:**

```env
GOOGLE_API_KEY=your-google-ai-studio-key
GEMINI_MODEL=gemini-2.0-flash
GEMINI_EMBEDDING_MODEL=text-embedding-004
HOST=0.0.0.0
PORT=8001
CHROMA_PERSIST_DIR=./data/chroma
CORE_API_URL=http://localhost:3000
MONGO_URI=mongodb://localhost:27017/coregula
```

### 4. Run Development Server

```bash
# Development with auto-reload
python main.py

# Or using uvicorn directly:
uvicorn main:app --reload --host 0.0.0.0 --port 8001
```

Server runs on `http://localhost:8001`

### 5. Access API Documentation

- **Interactive Docs (Swagger)**: http://localhost:8001/docs
- **Alternative Docs (ReDoc)**: http://localhost:8001/redoc
- **Health Check**: http://localhost:8001/api/health

## 📁 Project Structure

```
ai-engine/
├── main.py                          # FastAPI app entry point
├── requirements.txt                 # Python dependencies
├── .env.example                     # Environment template
├── Dockerfile                       # Docker build configuration
├── app/
│   ├── __init__.py
│   ├── api/
│   │   ├── __init__.py
│   │   ├── routes.py               # FastAPI endpoints
│   │   └── schemas.py              # Pydantic request/response models
│   ├── core/
│   │   ├── __init__.py
│   │   ├── config.py               # Settings & environment validation
│   │   ├── guardrails.py           # Safety checks (academic honesty, toxicity)
│   │   └── logging.py              # Process mining event logger
│   └── services/
│       ├── __init__.py
│       ├── llm.py                  # Gemini LLM wrapper
│       ├── embeddings.py           # Embedding generation
│       ├── vector_store.py         # ChromaDB management
│       ├── document_processor.py   # PDF/DOCX/PPTX + OCR processing
│       ├── rag.py                  # RAG pipeline (policy-based)
│       ├── nlp_analytics.py        # HOT detection, engagement scoring
│       ├── orchestration.py        # Main AI response pipeline
│       └── chat_analytics.py       # Chat quality & intervention analysis
├── utils/
│   ├── __init__.py
│   ├── logger.py                   # EPM-format event logging
│   └── helpers.py                  # Utility functions
└── data/
    ├── chroma/                     # ChromaDB persistent storage
    └── event_logs/                 # Process mining JSON logs
```

## 📡 API Endpoints

### Health Check

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/health` | Service status and dependencies |

**Response:**
```json
{
  "status": "ok",
  "timestamp": "2025-11-27T10:00:00Z",
  "dependencies": {
    "gemini": "available",
    "chromadb": "available",
    "ocr": "unavailable"
  }
}
```

### Document Ingestion

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/ingest/batch` | Upload & process multiple documents |
| POST | `/api/ingest/document` | Upload single document |
| DELETE | `/api/documents/{doc_id}` | Delete document from knowledge base |
| GET | `/api/collections` | List all knowledge base collections |

**Batch Upload Example:**
```bash
curl -X POST http://localhost:8001/api/ingest/batch \
  -H "Authorization: Bearer optional-secret" \
  -F "files=@document1.pdf" \
  -F "files=@document2.pdf" \
  -F "course_id=HCI2024" \
  -F "extract_images=true" \
  -F "perform_ocr=false"
```

**Response:**
```json
{
  "documents": [
    {
      "document_id": "doc-uuid-1",
      "filename": "document1.pdf",
      "pages": 25,
      "chunks": 45,
      "success": true,
      "message": "Processed successfully"
    },
    {
      "document_id": "doc-uuid-2",
      "filename": "document2.pdf",
      "pages": 18,
      "chunks": 32,
      "success": true
    }
  ],
  "total_documents": 2,
  "total_chunks": 77
}
```

### RAG Query

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/query` | Query knowledge base with RAG |
| POST | `/api/ask` | Ask AI with course context |

**Query Request:**
```json
{
  "question": "What is the definition of usability?",
  "course_id": "HCI2024",
  "top_k": 5,
  "include_sources": true
}
```

**Query Response:**
```json
{
  "answer": "Usability is the degree to which...",
  "sources": [
    {
      "document": "Chapter_3.pdf",
      "page": 42,
      "content": "Usability refers to..."
    }
  ],
  "confidence": 0.92,
  "timestamp": "2025-11-27T10:05:00Z"
}
```

### Chat Intervention

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/intervention/analyze` | Analyze chat for intervention opportunities |
| POST | `/api/intervention/summary` | Generate discussion summary |
| POST | `/api/intervention/prompt` | Generate guided discussion prompt |

**Analyze Request:**
```json
{
  "messages": [
    {"user": "student1", "content": "I think..."},
    {"user": "student2", "content": "But what about..."}
  ],
  "group_id": "group-123",
  "course_id": "HCI2024"
}
```

**Response:**
```json
{
  "needs_intervention": true,
  "engagement_level": "low",
  "hot_questions": [
    {
      "question": "Can you explain why that approach is better?",
      "target_student": "student1",
      "reason": "Missing critical analysis"
    }
  ],
  "quality_metrics": {
    "lexical_variety": 0.65,
    "hot_percentage": 0.15,
    "engagement_type": "cognitive"
  }
}
```

## 🔄 RAG Pipeline

### Policy-Based RAG

The RAG system intelligently decides whether to retrieve documents:

```
User Question
    ↓
Is it a greeting? (e.g., "Hi", "Hello") → SKIP RETRIEVAL
    ↓
Is it a simple fact? (e.g., "What time is it?") → SKIP RETRIEVAL
    ↓
Complex/Course-related question → FETCH from ChromaDB
    ↓
Semantic Search (top_k documents)
    ↓
Pass Context + Question to Gemini 2.0 Flash
    ↓
Generate Response
    ↓
Return Answer + Source Documents
```

**Benefits:**
- Faster responses for simple questions
- Reduced hallucination risk
- Proper attribution when using course materials
- Cost-effective LLM calls

### Document Processing Flow

```
User Upload (PDF/DOCX/PPTX)
    ↓
Text Extraction (PyPDF / python-docx / python-pptx)
    ↓
Optional: Image Extraction & OCR (PaddleOCR)
    ↓
Text Chunking (semantic chunks ~1000 tokens)
    ↓
Generate Embeddings (Gemini text-embedding-004)
    ↓
Store in ChromaDB (with metadata)
    ↓
Return success confirmation
```

## 📊 NLP Analytics

### Engagement Metrics

```typescript
// Calculate engagement scores
{
  "lexical_variety": 0.68,      // Unique word ratio (0-1)
  "hot_percentage": 0.25,        // Higher Order Thinking % (0-1)
  "engagement_type": "cognitive", // cognitive | behavioral | emotional
  "message_quality": 0.82        // Overall quality score
}
```

### HOT (Higher Order Thinking) Detection

Identifies indicators of critical thinking:
- Questions starting with "why", "how", "what if"
- Comparing/contrasting concepts
- Problem-solving language
- Justification patterns

### Engagement Classification

- **Cognitive**: Analysis, synthesis, evaluation
- **Behavioral**: Participation rate, response time
- **Emotional**: Sentiment, supportiveness, tone

## 🌐 Environment Variables

```env
# Server
HOST=0.0.0.0
PORT=8001
DEBUG=false

# Google API
GOOGLE_API_KEY=your-api-key (REQUIRED)
GEMINI_MODEL=gemini-2.0-flash
GEMINI_EMBEDDING_MODEL=text-embedding-004

# ChromaDB
CHROMA_PERSIST_DIR=./data/chroma
CHROMA_COLLECTION_PREFIX=coregula_

# Document Processing
MAX_UPLOAD_SIZE_MB=50
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
EXTRACT_IMAGES=false
PERFORM_OCR=false

# RAG
TOP_K_RESULTS=5
RELEVANCE_THRESHOLD=0.3

# Integration
CORE_API_URL=http://localhost:3000
CORE_API_SECRET=optional-bearer-token
MONGO_URI=mongodb://localhost:27017/coregula

# Logging
LOG_LEVEL=info
ENABLE_PROCESS_MINING=true

# Features
ENABLE_GUARDRAILS=true
ENABLE_INTERVENTION=true
ENABLE_ANALYTICS=true
```

## 🏗️ Building & Deployment

### Development

```bash
# Auto-reload server
python main.py

# Or explicit uvicorn:
uvicorn main:app --reload --host 0.0.0.0 --port 8001
```

### Production

```bash
# Non-reloading server
uvicorn main:app --host 0.0.0.0 --port 8001 --workers 4
```

### Docker

```bash
# Build
docker build -t coregula-ai .

# Run
docker run -p 8001:8001 \
  -e GOOGLE_API_KEY=your-key \
  -e CORE_API_URL=http://core-api:3000 \
  -v ai_engine_data:/app/data \
  coregula-ai
```

### Docker Compose

```bash
docker-compose up -d ai-engine
```

## 📚 Scripts

| Script | Description |
|--------|-------------|
| `python main.py` | Start development server |
| `uvicorn main:app --reload` | Explicit uvicorn reload |
| `black .` | Format code |
| `ruff check .` | Lint code |
| `pytest` | Run tests |
| `pytest --cov` | Coverage report |

## 🧪 Testing

```bash
# All tests
pytest

# Specific test file
pytest tests/test_rag.py

# With coverage
pytest --cov=app tests/

# Watch mode
pytest-watch
```

## 📖 Dependencies

### Core
- `fastapi` - Web framework
- `uvicorn` - ASGI server
- `pydantic` - Data validation

### AI & NLP
- `google-generativeai` - Gemini API
- `chromadb` - Vector database
- `nltk` - NLP toolkit
- `scikit-learn` - ML algorithms

### Document Processing
- `pypdf` - PDF text extraction
- `python-docx` - Word documents
- `python-pptx` - PowerPoint
- `pillow` - Image processing
- `paddleocr` - OCR (optional, graceful fallback)
- `paddlepaddle` - PaddleOCR runtime (optional)

### Data & Logging
- `pymongo` - MongoDB integration
- `python-dotenv` - Environment variables

## 🔗 Integration with Core-API

Core-API calls these endpoints:

```typescript
// Document upload
POST /api/ingest/batch
Headers: { Authorization: Bearer <optional-secret> }
Files: FormData with PDFs

// RAG query (when @AI mentioned in chat)
POST /api/ask
Body: { question, course_id, context }

// Chat analysis (intervention)
POST /api/intervention/analyze
Body: { messages, group_id, course_id }
```

## 🐛 Troubleshooting

### PaddleOCR Import Errors

```
Problem: "No module named 'paddle'"
Solution: Gracefully handled - OCR is optional and disabled with single warning log
```

The system detects missing PaddleOCR at startup and:
- Sets `OCR_AVAILABLE = False`
- Logs one-time warning instead of repeated errors
- Continues processing without OCR

### Google API Key Invalid

```bash
# Verify key
echo $GOOGLE_API_KEY

# Get new key: https://makersuite.google.com/app/apikey
# Update .env and restart
```

### ChromaDB Connection Issues

```bash
# Check persistence directory
ls -la ./data/chroma

# Reset database (loses all documents)
rm -rf ./data/chroma
```

### Memory Issues with Large PDFs

```python
# In .env, reduce chunk settings:
CHUNK_SIZE=500
MAX_UPLOAD_SIZE_MB=10
```

## 📄 License

MIT License - CoRegula Project

## 👥 Support

For issues, open an issue in the CoRegula repository.
