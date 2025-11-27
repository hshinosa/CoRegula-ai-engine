"""
API Routes - MVP Phase 1-1.5
CoRegula AI Engine

FastAPI router with all API endpoints.
"""

import uuid
from pathlib import Path
from datetime import datetime
from typing import Optional, List

from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Query
from fastapi.responses import JSONResponse

from app.core.config import settings
from app.core.logging import get_logger
from app.services.pdf_processor import get_pdf_processor
from app.services.document_processor import get_document_processor
from app.services.vector_store import get_vector_store
from app.services.rag import get_rag_pipeline
from app.services.intervention import get_intervention_service, InterventionType
from app.services.orchestration import get_orchestrator
from app.services.nlp_analytics import get_engagement_analyzer
from app.utils.logger import get_process_mining_logger
from app.api.schemas import (
    HealthResponse,
    PDFUploadResponse,
    DocumentProcessResult,
    BatchUploadResponse,
    IngestResponse,
    DocumentListResponse,
    DocumentInfo,
    QueryRequest,
    QueryResponse,
    SourceInfo,
    AskRequest,
    AskResponse,
    InterventionRequest,
    InterventionResponse,
    SummaryRequest,
    SummaryResponse,
    PromptRequest,
    PromptResponse,
    CreateCollectionRequest,
    CollectionResponse,
    CollectionListResponse,
    ErrorResponse,
    OrchestrationRequest,
    OrchestrationResponse,
    GroupAnalyticsRequest,
    GroupAnalyticsResponse,
    EngagementAnalysisRequest,
    EngagementAnalysisResponse,
    ProcessMiningExportResponse,
    GuardrailCheckRequest,
    GuardrailCheckResponse,
)
from app.core.guardrails import get_guardrails, GuardrailAction

logger = get_logger(__name__)

# Create router
router = APIRouter()


# ============== Core-API Integration Endpoints ==============
# These endpoints are called by Core-API (Express.js backend)

@router.post(
    "/ask",
    response_model=AskResponse,
    tags=["Core-API Integration"],
    summary="Answer question using RAG (for chat @AI mention)"
)
async def ask_question(request: AskRequest):
    """
    Answer a question using RAG - called when user mentions @AI in chat.
    
    This is the primary endpoint used by Core-API for chat AI responses.
    
    Args:
        request: AskRequest with query, course_id, user_name, chat_space_id
    
    Returns:
        AskResponse with answer field
    """
    try:
        rag_pipeline = get_rag_pipeline()
        
        # Use course-specific collection
        collection_name = f"course_{request.course_id}"
        
        result = await rag_pipeline.query(
            query=request.query,
            collection_name=collection_name,
            n_results=5
        )
        
        if result.success:
            # Format answer with sources if available
            answer = result.answer
            if result.sources:
                sources_text = "\n\n📚 *Sumber:*\n"
                for i, src in enumerate(result.sources[:3], 1):
                    source_name = src.get("source", "Dokumen")
                    page = src.get("page")
                    if page:
                        sources_text += f"{i}. {source_name} (hal. {page})\n"
                    else:
                        sources_text += f"{i}. {source_name}\n"
                answer += sources_text
            
            return AskResponse(answer=answer, success=True)
        else:
            return AskResponse(
                answer="Maaf, saya tidak bisa menemukan jawaban untuk pertanyaan tersebut dalam materi kuliah.",
                success=False,
                error=result.error
            )
            
    except Exception as e:
        logger.error("ask_question_failed", query=request.query[:100], error=str(e))
        return AskResponse(
            answer="Maaf, terjadi kesalahan saat memproses pertanyaan. Silakan coba lagi.",
            success=False,
            error=str(e)
        )


@router.post(
    "/ingest",
    response_model=IngestResponse,
    tags=["Core-API Integration"],
    summary="Ingest document into vector store (supports PDF, DOCX, PPTX, TXT, ZIP)"
)
async def ingest_document(
    file: UploadFile = File(...),
    course_id: str = Form(...),
    file_id: str = Form(...),
):
    """
    Ingest a document into the vector store.
    
    Supports multiple formats:
    - PDF (with image extraction and OCR)
    - DOCX (Microsoft Word)
    - PPTX (Microsoft PowerPoint)
    - TXT, MD (Plain text, Markdown)
    - ZIP (Archive containing multiple documents)
    
    Called by Core-API Knowledge Base service when a lecturer uploads a file.
    
    Args:
        file: Document file to process
        course_id: Course ID to associate the document with
        file_id: File ID from the database (for tracking)
    
    Returns:
        Processing result with chunks and stats
    """
    start_time = datetime.now()
    
    if not file.filename:
        raise HTTPException(status_code=400, detail="Filename is required")
    
    # Get file extension
    ext = Path(file.filename).suffix.lower()
    supported_extensions = ['.pdf', '.docx', '.doc', '.pptx', '.ppt', '.txt', '.md', '.zip']
    
    if ext not in supported_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {ext}. Supported: {', '.join(supported_extensions)}"
        )
    
    # Read file content
    contents = await file.read()
    
    # Validate file size (50MB for ZIP, 10MB for others)
    if ext == '.zip':
        max_size = settings.MAX_ZIP_SIZE_MB * 1024 * 1024
    else:
        max_size = settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024
    
    if len(contents) > max_size:
        raise HTTPException(
            status_code=400,
            detail=f"File size exceeds limit"
        )
    
    try:
        # Use document processor for comprehensive processing
        doc_processor = get_document_processor()
        document_id = file_id
        collection_name = f"course_{course_id}"
        
        result = await doc_processor.process_file(
            file_content=contents,
            filename=file.filename,
            document_id=document_id,
            collection_name=collection_name,
            course_id=course_id,
            metadata={
                "course_id": course_id,
                "file_id": file_id,
                "original_filename": file.filename,
                "upload_time": datetime.now().isoformat()
            }
        )
        
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        if result.success:
            logger.info(
                "document_ingested",
                file_id=file_id,
                course_id=course_id,
                filename=file.filename,
                file_type=result.file_type,
                chunks=len(result.chunks),
                pages=result.page_count,
                images=result.image_count,
                processing_time_ms=processing_time
            )
            
            return IngestResponse(
                success=True,
                message="Document ingested successfully",
                file_id=file_id,
                document_id=document_id,
                chunks_created=len(result.chunks),
                page_count=result.page_count,
                image_count=result.image_count,
                file_type=result.file_type,
                processing_time_ms=processing_time
            )
        else:
            raise HTTPException(
                status_code=500,
                detail=result.error or "Failed to process document"
            )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "document_ingest_failed",
            file_id=file_id,
            filename=file.filename,
            error=str(e)
        )
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process document: {str(e)}"
        )


@router.post(
    "/ingest/batch",
    response_model=BatchUploadResponse,
    tags=["Core-API Integration"],
    summary="Batch ingest multiple documents or ZIP archive"
)
async def ingest_batch(
    files: List[UploadFile] = File(...),
    course_id: str = Form(...),
    extract_images: bool = Form(True),
    perform_ocr: bool = Form(False),
):
    """
    Batch ingest multiple documents at once.
    
    Accepts multiple files or a single ZIP archive containing documents.
    
    Args:
        files: List of document files to process
        course_id: Course ID to associate documents with
    
    Returns:
        Batch processing result with individual document stats
    """
    start_time = datetime.now()
    doc_processor = get_document_processor()
    collection_name = f"course_{course_id}"
    
    all_results: List[DocumentProcessResult] = []
    total_chunks = 0
    
    for i, file in enumerate(files):
        if not file.filename:
            continue
            
        try:
            contents = await file.read()
            document_id = f"{course_id}_{datetime.now().strftime('%Y%m%d%H%M%S')}_{i}"
            
            result = await doc_processor.process_file(
                file_content=contents,
                filename=file.filename,
                document_id=document_id,
                collection_name=collection_name,
                course_id=course_id,
                metadata={
                    "course_id": course_id,
                    "batch_index": i,
                    "original_filename": file.filename,
                    "upload_time": datetime.now().isoformat(),
                    "extract_images": extract_images,
                    "perform_ocr": perform_ocr,
                }
            )
            
            chunks_count = len(result.chunks) if result.chunks else 0
            total_chunks += chunks_count
            
            all_results.append(DocumentProcessResult(
                filename=result.filename,
                file_type=result.file_type,
                chunks_created=chunks_count,
                page_count=result.page_count,
                image_count=result.image_count,
                total_characters=result.total_characters,
                processing_time_ms=result.processing_time_ms,
                success=result.success,
                error=result.error
            ))
            
        except Exception as e:
            logger.error("batch_file_failed", filename=file.filename, error=str(e))
            all_results.append(DocumentProcessResult(
                filename=file.filename or "unknown",
                file_type="unknown",
                chunks_created=0,
                page_count=0,
                image_count=0,
                total_characters=0,
                processing_time_ms=0,
                success=False,
                error=str(e)
            ))
    
    processing_time = (datetime.now() - start_time).total_seconds() * 1000
    successful = sum(1 for r in all_results if r.success)
    failed = len(all_results) - successful
    
    logger.info(
        "batch_ingest_complete",
        course_id=course_id,
        total=len(all_results),
        successful=successful,
        failed=failed,
        total_chunks=total_chunks,
        processing_time_ms=processing_time
    )
    
    return BatchUploadResponse(
        success=successful > 0,
        message=f"Processed {successful}/{len(all_results)} files successfully",
        total_files=len(all_results),
        successful_files=successful,
        failed_files=failed,
        total_chunks=total_chunks,
        documents=all_results,
        processing_time_ms=processing_time
    )


# ============== Health Check ==============

@router.get(
    "/health",
    response_model=HealthResponse,
    tags=["Health"],
    summary="Health check endpoint"
)
async def health_check():
    """
    Check the health status of the AI Engine.
    
    Returns status of all dependent services.
    """
    services = {
        "vector_store": False,
        "llm": False,
    }
    
    # Check vector store
    try:
        vector_store = get_vector_store()
        # Simple check - try to list collections
        await vector_store._ensure_collection("health_check")
        services["vector_store"] = True
    except Exception as e:
        logger.warning("health_check_vector_store_failed", error=str(e))
    
    # Check LLM (just verify it's initialized)
    try:
        from app.services.llm import get_llm_service
        llm = get_llm_service()
        services["llm"] = llm.model is not None
    except Exception as e:
        logger.warning("health_check_llm_failed", error=str(e))
    
    overall_status = "healthy" if all(services.values()) else "degraded"
    
    return HealthResponse(
        status=overall_status,
        version=settings.VERSION,
        timestamp=datetime.now(),
        services=services
    )


# ============== PDF Upload & Document Management ==============

@router.post(
    "/documents/upload",
    response_model=PDFUploadResponse,
    tags=["Documents"],
    summary="Upload and process a PDF document"
)
async def upload_pdf(
    file: UploadFile = File(...),
    course_id: Optional[str] = Form(None),
    collection_name: Optional[str] = Form(None),
):
    """
    Upload a PDF document for processing.
    
    The document will be:
    1. Validated (must be PDF, under size limit)
    2. Text extracted
    3. Chunked into smaller pieces
    4. Embedded and stored in vector database
    
    Args:
        file: PDF file to upload
        course_id: Optional course ID to associate
        collection_name: Optional collection name (defaults to course_id or 'default')
    
    Returns:
        Upload result with document ID and processing stats
    """
    start_time = datetime.now()
    
    # Validate file type
    if not file.filename or not file.filename.lower().endswith('.pdf'):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are allowed"
        )
    
    # Validate file size
    contents = await file.read()
    if len(contents) > settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024:
        raise HTTPException(
            status_code=400,
            detail=f"File size exceeds {settings.MAX_UPLOAD_SIZE_MB}MB limit"
        )
    
    try:
        # Generate document ID
        document_id = str(uuid.uuid4())
        
        # Determine collection
        target_collection = collection_name or (f"course_{course_id}" if course_id else "default")
        
        # Process PDF
        pdf_processor = get_pdf_processor()
        result = await pdf_processor.process_pdf(
            pdf_content=contents,
            filename=file.filename,
            document_id=document_id,
            collection_name=target_collection,
            metadata={
                "course_id": course_id,
                "original_filename": file.filename,
                "upload_time": datetime.now().isoformat()
            }
        )
        
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        logger.info(
            "pdf_upload_complete",
            document_id=document_id,
            filename=file.filename,
            chunks=result.get("chunks_count", 0),
            processing_time_ms=processing_time
        )
        
        return PDFUploadResponse(
            success=True,
            message="Document uploaded and processed successfully",
            document_id=document_id,
            filename=file.filename,
            chunks_created=result.get("chunks_count", 0),
            processing_time_ms=processing_time
        )
        
    except Exception as e:
        logger.error(
            "pdf_upload_failed",
            filename=file.filename,
            error=str(e)
        )
        
        return PDFUploadResponse(
            success=False,
            message="Failed to process document",
            error=str(e),
            processing_time_ms=(datetime.now() - start_time).total_seconds() * 1000
        )


@router.delete(
    "/documents/{document_id}",
    response_model=CollectionResponse,
    tags=["Documents"],
    summary="Delete a document"
)
async def delete_document(
    document_id: str,
    collection_name: Optional[str] = Query(None)
):
    """
    Delete a document and its chunks from the vector store.
    
    Args:
        document_id: ID of the document to delete
        collection_name: Optional collection name
    
    Returns:
        Deletion result
    """
    try:
        vector_store = get_vector_store()
        await vector_store.delete_documents(
            ids=[document_id],
            collection_name=collection_name,
            where={"document_id": document_id}
        )
        
        logger.info("document_deleted", document_id=document_id)
        
        return CollectionResponse(
            success=True,
            name=collection_name or "default",
            message=f"Document {document_id} deleted successfully"
        )
        
    except Exception as e:
        logger.error("document_delete_failed", document_id=document_id, error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


# ============== RAG Query ==============

@router.post(
    "/query",
    response_model=QueryResponse,
    tags=["RAG"],
    summary="Query documents using RAG"
)
async def query_documents(request: QueryRequest):
    """
    Query the knowledge base using RAG.
    
    The query will:
    1. Search vector store for relevant document chunks
    2. Use retrieved context to generate an answer
    3. Return answer with source citations
    
    Args:
        request: Query request with question and options
    
    Returns:
        Answer with sources and metadata
    """
    try:
        rag_pipeline = get_rag_pipeline()
        
        # Determine collection
        collection_name = f"course_{request.course_id}" if request.course_id else None
        
        result = await rag_pipeline.query(
            query=request.query,
            collection_name=collection_name,
            n_results=request.n_results
        )
        
        # Format sources
        sources = []
        if request.include_sources:
            sources = [
                SourceInfo(
                    source=s.get("source", "Unknown"),
                    page=s.get("page"),
                    chunk_index=s.get("chunk_index"),
                    relevance_score=s.get("relevance_score", 0)
                )
                for s in result.sources
            ]
        
        return QueryResponse(
            success=result.success,
            answer=result.answer,
            sources=sources,
            query=result.query,
            tokens_used=result.tokens_used,
            processing_time_ms=result.processing_time_ms,
            error=result.error
        )
        
    except Exception as e:
        logger.error("query_failed", query=request.query[:100], error=str(e))
        return QueryResponse(
            success=False,
            answer="",
            sources=[],
            query=request.query,
            error=str(e)
        )


# ============== Chat Intervention ==============

@router.post(
    "/intervention/analyze",
    response_model=InterventionResponse,
    tags=["Intervention"],
    summary="Analyze chat and generate intervention"
)
async def analyze_chat(request: InterventionRequest):
    """
    Analyze a chat conversation and generate an intervention if needed.
    
    The analysis checks for:
    - Off-topic discussions
    - Inactivity
    - Low engagement
    - Need for summary
    
    Args:
        request: Chat messages and context
    
    Returns:
        Intervention decision and message
    """
    try:
        intervention_service = get_intervention_service()
        
        # Convert messages to dict format
        messages = [
            {
                "sender": m.sender,
                "content": m.content,
                "timestamp": m.timestamp.isoformat() if m.timestamp else None,
                "sender_id": m.sender_id
            }
            for m in request.messages
        ]
        
        # If specific type requested, generate that type
        if request.intervention_type and request.force:
            result = await intervention_service.llm_service.generate_intervention(
                chat_messages=messages,
                intervention_type=request.intervention_type,
                topic=request.topic
            )
            
            return InterventionResponse(
                success=result.success,
                should_intervene=True,
                message=result.content,
                intervention_type=request.intervention_type,
                confidence=1.0,
                reason="Forced intervention",
                error=result.error
            )
        
        # Otherwise, analyze and decide
        result = await intervention_service.analyze_and_intervene(
            messages=messages,
            topic=request.topic,
            chat_room_id=request.chat_room_id
        )
        
        return InterventionResponse(
            success=result.success,
            should_intervene=result.should_intervene,
            message=result.message,
            intervention_type=result.intervention_type.value,
            confidence=result.confidence,
            reason=result.reason,
            error=result.error
        )
        
    except Exception as e:
        logger.error("intervention_analysis_failed", error=str(e))
        return InterventionResponse(
            success=False,
            should_intervene=False,
            message="",
            intervention_type="error",
            confidence=0,
            reason=str(e),
            error=str(e)
        )


@router.post(
    "/intervention/summary",
    response_model=SummaryResponse,
    tags=["Intervention"],
    summary="Generate discussion summary"
)
async def generate_summary(request: SummaryRequest):
    """
    Generate a summary of the chat discussion.
    
    Args:
        request: Chat messages to summarize
    
    Returns:
        Summary with key points
    """
    try:
        intervention_service = get_intervention_service()
        
        messages = [
            {
                "sender": m.sender,
                "content": m.content,
                "timestamp": m.timestamp.isoformat() if m.timestamp else None
            }
            for m in request.messages
        ]
        
        result = await intervention_service.generate_summary(
            messages=messages,
            chat_room_id=request.chat_room_id
        )
        
        return SummaryResponse(
            success=result.success,
            summary=result.message,
            message_count=len(request.messages),
            error=result.error
        )
        
    except Exception as e:
        logger.error("summary_generation_failed", error=str(e))
        return SummaryResponse(
            success=False,
            summary="",
            message_count=len(request.messages),
            error=str(e)
        )


@router.post(
    "/intervention/prompt",
    response_model=PromptResponse,
    tags=["Intervention"],
    summary="Generate discussion prompt"
)
async def generate_prompt(request: PromptRequest):
    """
    Generate a discussion prompt for the given topic.
    
    Args:
        request: Topic and difficulty level
    
    Returns:
        Generated discussion prompt
    """
    try:
        intervention_service = get_intervention_service()
        
        result = await intervention_service.generate_discussion_prompt(
            topic=request.topic,
            context=request.context,
            difficulty=request.difficulty
        )
        
        return PromptResponse(
            success=result.success,
            prompt=result.message,
            topic=request.topic,
            error=result.error
        )
        
    except Exception as e:
        logger.error("prompt_generation_failed", error=str(e))
        return PromptResponse(
            success=False,
            prompt="",
            topic=request.topic,
            error=str(e)
        )


# ============== Collection Management ==============

@router.post(
    "/collections",
    response_model=CollectionResponse,
    tags=["Collections"],
    summary="Create a new collection"
)
async def create_collection(request: CreateCollectionRequest):
    """
    Create a new vector store collection.
    
    Args:
        request: Collection details
    
    Returns:
        Created collection info
    """
    try:
        vector_store = get_vector_store()
        await vector_store._ensure_collection(request.name)
        
        logger.info("collection_created", name=request.name)
        
        return CollectionResponse(
            success=True,
            name=request.name,
            message="Collection created successfully"
        )
        
    except Exception as e:
        logger.error("collection_create_failed", name=request.name, error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/collections",
    response_model=CollectionListResponse,
    tags=["Collections"],
    summary="List all collections"
)
async def list_collections():
    """
    List all vector store collections.
    
    Returns:
        List of collections with counts
    """
    try:
        vector_store = get_vector_store()
        collections_data = await vector_store.list_collections()
        
        return CollectionListResponse(
            success=True,
            collections=collections_data,
            total=len(collections_data)
        )
        
    except Exception as e:
        logger.error("collections_list_failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.delete(
    "/collections/{collection_name}",
    response_model=CollectionResponse,
    tags=["Collections"],
    summary="Delete a collection"
)
async def delete_collection(collection_name: str):
    """
    Delete a vector store collection.
    
    Args:
        collection_name: Name of collection to delete
    
    Returns:
        Deletion result
    """
    try:
        vector_store = get_vector_store()
        await vector_store.delete_collection(collection_name)
        
        logger.info("collection_deleted", name=collection_name)
        
        return CollectionResponse(
            success=True,
            name=collection_name,
            message="Collection deleted successfully"
        )
        
    except Exception as e:
        logger.error("collection_delete_failed", name=collection_name, error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


# ============== Orchestration (Teacher-AI Complementarity) ==============

@router.post(
    "/chat",
    response_model=OrchestrationResponse,
    tags=["Orchestration"],
    summary="Handle chat message with full orchestration pipeline"
)
async def chat_endpoint(request: OrchestrationRequest):
    """
    Main endpoint for orchestrated chat handling.
    
    Implements Teacher-AI Complementarity framework:
    1. Analyzes engagement (Cognitive, Behavioral, Emotional)
    2. Generates RAG response with policy optimization (FETCH/NO_FETCH)
    3. Logs events for Educational Process Mining
    4. Triggers interventions when discussion quality is low
    5. Notifies teacher if quality drops below threshold
    
    Args:
        request: OrchestrationRequest with user_id, group_id, message
    
    Returns:
        OrchestrationResponse with bot_response, intervention, analytics
    """
    try:
        orchestrator = get_orchestrator()
        
        result = await orchestrator.handle_message(
            user_id=request.user_id,
            group_id=request.group_id,
            message=request.message,
            topic=request.topic,
            collection_name=request.collection_name,
            course_id=request.course_id,
            chat_room_id=request.chat_room_id
        )
        
        return OrchestrationResponse(
            success=result.success,
            bot_response=result.reply,
            system_intervention=result.intervention,
            intervention_type=result.intervention_type,
            action_taken=result.action_taken,
            should_notify_teacher=result.should_notify_teacher,
            quality_score=result.quality_score,
            meta=result.analytics,
            error=result.error
        )
        
    except Exception as e:
        logger.error("orchestration_failed", error=str(e))
        return OrchestrationResponse(
            success=False,
            bot_response="Maaf, terjadi kesalahan sistem.",
            action_taken="ERROR",
            error=str(e)
        )


@router.get(
    "/analytics/group/{group_id}",
    response_model=GroupAnalyticsResponse,
    tags=["Analytics"],
    summary="Get group analytics and discussion quality"
)
async def get_group_analytics(group_id: str):
    """
    Get aggregated analytics for a group's discussion.
    
    Returns:
    - Quality score with breakdown
    - Engagement type distribution
    - HOT (Higher-Order Thinking) percentage
    - Recommendations for improvement
    
    Args:
        group_id: The group ID to get analytics for
    
    Returns:
        GroupAnalyticsResponse with metrics
    """
    try:
        orchestrator = get_orchestrator()
        analytics = await orchestrator.get_group_analytics(group_id)
        
        return GroupAnalyticsResponse(
            success=True,
            group_id=group_id,
            message_count=analytics.get("message_count", 0),
            quality_score=analytics.get("quality_score"),
            quality_breakdown=analytics.get("quality_breakdown", {}),
            recommendation=analytics.get("recommendation"),
            participants=analytics.get("participants", []),
            participant_count=analytics.get("participant_count", 0),
            engagement_distribution=analytics.get("engagement_distribution", {})
        )
        
    except Exception as e:
        logger.error("group_analytics_failed", group_id=group_id, error=str(e))
        return GroupAnalyticsResponse(
            success=False,
            group_id=group_id,
            error=str(e)
        )


@router.post(
    "/analytics/engagement",
    response_model=EngagementAnalysisResponse,
    tags=["Analytics"],
    summary="Analyze text for engagement metrics"
)
async def analyze_engagement(request: EngagementAnalysisRequest):
    """
    Analyze a text for engagement metrics (SSRL).
    
    Computes:
    - Lexical Variety (vocabulary richness)
    - Higher-Order Thinking detection
    - Engagement type classification
    
    Args:
        request: EngagementAnalysisRequest with text
    
    Returns:
        EngagementAnalysisResponse with metrics
    """
    try:
        analyzer = get_engagement_analyzer()
        result = analyzer.analyze_interaction(request.text)
        
        return EngagementAnalysisResponse(
            success=True,
            lexical_variety=result.lexical_variety,
            engagement_type=result.engagement_type.value,
            is_higher_order=result.is_higher_order,
            hot_indicators=result.hot_indicators,
            word_count=result.word_count,
            unique_words=result.unique_words,
            confidence=result.confidence
        )
        
    except Exception as e:
        logger.error("engagement_analysis_failed", error=str(e))
        return EngagementAnalysisResponse(
            success=False,
            lexical_variety=0,
            engagement_type="unknown",
            is_higher_order=False,
            word_count=0,
            unique_words=0,
            confidence=0,
            error=str(e)
        )


@router.get(
    "/analytics/export",
    response_model=ProcessMiningExportResponse,
    tags=["Analytics"],
    summary="Export event logs for Process Mining (ProM/Disco)"
)
async def export_process_mining_data():
    """
    Export event logs in a format ready for Educational Process Mining.
    
    The exported CSV follows EPM standards with:
    - CaseID (Group ID)
    - Activity (Interaction type)
    - Timestamp
    - Resource (User/Agent ID)
    
    Compatible with ProM, Disco, PM4Py.
    
    Returns:
        Export result with file URL
    """
    try:
        pm_logger = get_process_mining_logger()
        
        # Export to ProM format
        export_path = pm_logger.export_for_prom()
        stats = pm_logger.get_statistics()
        
        return ProcessMiningExportResponse(
            success=True,
            file_url=f"/data/event_logs/export_prom.csv",
            total_events=stats.get("total_events", 0),
            unique_cases=stats.get("unique_cases", 0),
            message="Export ready for ProM/Disco import"
        )
        
    except Exception as e:
        logger.error("process_mining_export_failed", error=str(e))
        return ProcessMiningExportResponse(
            success=False,
            file_url="",
            error=str(e)
        )


# ============== Guardrails ==============

@router.post(
    "/guardrails/check",
    response_model=GuardrailCheckResponse,
    tags=["Guardrails"],
    summary="Check text against safety guardrails"
)
async def check_guardrails(request: GuardrailCheckRequest):
    """
    Check text input against safety guardrails.
    
    Validates for:
    - Academic dishonesty (homework completion requests)
    - Off-topic content
    - Harmful/dangerous content
    - PII (Personally Identifiable Information)
    - Toxicity and profanity
    
    Args:
        request: GuardrailCheckRequest with text to check
    
    Returns:
        GuardrailCheckResponse with action and details
    """
    try:
        guardrails = get_guardrails()
        result = guardrails.check_input(request.text, request.context)
        
        return GuardrailCheckResponse(
            allowed=result.action == GuardrailAction.ALLOW,
            action=result.action.value,
            reason=result.reason,
            message=result.message,
            sanitized_text=result.sanitized_input,
            triggered_rules=result.triggered_rules or [],
            confidence=result.confidence
        )
        
    except Exception as e:
        logger.error("guardrails_check_failed", error=str(e))
        return GuardrailCheckResponse(
            allowed=True,  # Fail open for safety
            action="allow",
            reason="check_failed",
            message=str(e),
            triggered_rules=[],
            confidence=0
        )


@router.post(
    "/guardrails/check-output",
    response_model=GuardrailCheckResponse,
    tags=["Guardrails"],
    summary="Check AI output against safety guardrails"
)
async def check_output_guardrails(
    response_text: str = Form(...),
    original_query: str = Form(...)
):
    """
    Check AI-generated output against safety guardrails.
    
    Validates for:
    - Complete homework solutions (should provide hints instead)
    - PII in generated content
    - Harmful content in responses
    
    Args:
        response_text: Generated AI response to check
        original_query: Original user query for context
    
    Returns:
        GuardrailCheckResponse with action and sanitized output if needed
    """
    try:
        guardrails = get_guardrails()
        result = guardrails.check_output(response_text, original_query)
        
        return GuardrailCheckResponse(
            allowed=result.action == GuardrailAction.ALLOW,
            action=result.action.value,
            reason=result.reason,
            message=result.message,
            sanitized_text=result.sanitized_input,
            triggered_rules=result.triggered_rules or [],
            confidence=result.confidence
        )
        
    except Exception as e:
        logger.error("output_guardrails_check_failed", error=str(e))
        return GuardrailCheckResponse(
            allowed=True,
            action="allow",
            reason="check_failed",
            triggered_rules=[],
            confidence=0
        )
