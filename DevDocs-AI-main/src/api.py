"""
FastAPI Backend for DevDocs AI
-------------------------------
RESTful API for document upload and intelligent querying.
"""

import os
import time
import logging
from typing import List, Optional, Dict, Any
from datetime import datetime

from fastapi import FastAPI, File, UploadFile, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# Add parent directory to path
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import core components
from src.document_processor import DocumentProcessor
from src.rag_engine import create_rag_engine

from config import (
    PROJECT_NAME,
    VERSION,
    DESCRIPTION,
    EXAMPLE_QUERIES,
    SUPPORTED_EXTENSIONS,
    RAW_DOCS_DIR
)

# ---------------------------------------------------------------------------
# Logging Configuration
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# FastAPI App
# ---------------------------------------------------------------------------
app = FastAPI(
    title=PROJECT_NAME,
    version=VERSION,
    description=DESCRIPTION,
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Pydantic Models
# ---------------------------------------------------------------------------
class QueryRequest(BaseModel):
    """Request model for queries."""
    question: str = Field(..., min_length=1, description="User's question")
    filters: Optional[Dict[str, str]] = Field(None, description="Optional metadata filters")
    
    class Config:
        schema_extra = {
            "example": {
                "question": "How do I create a FastAPI endpoint?",
                "filters": {"content_type": "api_documentation"}
            }
        }


class QueryResponse(BaseModel):
    """Response model for queries."""
    question: str
    answer: str
    sources: List[Dict[str, Any]]
    confidence: float
    chunks_used: int
    processing_time_ms: float


class UploadResponse(BaseModel):
    """Response model for document uploads."""
    message: str
    document_name: str
    chunks_processed: int
    processing_time_ms: float


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    version: str
    timestamp: str
    collection_stats: Dict[str, Any]


class ErrorResponse(BaseModel):
    """Error response model."""
    error: str
    details: Optional[str] = None
    timestamp: str


# ---------------------------------------------------------------------------
# Global State
# ---------------------------------------------------------------------------
rag_engine = None
doc_processor = None


@app.on_event("startup")
async def startup_event():
    """Initialize components on startup."""
    global rag_engine, doc_processor
    
    try:
        logger.info("🚀 Starting DevDocs AI...")
        
        # Initialize document processor
        doc_processor = DocumentProcessor()
        logger.info("✅ Document processor initialized")
        
        # Initialize RAG engine
        rag_engine = create_rag_engine()
        logger.info("✅ RAG engine initialized")
        
        # Ensure data directory exists
        os.makedirs(RAW_DOCS_DIR, exist_ok=True)
        
        logger.info("✅ Startup complete")
        
    except Exception as e:
        logger.exception(f"❌ Startup failed: {e}")
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("👋 Shutting down DevDocs AI...")


# ---------------------------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------------------------
@app.get("/", response_model=Dict[str, Any])
async def root():
    """Root endpoint with API information."""
    return {
        "name": PROJECT_NAME,
        "version": VERSION,
        "description": DESCRIPTION,
        "status": "operational",
        "endpoints": {
            "health": "/health",
            "upload": "/upload",
            "query": "/query",
            "documents": "/documents",
            "examples": "/examples",
            "stats": "/stats"
        }
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    try:
        stats = rag_engine.get_collection_stats() if rag_engine else {}
        
        return HealthResponse(
            status="healthy",
            version=VERSION,
            timestamp=datetime.now().isoformat(),
            collection_stats=stats
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return HealthResponse(
            status="degraded",
            version=VERSION,
            timestamp=datetime.now().isoformat(),
            collection_stats={"error": str(e)}
        )


@app.post("/upload", response_model=UploadResponse, status_code=status.HTTP_201_CREATED)
async def upload_document(
    file: UploadFile = File(...),
    document_name: Optional[str] = None
):
    """
    Upload and process a document.
    
    - **file**: PDF or TXT file
    - **document_name**: Optional custom name for the document
    """
    start_time = time.time()
    
    try:
        # Validate file type
        file_extension = os.path.splitext(file.filename)[1].lower()
        if file_extension not in SUPPORTED_EXTENSIONS:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unsupported file type: {file_extension}. Supported: {', '.join(SUPPORTED_EXTENSIONS)}"
            )
        
        # Use provided name or original filename
        doc_name = document_name or file.filename
        
        # Save uploaded file temporarily
        temp_path = os.path.join(RAW_DOCS_DIR, f"temp_{int(time.time())}_{doc_name}")
        
        try:
            # Write file
            with open(temp_path, "wb") as f:
                content = await file.read()
                f.write(content)
            
            logger.info(f"Processing uploaded file: {doc_name}")
            
            # Process document
            chunks = doc_processor.process_file(temp_path, doc_name)
            
            # Add to vector database
            chunks_added = rag_engine.add_documents(chunks)
            
            processing_time = (time.time() - start_time) * 1000
            
            logger.info(f"✅ Uploaded {doc_name}: {chunks_added} chunks in {processing_time:.0f}ms")
            
            return UploadResponse(
                message="Document processed successfully",
                document_name=doc_name,
                chunks_processed=chunks_added,
                processing_time_ms=round(processing_time, 2)
            )
            
        finally:
            # Clean up temp file
            if os.path.exists(temp_path):
                os.remove(temp_path)
                
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Upload failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Document processing failed: {str(e)}"
        )


@app.post("/query", response_model=QueryResponse)
async def query_knowledge_base(request: QueryRequest):
    """
    Query the knowledge base.
    
    - **question**: Your question in natural language
    - **filters**: Optional metadata filters (e.g., content_type, importance)
    """
    try:
        # Validate
        if not rag_engine:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="RAG engine not initialized"
            )
        
        if not request.question.strip():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Question cannot be empty"
            )
        
        logger.info(f"Query: {request.question}")
        
        # Process query
        start_time = time.time()
        result = rag_engine.query(request.question, request.filters)
        processing_time = (time.time() - start_time) * 1000
        
        response = QueryResponse(
            question=result["question"],
            answer=result["answer"],
            sources=result["sources"],
            confidence=result["confidence"],
            chunks_used=result["chunks_used"],
            processing_time_ms=round(processing_time, 2)
        )
        
        logger.info(f"✅ Query completed: {processing_time:.0f}ms, confidence: {result['confidence']}")
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Query failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Query processing failed: {str(e)}"
        )


@app.get("/documents", response_model=Dict[str, Any])
async def list_documents():
    """Get information about indexed documents."""
    try:
        stats = rag_engine.get_collection_stats()
        return {
            "total_chunks": stats.get("total_chunks", 0),
            "collection_name": stats.get("collection_name", "unknown"),
            "embedding_model": stats.get("embedding_model", "unknown"),
            "message": "Use /stats for detailed statistics"
        }
    except Exception as e:
        logger.error(f"Failed to list documents: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@app.delete("/documents", status_code=status.HTTP_200_OK)
async def clear_documents():
    """Clear all documents from the knowledge base."""
    try:
        rag_engine.clear_collection()
        logger.info("✅ Collection cleared")
        return {"message": "All documents cleared successfully"}
    except Exception as e:
        logger.error(f"Failed to clear documents: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@app.get("/examples", response_model=Dict[str, List[str]])
async def get_examples():
    """Get example queries."""
    return {"examples": EXAMPLE_QUERIES}


@app.get("/stats", response_model=Dict[str, Any])
async def get_stats():
    """Get detailed system statistics."""
    try:
        collection_stats = rag_engine.get_collection_stats()
        
        return {
            "system": {
                "name": PROJECT_NAME,
                "version": VERSION,
                "status": "operational"
            },
            "knowledge_base": collection_stats
        }
    except Exception as e:
        logger.error(f"Failed to get stats: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


# ---------------------------------------------------------------------------
# Error Handlers
# ---------------------------------------------------------------------------
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc: HTTPException):
    """Handle HTTP exceptions."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "timestamp": datetime.now().isoformat()
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc: Exception):
    """Handle general exceptions."""
    logger.exception(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "Internal server error",
            "details": str(exc),
            "timestamp": datetime.now().isoformat()
        }
    )


# ---------------------------------------------------------------------------
# Main Entry Point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    
    print("="*60)
    print(f"🚀 Starting {PROJECT_NAME} v{VERSION}")
    print("="*60)
    print(f"📚 API Documentation: http://localhost:8000/docs")
    print(f"🔧 Health Check: http://localhost:8000/health")
    print("="*60)
    
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )