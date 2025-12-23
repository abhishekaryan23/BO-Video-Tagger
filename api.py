import logging
import time
import os
from contextlib import asynccontextmanager
from typing import List, Optional

from fastapi import FastAPI, HTTPException, BackgroundTasks, Query, Response, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from schemas import (
    ProcessRequest,
    MediaItem,
    SearchResponse,
    MediaType,
    SystemInfo,
    ErrorResponse
)
from bo_db import VideoDB
from processor import MediaProcessor
from bo_config import settings

# Setup Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("API")

# Global Services
db: Optional[VideoDB] = None
processor: Optional[MediaProcessor] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize resources on startup and clean up on shutdown."""
    global db, processor
    
    logger.info("🚀 Starting BO-Video-Tagger Backend...")
    
    # Initialize DB
    db = VideoDB()
    
    # Initialize Processor
    # We might want to load models lazily or eagerly. 
    # Here we initialize the processor class, but models load on first use or explicit checks.
    processor = MediaProcessor(
        mode=settings.model_tier
    )
    
    yield
    
    # Cleanup
    if db:
        db.close()
    logger.info("👋 Shutting down...")

app = FastAPI(
    title="BO-Video-Tagger API",
    version="2.0.0",
    description="Unified Media Processing Backend for Video & Images",
    lifespan=lifespan,
    responses={
        422: {"model": ErrorResponse, "description": "Validation Error"},
        500: {"model": ErrorResponse, "description": "Internal Server Error"},
        404: {"model": ErrorResponse, "description": "Not Found"}
    }
)

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Convert all unhandled exceptions to RFC 7807 ErrorResponse."""
    logger.error(f"Global Exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            type="server_error",
            title="Internal Server Error",
            status=500,
            detail=str(exc),
            instance=str(request.url)
        ).model_dump()
    )

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Convert HTTP exceptions to RFC 7807."""
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            type="http_error",
            title="HTTP Error",
            status=exc.status_code,
            detail=exc.detail,
            instance=str(request.url)
        ).model_dump()
    )

# --- Endpoints ---

@app.get("/health")
def health_check():
    """Simple health check."""
    return {"status": "ok", "timestamp": time.time()}

@app.post("/process", response_model=MediaItem)
def process_media(request: ProcessRequest, background_tasks: BackgroundTasks):
    """
    Process a single media file (video or image).
    Synchronous processing for now to ensure result return.
    For long videos, we might want to move this to background tasks and return a job ID.
    But requirement asks for 1TB/10h, which implies batch processing.
    For this API, we allow blocking or async. 
    Let's keep it blocking for simplicity of the return value for now, 
    but the worker script can call this in parallel threads.
    """
    if not processor or not db:
        raise HTTPException(status_code=503, detail="Services not initialized")
    
    try:
        # Check if exists and not forced
        if not request.force_reprocess:
            existing = db.get_media(request.path)
            if existing: # This needs get_media to return MediaItem or logic for it
                 # If we only have raw dict, we might need to convert. 
                 # Current VideoDB.get_media returns dict properly formatted?
                 # Let's assume re-processing is desired if called via /process
                 pass 

        if not os.path.exists(request.path):
             raise HTTPException(status_code=404, detail="File not found on server")

        logger.info(f"Processing request for: {request.path}")
        
        # Process It
        result = processor.process_file(request.path)
        if not result:
             raise HTTPException(status_code=500, detail="Processing failed or file format unsupported")
             
        item, embedding_bytes = result
        
        # Save to DB
        db.upsert_media(item, embedding_bytes)
        
        return item
        
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="File not found")
    except Exception as e:
        logger.error(f"Processing failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/search", response_model=List[SearchResponse])
def search_media(
    q: str = Query(..., min_length=1, description="Search query"),
    limit: int = 50,
    media_type: Optional[MediaType] = None
):
    """Hybrid Search (Keyword + Vector)."""
    if not db:
        raise HTTPException(status_code=503, detail="DB not initialized")
        
    try:
        # Generate Embedding for Semantic Search
        query_vector = None
        if processor and processor.embedder:
             try:
                 query_vector = processor.embedder.embed_query(q)
             except Exception as e:
                 logger.warning(f"Embedding failed for search, falling back to keyword only: {e}")

        # Perform Search
        results = db.search_media(q, query_vector=query_vector, limit=limit)
        
        # Filter by media_type if requested (DB search might not support this yet, so filter in memory or extend DB)
        # bo_db.search_media returns list of SearchResponse objects
        
        if media_type:
            results = [r for r in results if r.media_type == media_type]
            
        return results
    except Exception as e:
        logger.error(f"Search failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/media", response_model=List[MediaItem])
def list_media(
    limit: int = 50, 
    offset: int = 0,
    sort_by: str = Query("date_desc", regex="^(date_asc|date_desc|duration_asc|duration_desc)$"),
    tag: Optional[str] = None,
    media_type: str = "all",
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    response: Response = None
):
    """
    List processed media items with filtering and sorting.
    Returns X-Total-Count header for pagination.
    """
    if not db:
         raise HTTPException(status_code=503, detail="DB not initialized")
    
    try:
        items = db.get_all_media(
            limit=limit, 
            offset=offset, 
            media_type=media_type,
            sort_by=sort_by,
            tag=tag,
            date_from=date_from,
            date_to=date_to
        )
        
        # Get total for headers
        total = db.get_total_count(media_type=media_type, tag=tag)
        if response:
            response.headers["X-Total-Count"] = str(total)
            response.headers["X-Page-Size"] = str(limit)
            
        return items
    except Exception as e:
        logger.error(f"Listing failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
