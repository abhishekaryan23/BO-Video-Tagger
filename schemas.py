from enum import Enum
from typing import List, Optional, Dict, Any, Union
from pydantic import BaseModel, Field, ConfigDict
from datetime import datetime

class MediaType(str, Enum):
    VIDEO = "video"
    IMAGE = "image"
    UNKNOWN = "unknown"

class MediaMetadata(BaseModel):
    """File-level metadata."""
    filename: str
    path: str
    size_mb: float
    parent_folder: str
    created_at: Optional[datetime] = None
    
    # Video Specific
    duration_sec: Optional[float] = None
    resolution: Optional[str] = None
    fps: Optional[float] = None
    frame_count: Optional[int] = None
    
    # Image Specific
    width: Optional[int] = None
    height: Optional[int] = None
    format: Optional[str] = None

class AIResponse(BaseModel):
    """Output from the Visual Language Model (VLM)."""
    description: str = Field(..., description="Detailed description of the media content")
    summary: str = Field(..., description="Short summary for quick display")
    tags: List[str] = Field(default_factory=list, description="Extracted keywords/metrics")
    
class TranscriptionSegment(BaseModel):
    """A single segment of transcribed audio."""
    start: float
    end: float
    text: str

class TranscriptionData(BaseModel):
    """Full transcription result."""
    full_text: str
    segments: List[TranscriptionSegment] = []
    language: str = Field(default="en", description="Detected language code (e.g., 'en', 'fr')")
    language_probability: float = Field(default=0.0, description="Confidence score of language detection")
    
class SystemInfo(BaseModel):
    """Processing diagnostics."""
    model_name: str
    timestamp: datetime = Field(default_factory=datetime.now)
    processing_time_sec: float = 0.0

class MediaItem(BaseModel):
    """Main Data Model for either Video or Image."""
    model_config = ConfigDict(from_attributes=True) # Allows creation from ORM objects if needed

    media_type: MediaType = MediaType.UNKNOWN
    meta: MediaMetadata
    ai: Optional[AIResponse] = None
    transcription: Optional[TranscriptionData] = None
    system: Optional[SystemInfo] = None
    
    # Vector Embedding (Stored as bytes/blob in DB, kept as list[float] or None here)
    # Exclude from default dict export to save space unless requested?
    # For simplicity, we keep it out of the main model or make it optional.
    # embedding_shape: Optional[Tuple[int]] = None 
    
class ProcessRequest(BaseModel):
    """API Request Model."""
    path: str
    force_reprocess: bool = False

class SearchResponse(BaseModel):
    """Search Result Item."""
    filename: str
    path: str
    score: float
    media_type: MediaType
    summary: str
    description: Optional[str] = None # Full text without truncation
    tags: List[str]
    thumbnail_path: Optional[str] = None

class ErrorResponse(BaseModel):
    """RFC 7807 Problem Details."""
    type: str = Field(default="about:blank", description="URI reference identifying the problem type")
    title: str = Field(..., description="Short summary of the problem")
    status: int = Field(..., description="HTTP status code")
    detail: str = Field(..., description="Specific explanation of this occurrence")
    instance: Optional[str] = Field(None, description="URI reference to the specific occurrence")
