import pytest
from schemas import MediaItem, MediaType, MediaMetadata, AIResponse, TranscriptionData

def test_media_type_defaults():
    item = MediaItem(
        meta=MediaMetadata(
            filename="test.mp4", 
            path="/tmp/test.mp4", 
            size_mb=10.0, 
            parent_folder="tmp"
        )
    )
    assert item.media_type == MediaType.UNKNOWN

def test_transcription_defaults():
    t = TranscriptionData(full_text="Hello world")
    assert t.language == "en"
    assert t.language_probability == 0.0

def test_serialization():
    item = MediaItem(
        media_type=MediaType.VIDEO,
        meta=MediaMetadata(
            filename="video.mp4", 
            path="/v.mp4", 
            size_mb=1.0, 
            parent_folder="root",
            duration_sec=10.0
        ),
        ai=AIResponse(description="desc", summary="sum", tags=["a", "b"])
    )
    
    json_str = item.model_dump_json()
    assert "video" in json_str
    assert "desc" in json_str
    
    item2 = MediaItem.model_validate_json(json_str)
    assert item2.meta.duration_sec == 10.0
