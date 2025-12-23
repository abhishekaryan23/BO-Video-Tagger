import pytest
from unittest.mock import MagicMock, patch
import numpy as np
import os
from services.transcription import TranscriptionService
from services.embedding import EmbeddingService
from schemas import MediaItem, MediaType, MediaMetadata, AIResponse, TranscriptionData

# --- Transcription Tests ---
@pytest.fixture
def mock_whisper():
    with patch("services.transcription.WhisperModel") as MockClass:
        mock_instance = MockClass.return_value
        yield mock_instance

def test_transcription_service_lazy_load(mock_whisper):
    service = TranscriptionService(device="cpu")
    # Should not have called constructor yet
    assert service._model is None
    
    # Access property
    _ = service.model
    assert service._model is not None

def test_transcription_flow(mock_whisper):
    service = TranscriptionService()
    
    # Mock result
    Segment = MagicMock()
    Segment.start = 0.0
    Segment.end = 1.0
    Segment.text = " Hello "
    
    Info = MagicMock()
    Info.language = "fr"
    Info.language_probability = 0.99
    
    service.model.transcribe.return_value = ([Segment], Info)
    
    # Needs a real file check pass, mock os.path.exists
    with patch("os.path.exists", return_value=True):
        data = service.transcribe("fake.wav")
        
    assert data.full_text == "Hello"
    assert data.language == "fr"
    assert data.language_probability == 0.99

# --- Embedding Tests ---
@pytest.fixture
def mock_sbert():
    with patch("services.embedding.SentenceTransformer") as MockClass:
        mock_instance = MockClass.return_value
        yield mock_instance

def test_embedding_context_generation():
    service = EmbeddingService()
    item = MediaItem(
        media_type=MediaType.VIDEO,
        meta=MediaMetadata(filename="movie.mp4", path="/p", size_mb=1, parent_folder="p"),
        ai=AIResponse(description="A scene", summary="Short", tags=["action"]),
        transcription=TranscriptionData(full_text="Dialog here")
    )
    
    ctx = service.generate_context_string(item)
    assert "Title: movie.mp4" in ctx
    assert "Type: video" in ctx
    assert "Visual Description: A scene" in ctx
    assert "Audio Transcript: Dialog here" in ctx

def test_embedding_generation(mock_sbert):
    service = EmbeddingService()
    
    # Mock encode return (numpy array)
    fake_vec = np.array([0.1, 0.2], dtype=np.float32)
    service.model.encode.return_value = fake_vec
    
    item = MediaItem(
        media_type=MediaType.IMAGE,
        meta=MediaMetadata(filename="img.jpg", path="/p", size_mb=1, parent_folder="p")
    )
    
    blob = service.generate_embedding(item)
    assert blob == fake_vec.tobytes()
