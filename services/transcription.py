import logging
import os
import torch
from faster_whisper import WhisperModel
from schemas import TranscriptionData, TranscriptionSegment
from bo_config import settings

logger = logging.getLogger("Transcriber")

class TranscriptionService:
    def __init__(self, model_size: str = settings.WHISPER_MODEL_SIZE, device: str = None):
        self.model_size = model_size
        self._model = None
        
        if device:
            self.device = device
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            if self.device == "cpu" and torch.backends.mps.is_available():
                self.device = "mps" # Faster-Whisper has partial MPS support, usually CPU is safer fallback for now unless specifically optimized
                # Actually faster-whisper uses CTranslate2. CTranslate2 DOES support CoreML/MPS but it's tricky. 
                # For stability on Mac, we often stick to 'cpu' with int8 quantization which is very fast on M-chips, 
                # or use 'auto' if generic.
                # Let's use "cpu" for Mac by default to ensure stability, as MPS support in CTrans2 is experimental.
                self.device = "cpu" 
        
        self.compute_type = "float16" if self.device == "cuda" else "int8"
        logger.info(f"🎙️ Transcription Service Config: {self.device} ({self.compute_type})")

    @property
    def model(self):
        """Lazy load the model."""
        if self._model is None:
            logger.info(f"⬇️ Loading Whisper Model ({self.model_size})...")
            try:
                self._model = WhisperModel(
                    self.model_size, 
                    device=self.device, 
                    compute_type=self.compute_type
                )
                logger.info("✅ Whisper Model Loaded.")
            except Exception as e:
                logger.error(f"Failed to load Whisper: {e}")
                raise e
        return self._model

    def transcribe(self, audio_path: str) -> TranscriptionData:
        """Transcribes audio file to text."""
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        try:
            logger.info(f"🎤 Transcribing {os.path.basename(audio_path)}...")
            segments_generator, info = self.model.transcribe(
                audio_path, 
                beam_size=5,
                word_timestamps=False
            )

            # Consume generator
            segments = list(segments_generator)
            
            # Convert to Schema
            out_segments = [
                TranscriptionSegment(start=s.start, end=s.end, text=s.text.strip())
                for s in segments
            ]
            
            full_text = " ".join([s.text for s in out_segments])
            
            return TranscriptionData(
                full_text=full_text,
                segments=out_segments,
                language=info.language,
                language_probability=info.language_probability
            )

        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            return TranscriptionData(full_text="", segments=[], language="error", language_probability=0.0)
