import logging
import numpy as np
from sentence_transformers import SentenceTransformer
from schemas import MediaItem, MediaType
from bo_config import settings

logger = logging.getLogger("Embedder")

class EmbeddingService:
    def __init__(self, model_name: str = settings.EMBEDDING_MODEL_NAME):
        self.model_name = model_name
        self._model = None
        self.device = "cpu" # SentenceTransformers defaults effectively. MPS on Mac is good if available.
        # We let the library auto-detect best device usually, but can enforce.
        
    @property
    def model(self):
        """Lazy load model."""
        if self._model is None:
            logger.info(f"🧠 Loading Embedding Model ({self.model_name})...")
            try:
                self._model = SentenceTransformer(self.model_name)
                logger.info("✅ Embedding Model Loaded.")
            except Exception as e:
                logger.error(f"Failed to load Embedding Model: {e}")
                raise e
        return self._model

    def generate_context_string(self, item: MediaItem) -> str:
        """Creates the rich text representation for embedding."""
        parts = []
        
        # 1. Title/Filename (High weight conceptually, usually comes first)
        parts.append(f"Title: {item.meta.filename}")
        
        # 2. Type
        parts.append(f"Type: {item.media_type.value}")
        
        # 3. Tags
        if item.ai and item.ai.tags:
            parts.append(f"Tags: {', '.join(item.ai.tags)}")
            
        # 4. Visual Description
        if item.ai and item.ai.description:
            parts.append(f"Visual Description: {item.ai.description}")
            
        # 5. Transcription (Audio Content)
        if item.transcription and item.transcription.full_text:
            # Truncate very long transcripts to fit context window (usually 256/512 tokens)
            # MiniLM is 256 up to 512. We take first ~1000 chars roughly.
            # Ideally we chunk, but for single-vector representation, early text is key.
            transcript_snippet = item.transcription.full_text[:1000] 
            parts.append(f"Audio Transcript: {transcript_snippet}")
            
        return "\n".join(parts)

    def generate_embedding(self, item: MediaItem) -> bytes:
        """New embedding generation returning Bytes for DB storage."""
        text_context = self.generate_context_string(item)
        
        try:
            # Generate Metadata
            # encode() returns a numpy array
            vector = self.model.encode(text_context, convert_to_numpy=True, normalize_embeddings=True)
            
            # Convert to float32 bytes for storage
            return vector.astype(np.float32).tobytes()
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            return b""
            
    def embed_query(self, query: str) -> np.ndarray:
        """Embeds a search query for vector comparison."""
        return self.model.encode(query, convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)
