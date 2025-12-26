import os
import cv2
import time
import logging
import base64
import numpy as np
from typing import List, Optional, Dict
from datetime import datetime
from PIL import Image

# Import Schemas & Models
from schemas import (
    MediaItem, MediaType, MediaMetadata, AIResponse, 
    TranscriptionData, SystemInfo
)

# Import Services
from services.transcription import TranscriptionService
from services.embedding import EmbeddingService
from bo_config import settings

# VLM Imports (Keeping your original VLM logic but refactored)
from llama_cpp import Llama
from llama_cpp.llama_chat_format import Llava15ChatHandler
from huggingface_hub import hf_hub_download
import textwrap
import re
import yake

logger = logging.getLogger("Processor")

# --- Custom Exceptions ---
class ProcessingError(Exception):
    """Raised when media processing fails with a known cause."""
    pass

# --- Configuration Constants (Can be moved to config) ---
class MediaProcessor:
    def __init__(self, mode: str = "smart", device: str = None):
        self.mode = mode
        self.model_dir = settings.MODELS_DIR
        
        # Initialize Services (Lazy Loading built-in)
        self.transcriber = TranscriptionService(device=device)
        self.embedder = EmbeddingService()
        
        # VLM State
        self.llm = None
        self.mmproj_path = None
        self.model_path = None
        
        self._setup_vlm()

    def _setup_vlm(self):
        """Prepare VLM paths/configs."""
        from bo_config import MODEL_TIERS
        
        if self.mode not in MODEL_TIERS:
             logger.warning(f"Unknown tier {self.mode}, defaulting to smart")
             self.mode = "smart"
             
        self.config = MODEL_TIERS[self.mode]
        
        os.makedirs(self.model_dir, exist_ok=True)
        self.model_path = os.path.join(self.model_dir, self.config.filename)
        self.mmproj_path = os.path.join(self.model_dir, self.config.mmproj)
        
        # Auto-Download if missing
        for fname in [self.config.filename, self.config.mmproj]:
            if not os.path.exists(os.path.join(self.model_dir, fname)):
                logger.info(f"⬇️ Downloading {fname}...")
                hf_hub_download(repo_id=settings.REPO_ID, filename=fname, local_dir=self.model_dir, local_dir_use_symlinks=False)

    def load_vlm(self):
        """Loads the Vision Language Model."""
        if self.llm: return
        
        logger.info("🤖 Loading Visual Intelligence Engine...")
        try:
            chat_handler = Llava15ChatHandler(clip_model_path=self.mmproj_path)
            self.llm = Llama(
                model_path=self.model_path,
                chat_handler=chat_handler,
                n_ctx=settings.CONTEXT_SIZE,
                n_gpu_layers=-1, 
                verbose=False
            )
        except Exception as e:
            logger.error(f"Failed to load VLM: {e}")
            raise e

    def process_file(self, path: str) -> Optional[tuple[MediaItem, bytes]]:
        """
        Main Pipeline Entrypoint.
        Returns: (MediaItem object, embedding_bytes)
        Raises: ProcessingError, FileNotFoundError
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}")
            
        start_time = time.time()
        
        try:
            self.load_vlm() # Ensure loaded
        except Exception as e:
            raise ProcessingError(f"Failed to load AI Model: {e}")
        
        # 1. Determine Type
        ext = os.path.splitext(path)[1].lower()
        if ext in ['.jpg', '.jpeg', '.png', '.webp', '.bmp']:
            media_type = MediaType.IMAGE
        elif ext in ['.mp4', '.mov', '.avi', '.mkv', '.webm']:
            media_type = MediaType.VIDEO
        else:
            logger.warning(f"Unsupported media type: {path}")
            raise ProcessingError(f"Unsupported file format: {ext}")

        logger.info(f"Processing {media_type.value.upper()}: {os.path.basename(path)}")

        try:
            # 2. Extract Metadata & Frames/Image
            if media_type == MediaType.VIDEO:
                try:
                    frames, meta = self._extract_video_frames(path)
                    transcription = self.transcriber.transcribe(path)
                except Exception as e:
                    raise ProcessingError(f"Video Extraction Failed: {e}")
            else:
                try:
                    frames, meta = self._process_image(path)
                    transcription = None # No audio for images usually
                except Exception as e:
                    raise ProcessingError(f"Image Processing Failed: {e}")
            
            # 3. Visual Tagging (VLM)
            try:
                ai_response = self._run_vlm_inference(frames, media_type)
            except Exception as e:
                raise ProcessingError(f"AI Inference Failed: {e}")
            
            # 4. Construct Object
            item = MediaItem(
                media_type=media_type,
                meta=meta,
                ai=ai_response,
                transcription=transcription,
                system=SystemInfo(
                    model_name="SmolVLM + Whisper",
                    processing_time_sec=time.time() - start_time
                )
            )
            
            # 5. Generate Embedding
            embedding_bytes = self.embedder.generate_embedding(item)
            
            return item, embedding_bytes

        except ProcessingError:
             raise
        except Exception as e:
            logger.error(f"Critical error processing {path}: {e}")
            raise ProcessingError(f"Critical Failure: {str(e)}")

    def _extract_video_frames(self, path: str) -> tuple[List[str], MediaMetadata]:
        """Extracts frames for VLM and basic metadata."""
        cap = cv2.VideoCapture(path)
        
        try:
            fps = cap.get(cv2.CAP_PROP_FPS)
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = total_frames / fps if fps > 0 else 0
            
            meta = MediaMetadata(
                filename=os.path.basename(path),
                path=path,
                size_mb=round(os.path.getsize(path) / (1024*1024), 2),
                parent_folder=os.path.basename(os.path.dirname(path)),
                created_at=datetime.fromtimestamp(os.path.getctime(path)),
                duration_sec=round(duration, 2),
                fps=round(fps, 2),
                resolution=f"{w}x{h}",
                frame_count=total_frames
            )
            
            # Frame Extraction Logic (Simplified from original)
            base64_frames = []
            max_frames = 5
            interval = int(total_frames / max_frames) if total_frames > max_frames else 1
            
            count = 0
            extracted = 0
            while cap.isOpened() and extracted < max_frames:
                ret, frame = cap.read()
                if not ret: break
                
                if count % interval == 0:
                    resized = cv2.resize(frame, (384, 384))
                    _, buffer = cv2.imencode('.jpg', resized)
                    b64 = base64.b64encode(buffer).decode('utf-8')
                    base64_frames.append(f"data:image/jpeg;base64,{b64}")
                    extracted += 1
                count += 1
        finally:
            cap.release()
            
        return base64_frames, meta

    def _process_image(self, path: str) -> tuple[List[str], MediaMetadata]:
        """Processes a single image."""
        img = Image.open(path)
        w, h = img.size
        
        meta = MediaMetadata(
            filename=os.path.basename(path),
            path=path,
            size_mb=round(os.path.getsize(path) / (1024*1024), 2),
            parent_folder=os.path.basename(os.path.dirname(path)),
            created_at=datetime.fromtimestamp(os.path.getctime(path)),
            width=w,
            height=h,
            format=img.format
        )
        
        # Optimize for VLM (Resize Max 1024)
        img.thumbnail((1024, 1024))
        
        # Convert to Base64
        import io
        buffered = io.BytesIO()
        img.save(buffered, format="JPEG")
        b64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        
        return [f"data:image/jpeg;base64,{b64}"], meta

    def _run_vlm_inference(self, frames: List[str], media_type: MediaType) -> AIResponse:
        """Runs the Llama/Llava model."""
        content = [{"type": "image_url", "image_url": {"url": f}} for f in frames]
        
        if media_type == MediaType.VIDEO:
            prompt = "Describe the video content in detail. Focus on actions, scene context, and key objects."
        else:
            prompt = "Describe this image in detail. Focus on objects, setting, colors, and text."
            
        content.append({"type": "text", "text": prompt})
        
        response = self.llm.create_chat_completion(
            messages=[{"role": "user", "content": content}],
            max_tokens=512,
            temperature=0.4
        )
        
        raw_text = response['choices'][0]['message']['content']
        return self._parse_ai_response(raw_text)

    def _parse_ai_response(self, text: str) -> AIResponse:
        """Parses raw text into AIResponse with YAKE tags."""
        clean_text = re.sub(r'(?i)^(here is|sure|okay|assistant|ai):?\s*', '', text.strip())
        summary = textwrap.shorten(clean_text, width=150, placeholder="...")
        
        # YAKE Extraction
        kw_extractor = yake.KeywordExtractor(lan="en", n=2, dedupLim=0.4, top=5)
        keywords = kw_extractor.extract_keywords(clean_text)
        tags = [kw for kw, score in keywords]
        
        return AIResponse(
            description=clean_text,
            summary=summary,
            tags=tags
        )
