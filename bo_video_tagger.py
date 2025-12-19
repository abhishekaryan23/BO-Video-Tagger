#!/usr/bin/env python3
import os
__version__ = "2.0.0"
import sys
import json
import time
import argparse
import logging
import psutil
import cv2
import re
import textwrap
import base64
import numpy as np
import yake
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from huggingface_hub import hf_hub_download
from llama_cpp import Llama
from llama_cpp.llama_chat_format import Llava15ChatHandler
from tqdm import tqdm

# --- Configuration Constants ---
REPO_ID = "ggml-org/SmolVLM2-500M-Video-Instruct-GGUF"
CONTEXT_SIZE = 8192
DEFAULT_DEBUG_DIR = "debug_frames"

# Setup Logger (Configured in main)
logger = logging.getLogger("VideoTagger")

@dataclass
class ModelConfig:
    filename: str
    mmproj: str
    desc: str
    min_ram_gb: float

MODEL_TIERS = {
    "smart": ModelConfig(
        filename="SmolVLM2-500M-Video-Instruct-Q8_0.gguf",
        mmproj="mmproj-SmolVLM2-500M-Video-Instruct-Q8_0.gguf",
        desc="Balanced (Q8 Quantization)",
        min_ram_gb=2.5
    ),
    "super": ModelConfig(
        filename="SmolVLM2-500M-Video-Instruct-f16.gguf",
        mmproj="mmproj-SmolVLM2-500M-Video-Instruct-f16.gguf",
        desc="Max Precision (F16 - Lossless)",
        min_ram_gb=4.0
    )
}

class VideoTagger:
    def __init__(self, tier: str = "smart", debug: bool = False, interval: int = 10):
        self.interval = interval
        self.debug = debug
        self.tier_name = tier
        
        if tier not in MODEL_TIERS:
            raise ValueError(f"Invalid tier: {tier}. Choices: {list(MODEL_TIERS.keys())}")
            
        self.config = MODEL_TIERS[tier]
        self.model_dir = os.path.join(os.getcwd(), "models")
        self.debug_dir = os.path.join(os.getcwd(), DEFAULT_DEBUG_DIR)
        
        self.llm: Optional[Llama] = None
        self.model_path = os.path.join(self.model_dir, self.config.filename)
        self.mmproj_path = os.path.join(self.model_dir, self.config.mmproj)

    def prepare(self):
        """Downloads models and initializes the inference engine."""
        self._setup_directories()
        self._download_models()
        self._load_engine()

    def _setup_directories(self):
        os.makedirs(self.model_dir, exist_ok=True)
        if self.debug:
            os.makedirs(self.debug_dir, exist_ok=True)
            logger.info(f"🐛 Debug mode ON. Frames: {self.debug_dir}")

    def _download_models(self):
        """Ensures both model and projector exist."""
        if os.path.exists(self.model_path) and os.path.exists(self.mmproj_path):
            return

        logger.info(f"⬇️  Downloading {self.tier_name.upper()} model files from {REPO_ID}...")
        try:
            for fname in [self.config.filename, self.config.mmproj]:
                hf_hub_download(
                    repo_id=REPO_ID,
                    filename=fname,
                    local_dir=self.model_dir,
                    local_dir_use_symlinks=False
                )
            logger.info("✅ Download complete.")
        except Exception as e:
            logger.exception("Failed to download model files.")
            sys.exit(1)

    def _load_engine(self):
        """Loads Llama with Vision Handler."""
        logger.info(f"🤖 Loading {self.tier_name.upper()} Engine (Vision Enabled)...")
        try:
            chat_handler = Llava15ChatHandler(clip_model_path=self.mmproj_path)
            self.llm = Llama(
                model_path=self.model_path,
                chat_handler=chat_handler,
                n_ctx=CONTEXT_SIZE,
                n_gpu_layers=-1, # Auto-offload
                verbose=False
            )
        except Exception as e:
            logger.exception("Failed to initialize Inference Engine.")
            sys.exit(1)

    def extract_frames(self, video_path: str, max_frames: int = 5) -> tuple[List[str], Dict[str, Any]]:
        cap = cv2.VideoCapture(video_path)
        metadata = {"duration_sec": 0, "resolution": "unknown", "fps": 0, "frame_count": 0}
        
        if not cap.isOpened():
            logger.warning(f"Could not open video: {video_path}")
            return [], metadata

        try:
            fps = cap.get(cv2.CAP_PROP_FPS)
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            metadata["fps"] = round(fps, 2)
            metadata["resolution"] = f"{w}x{h}"
            metadata["frame_count"] = total_frames
            metadata["duration_sec"] = round(total_frames / fps, 2) if fps > 0 else 0
        except Exception as e:
            logger.warning(f"Metadata extraction warning: {e}")

        # Fallback to 30fps if metadata is broken
        frame_interval = int(fps * self.interval) if fps > 0 else 300 
        
        base64_frames = []
        count = 0
        extracted_count = 0
        
        while cap.isOpened() and len(base64_frames) < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            if count % frame_interval == 0:
                # Validation: Skip empty/black frames
                if np.var(frame) < 10:
                    count += 1
                    continue

                resized = cv2.resize(frame, (384, 384))
                
                if self.debug:
                    debug_name = f"{os.path.basename(video_path)}_{extracted_count}.jpg"
                    cv2.imwrite(os.path.join(self.debug_dir, debug_name), resized)

                _, buffer = cv2.imencode('.jpg', resized, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                b64_str = base64.b64encode(buffer).decode('utf-8')
                base64_frames.append(f"data:image/jpeg;base64,{b64_str}")
                extracted_count += 1
            
            count += 1
            
        cap.release()
        return base64_frames, metadata

    def _parse_ai_response(self, text: str) -> Dict[str, Any]:
        """Robustly parses the AI output using YAKE-First Strategy."""
        
        # 1. Clean Chatty Prefixes ("Sure, here is...", "Assistant:")
        # Handles optional colon and varying casing
        clean_text = re.sub(r'(?i)^(here is|sure|okay|assistant|ai):?\s*', '', text.strip())
        
        # 2. Generate Professional Summary (Smart Truncation)
        # Uses textwrap to avoid breaking words mid-string
        summary = textwrap.shorten(clean_text, width=150, placeholder="...")
        
        # 3. YAKE Extraction (Primary Tagging Source)
        tags = self._extract_yake_tags(clean_text)
        
        return {
            "summary": summary,
            "description": clean_text,
            "tags": tags
        }

    def _extract_yake_tags(self, text: str) -> List[str]:
        """Extracts significant keywords using YAKE (Unsupervised Statistical Learning)."""
        try:
            # Init YAKE: English, max n-gram=2 (e.g. "Workload Domain")
            # dedupLim=0.3 -> STRICT deduplication to avoid "Video" vs "Video Tutorial"
            kw_extractor = yake.KeywordExtractor(lan="en", n=2, dedupLim=0.4, top=5, features=None)
            keywords = kw_extractor.extract_keywords(text)
            
            # YAKE returns [(kw, score), ...] - lower score is better
            tags = [kw for kw, score in keywords]
            return tags if tags else ["untagged"]
        except Exception as e:
            logger.warning(f"YAKE extraction failed: {e}")
            return ["untagged", "error"]

    def process_video(self, video_path: str) -> Dict[str, Any]:
        """Runs the VLM on a single video file."""
        if not self.llm:
            raise RuntimeError("Engine not loaded. Call prepare() first.")

        start_time = time.time()
        try:
            frames, vid_meta = self.extract_frames(video_path)
            if not frames:
                return {"meta": {"file": os.path.basename(video_path)}, "error": "No valid frames extracted"}

            content = [{"type": "image_url", "image_url": {"url": img}} for img in frames]
            
            # Simplified Prompt (Focused on Description)
            prompt_text = (
                "Describe the content of this video in detail. "
                "Focus on technical terms, objects present in the scene, and actions being performed."
            )
            
            content.append({
                "type": "text", 
                "text": prompt_text
            })

            response = self.llm.create_chat_completion(
                messages=[{"role": "user", "content": content}],
                max_tokens=512, # Increased for detailed desc + tags
                temperature=0.5, # Lower temp for strict formatting
                repeat_penalty=1.1
            )
            
            raw_text = response['choices'][0]['message']['content']
            parsed_ai = self._parse_ai_response(raw_text)
            
            # Construct Rich Dictionary
            return {
                "meta": {
                    "file": os.path.basename(video_path),
                    "path": video_path,
                    "size_mb": round(os.path.getsize(video_path) / (1024*1024), 2),
                    **vid_meta
                },
                "ai": parsed_ai,
                "system": {
                    "model": self.config.filename,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "processing_time_sec": round(time.time() - start_time, 2)
                }
            }

        except Exception as e:
            logger.exception(f"Error processing {video_path}")
            return {"meta": {"file": os.path.basename(video_path)}, "error": str(e)}

def check_system_resources(tier: str) -> bool:
    mem = psutil.virtual_memory()
    available_gb = mem.available / (1024 ** 3)
    req_gb = MODEL_TIERS[tier].min_ram_gb
    
    logger.info(f"System Check ({tier.upper()}): {available_gb:.2f}GB Available / {req_gb}GB Required")
    
    if available_gb < req_gb:
        print(f"⚠️  WARNING: Low Memory. Required: {req_gb}GB, Available: {available_gb:.2f}GB")
        if sys.stdin.isatty():
             return input("Continue anyway? (y/n): ").strip().lower() == 'y'
        return False
    return True

def main():
    # Configure Logging
    logging.basicConfig(
        level=logging.INFO, 
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )

    parser = argparse.ArgumentParser(description=f"BO Video Tagger v{__version__}")
    parser.add_argument("folder", help="Path to video folder")
    parser.add_argument("--mode", choices=MODEL_TIERS.keys(), default="smart", help="Processing mode")
    parser.add_argument("--interval", type=int, default=10, help="Frame extraction interval in seconds (default: 10)")
    parser.add_argument("--output", help="Custom output directory or filename")
    parser.add_argument("--debug", action="store_true", help="Save debug frames")
    args = parser.parse_args()

    # Early Validation
    if not os.path.exists(args.folder):
        logger.error(f"Directory not found: {args.folder}")
        sys.exit(1)

    if not check_system_resources(args.mode):
        logger.error("Aborted due to system requirements.")
        sys.exit(1)

    # Initialize Tagger
    tagger = VideoTagger(tier=args.mode, debug=args.debug, interval=args.interval)
    tagger.prepare()

    # Find Videos
    valid_exts = ('.mp4', '.mov', '.avi', '.mkv', '.webm')
    video_files = [
        os.path.join(args.folder, f) for f in os.listdir(args.folder) 
        if f.lower().endswith(valid_exts)
    ]
    
    if not video_files:
        logger.warning("No video files found.")
        sys.exit(0)

    logger.info(f"Processing {len(video_files)} videos...")

    # Determine Output Path
    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    folder_name = os.path.basename(os.path.normpath(args.folder))
    # User Request: "{Folder}_video_tags_{Date}.jsonl"
    default_filename = f"{folder_name}_video_tags_{timestamp}.jsonl"

    if args.output:
        if os.path.isdir(args.output):
             # User gave a directory: /path/to/save/ -> /path/to/save/Folder_video_tags_Date.jsonl
            output_path = os.path.join(args.output, default_filename)
        else:
            # User gave a specific file: /path/to/my_file.jsonl
            output_path = args.output
    else:
        # Default: Save in CWD with dynamic name
        output_path = default_filename

    logger.info(f"💾 Saving results to: {output_path}")
    
    # Process Loop
    with tqdm(total=len(video_files), unit="vid") as pbar:
        # Open in append mode (Line-Delimited JSON)
        with open(output_path, 'a') as f:
            for vid in video_files:
                result = tagger.process_video(vid)
                
                # O(1) Write
                f.write(json.dumps(result) + "\n")
                f.flush() # Ensure it hits disk immediately
                
                pbar.set_postfix(file=os.path.basename(vid)[:10])
                pbar.update(1)

    logger.info(f"Done! Results saved to {output_path}")

if __name__ == "__main__":
    main()