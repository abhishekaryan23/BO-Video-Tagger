import os
import logging
from dataclasses import dataclass
from logging.handlers import RotatingFileHandler

@dataclass
class ModelConfig:
    filename: str
    mmproj: str
    desc: str
    min_ram_gb: float
    sha256: str
    mmproj_sha256: str

MODEL_TIERS = {
    "smart": ModelConfig(
        filename="SmolVLM2-500M-Video-Instruct-Q8_0.gguf",
        mmproj="mmproj-SmolVLM2-500M-Video-Instruct-Q8_0.gguf",
        desc="Balanced (Q8 Quantization)",
        min_ram_gb=2.5,
        sha256="6f67b8036b2469fcd71728702720c6b51aebd759b78137a8120733b4d66438bc",
        mmproj_sha256="921dc7e259f308e5b027111fa185efcbf33db13f6e35749ddf7f5cdb60ef520b"
    ),
    "super": ModelConfig(
        filename="SmolVLM2-500M-Video-Instruct-f16.gguf",
        mmproj="mmproj-SmolVLM2-500M-Video-Instruct-f16.gguf",
        desc="Max Precision (F16 - Lossless)",
        min_ram_gb=4.0,
        sha256="80f7e3f04bc2d3324ac1a9f52f5776fe13a69912adf74f8e7edacf773d140d77",
        mmproj_sha256="b5dc8ebe7cbeab66a5369693960a52515d7824f13d4063ceca78431f2a6b59b0"
    )
}

@dataclass
class Settings:
    """Central Configuration for BO Video Tagger."""
    
    # =========================================================================
    # 📁 Paths & System
    # =========================================================================
    BASE_DIR: str = os.path.dirname(os.path.abspath(__file__))
    CACHE_DIR: str = os.getenv("BO_CACHE_DIR", os.path.expanduser("~/.cache/bo_video_tagger"))
    
    # Derived Paths
    DB_PATH: str = os.path.join(CACHE_DIR, "library.db")
    THUMBS_DIR: str = os.path.join(CACHE_DIR, "thumbs")
    MODELS_DIR: str = os.path.join(CACHE_DIR, "models")
    LOG_FILE: str = os.path.join(CACHE_DIR, "app.log")
    
    # System Limits
    MAX_LOG_SIZE_BYTES: int = 5 * 1024 * 1024 # 5MB
    LOG_BACKUP_COUNT: int = 3
    ANALYTICS_TTL: int = 60
    
    # Scanning Exclusions
    IGNORE_DIRS = ('.git', 'node_modules', '__pycache__', 'env', 'venv', 'Library', 'AppData', '.Trash')

    # =========================================================================
    # 🤖 AI Model Configuration
    # =========================================================================
    # Vision Language Model (SmolVLM)
    REPO_ID: str = "ggml-org/SmolVLM2-500M-Video-Instruct-GGUF"
    CONTEXT_SIZE: int = 8192
    
    # Transcription (Faster-Whisper)
    WHISPER_MODEL_SIZE: str = "medium"
    
    # Vector Embeddings (SentenceTransformers)
    EMBEDDING_MODEL_NAME: str = "all-MiniLM-L6-v2"

    # =========================================================================
    # 🔍 Search & Retrieval Tuning
    # =========================================================================
    # Hybrid Search Weights (must sum to ~1.0 generally, but relative magnitude matters)
    SEARCH_FTS_WEIGHT: float = 0.3   # Keyword Match Importance
    SEARCH_VEC_WEIGHT: float = 0.7   # Semantic Match Importance
    
    SEARCH_CUTOFF: float = 0.18      # Minimum score to be considered a match

    # =========================================================================
    # 🖥️ UI & Application Defaults
    # =========================================================================
    MAX_CONCURRENT_JOBS: int = 1     # Max parallel heavy processing jobs
    # =========================================================================
    GRID_COLUMNS: int = 5
    model_tier: str = "smart" # Default tier if not specified
    debug: bool = False

    def __post_init__(self):
        # Allow env var overrides
        if os.environ.get("BO_DEBUG"):
            self.debug = os.environ.get("BO_DEBUG").lower() == "true"
        
        if os.environ.get("BO_MODEL_TIER"):
            self.model_tier = os.environ.get("BO_MODEL_TIER")
            
settings = Settings()

def configure_logging():
    """Sets up robust logging with rotation."""
    os.makedirs(Settings.CACHE_DIR, exist_ok=True)
    
    # Root Logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # Prevent duplicate handlers on re-runs
    if logger.handlers:
        return

    # File Handler (Rotating)
    try:
        file_handler = RotatingFileHandler(
            settings.LOG_FILE, 
            maxBytes=settings.MAX_LOG_SIZE_BYTES, 
            backupCount=settings.LOG_BACKUP_COUNT
        )
        file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    except Exception as e:
        print(f"Failed to setup file logging: {e}")

    # Console Handler (Streamlit friendly)
    console_handler = logging.StreamHandler()
    console_formatter = logging.Formatter('%(levelname)s: %(message)s')
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    logging.info("🚀 BO Video Tagger Logging Initialized")
