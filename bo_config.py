import os
import logging
from dataclasses import dataclass
from logging.handlers import RotatingFileHandler

@dataclass
class Settings:
    """Central Configuration for BO-View."""
    # Paths
    BASE_DIR: str = os.path.dirname(os.path.abspath(__file__))
    CACHE_DIR: str = os.path.expanduser("~/.cache/bo_video_tagger")
    DB_PATH: str = os.path.join(CACHE_DIR, "library.db")
    THUMBS_DIR: str = os.path.join(CACHE_DIR, "thumbs")
    MODELS_DIR: str = os.path.join(CACHE_DIR, "models")
    LOG_FILE: str = os.path.join(CACHE_DIR, "app.log")

    # UI Settings
    GRID_COLUMNS: int = 5
    
    # System
    MAX_LOG_SIZE_BYTES: int = 5 * 1024 * 1024 # 5MB
    LOG_BACKUP_COUNT: int = 3

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
            Settings.LOG_FILE, 
            maxBytes=Settings.MAX_LOG_SIZE_BYTES, 
            backupCount=Settings.LOG_BACKUP_COUNT
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

    logging.info("🚀 BO-View Logging Initialized")
