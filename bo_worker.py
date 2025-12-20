import threading
import queue
import time
import logging
from typing import Optional, Callable
from bo_video_tagger import VideoTagger, MODEL_TIERS
from bo_db import VideoDB

logger = logging.getLogger("VideoWorker")

class WorkerSignals:
    """Standardized signals for queue communication."""
    STATUS = "STATUS"   # Simple text update
    PROGRESS = "PROGRESS" # (current, total)
    RESULT = "RESULT"   # Completed video data
    ERROR = "ERROR"
    DONE = "DONE"

class VideoWorker:
    def __init__(self, db: VideoDB):
        self.db = db
        self.queue = queue.Queue()
        self.thread: Optional[threading.Thread] = None
        self.stop_event = threading.Event()
        self.is_running = False

    def start_processing(self, tier: str, target_dir: str, interval: int, debug: bool):
        """Starts the background processing thread."""
        if self.thread and self.thread.is_alive():
            logger.warning("Worker already running.")
            return

        self.stop_event.clear()
        self.is_running = True
        
        # Start Thread
        self.thread = threading.Thread(
            target=self._run_job,
            args=(tier, target_dir, interval, debug),
            daemon=True
        )
        self.thread.start()

    def stop(self):
        """Signals the thread to stop."""
        self.stop_event.set()
        self.is_running = False
        self.queue.put((WorkerSignals.STATUS, "Stopping..."))

    def _run_job(self, tier: str, target_dir: str, interval: int, debug: bool):
        try:
            self.queue.put((WorkerSignals.STATUS, "Initializing Engine..."))
            
            # 1. Initialize Tagger
            tagger = VideoTagger(tier=tier, debug=debug, interval=interval)
            try:
                tagger.prepare()
            except Exception as e:
                self.queue.put((WorkerSignals.ERROR, f"Engine Init Failed: {e}"))
                return

            # 2. Migration Check
            self.queue.put((WorkerSignals.STATUS, "Checking for legacy data..."))
            try:
                self.db.migrate_jsonl(target_dir)
            except Exception as e:
                logger.error(f"Migration error: {e}")

            # 3. Scan Files
            import os
            video_extensions = ('.mp4', '.mov', '.avi', '.mkv', '.webm')
            files = []
            for root, _, filenames in os.walk(target_dir):
                for f in filenames:
                    if f.lower().endswith(video_extensions) and not f.startswith("._"):
                        files.append(os.path.join(root, f))
            
            total = len(files)
            if total == 0:
                self.queue.put((WorkerSignals.ERROR, "No video files found."))
                return

            self.queue.put((WorkerSignals.STATUS, f"Found {total} videos."))

            # 4. Processing Loop
            for idx, file_path in enumerate(files):
                if self.stop_event.is_set():
                    break
                
                # Check if already processed (skip logic could be smarter, but re-processing is sometimes desired)
                # For now, we process everything. Real app might check DB first.
                
                self.queue.put((WorkerSignals.PROGRESS, (idx + 1, total, os.path.basename(file_path))))
                
                try:
                    # Modified VideoTagger.process_video now returns data dict
                    result = tagger.process_video(file_path)
                    
                    if "error" in result:
                         logger.error(f"Failed {file_path}: {result['error']}")
                    else:
                        # Save to DB
                        self.db.upsert_video(result)
                        
                except Exception as e:
                    logger.error(f"Crash on {file_path}: {e}", exc_info=True)
                
            self.queue.put((WorkerSignals.DONE, "Processing Complete"))

        except Exception as e:
            logger.exception(f"Worker Crash: {e}")
            self.queue.put((WorkerSignals.ERROR, f"Worker Crash: {e}"))
        finally:
            self.is_running = False
