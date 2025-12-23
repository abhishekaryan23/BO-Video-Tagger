import os
import argparse
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List

from bo_db import VideoDB
from processor import MediaProcessor
from schemas import MediaType
from bo_config import settings

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("Worker")

def process_single_file(processor: MediaProcessor, db: VideoDB, file_path: str, force: bool = False):
    """Processes a single file and saves to DB."""
    try:
        # Check if already processed
        if not force:
            # We need a robust check. verify_path_exists logic in DB or just get_media
            # For now, simplistic check
            existing = db.get_media(file_path)
            if existing: # Returns dict or None
                logger.debug(f"Skipping {file_path} (already processed).")
                return
        
        logger.info(f"Processing: {file_path}")
        start_time = time.time()
        
        item, embedding_bytes = processor.process_file(file_path)
        
        # Save
        db.upsert_media(item, embedding_bytes)
        
        duration = time.time() - start_time
        logger.info(f"Done: {file_path} ({duration:.2f}s)")
        
    except KeyboardInterrupt:
        raise
    except BaseException as e:
        logger.error(f"CRITICAL WORKER FAILURE on {file_path}: {e}", exc_info=True)

def main():
    parser = argparse.ArgumentParser(description="BO-Video-Tagger Batch Worker")
    parser.add_argument("--dir", type=str, required=True, help="Directory to scan")
    parser.add_argument("--workers", type=int, default=1, help="Number of concurrent workers (threads)")
    parser.add_argument("--force", action="store_true", help="Reprocess all files")
    parser.add_argument("--tier", type=str, default="smart", help="Model tier")
    parser.add_argument("--ext", nargs="+", default=["mp4", "mov", "mkv", "avi", "jpg", "jpeg", "png", "webp"], help="Extensions to process")
    
    args = parser.parse_args()
    
    # Init DB
    db = VideoDB()
    
    # Init Processor (Global for now, or per worker if needed, but models are memory heavy)
    # If we use ThreadPool, we share the processor. 
    # Python threads are GIL-limited, but I/O (GPU/Disk) releases GIL.
    # For VLM/Whisper, they might be CPU heavy or Lock GPU. 
    # Multiple threads sharing one MPS model might fail or be slow.
    # Safest is 1 worker per process. Scaling via Docker containers is preferred.
    if args.workers > 1:
        logger.warning("Using >1 workers. Ensure your hardware (GPU/RAM) can handle concurrent model inference.")
    
    processor = MediaProcessor(mode=args.tier)
    
    # Scan Files
    files_to_process = []
    logger.info(f"Scanning {args.dir}...")
    
    # Get all processed paths to skip quickly
    # This optimization requires a DB method. 
    # existing_paths = db.get_all_file_paths_set() # Assuming this method exists
    # For correctness, let's rely on process_single_file check or implement batch check
    
    exts = tuple(f".{e.lower()}" for e in args.ext)
    
    for root, dirs, files in os.walk(args.dir):
        for file in files:
            if file.lower().endswith(exts):
                full_path = os.path.join(root, file)
                files_to_process.append(full_path)
                
    logger.info(f"Found {len(files_to_process)} candidate files.")
    
    # Processing Loop
    if args.workers == 1:
        for f in files_to_process:
            process_single_file(processor, db, f, args.force)
    else:
        # Threaded
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = [executor.submit(process_single_file, processor, db, f, args.force) for f in files_to_process]
            for future in as_completed(futures):
                pass # Exceptions handled in function
                
    logger.info("Batch processing complete.")
    db.close()

if __name__ == "__main__":
    main()
