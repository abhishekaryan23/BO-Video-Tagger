import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from bo_db import VideoDB
import json
import logging

# Suppress Logs for clean output
logging.getLogger("VideoDB").setLevel(logging.ERROR)

def main():
    try:
        db = VideoDB()
        items = db.get_all_media(limit=10)
        
        print(f"\n📊 Processed Media Items: {len(items)}")
        
        for item in items:
            print(f"\n{'='*40}")
            print(f"📂 File: {item.meta.filename}")
            print(f"🎞️ Type: {item.media_type}")
            
            # Metadata is already parsed in MediaItem
            if item.ai:
                print(f"📝 Summary: {item.ai.summary}")
                print(f"🏷️ Tags: {item.ai.tags}")
            else:
                 print("📝 Summary: (No AI Data)")
            
            if item.transcription and item.transcription.full_text:
                 print(f"💬 Transcript: {item.transcription.full_text[:200]}...")
            else:
                 print("💬 Transcript: (None)")
                
        db.close()
    except Exception as e:
        print(f"Inspection failed: {e}")

if __name__ == "__main__":
    main()
