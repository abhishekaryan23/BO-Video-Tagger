import time
import logging
import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from bo_db import VideoDB
from services.embedding import EmbeddingService

# Configure logging to show only critical errors to keep output clean
logging.getLogger("VideoDB").setLevel(logging.ERROR)
logging.getLogger("Embedder").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)

def main():
    print("🚀 Initializing Semantic Search Test...")
    
    # 1. Initialize DB (Loads Vector Cache)
    t0 = time.time()
    db = VideoDB()
    t_db = time.time() - t0
    print(f"✅ Database Initialized in {t_db:.4f}s")
    
    # 2. Initialize Model (Loads into RAM)
    print("🧠 Loading Embedding Model...")
    t0 = time.time()
    embedder = EmbeddingService()
    # Force load
    _ = embedder.model
    t_model = time.time() - t0
    print(f"✅ Model Loaded in {t_model:.4f}s")

    queries = [
        "people discussing technology",   # Should match robot video
        "guard dog in rural setting",     # Should match farm video
        "woman drinking coffee",          # Should match image
        "something completely random"     # Should match nothing (low scores)
    ]
    
    print(f"\n🔎 Testing {len(queries)} queries...\n")
    
    for q in queries:
        print(f"Query: '{q}'")
        
        # A. Embed Query
        t0 = time.time()
        query_vector = embedder.embed_query(q)
        t_embed = time.time() - t0
        
        # B. Search DB
        t0 = time.time()
        results = db.search_media(q, query_vector=query_vector, limit=3)
        t_search = time.time() - t0
        
        # Output Stats
        total_time = (t_embed + t_search) * 1000 # ms
        print(f"   ⏱️  Embedding: {t_embed*1000:.2f}ms | Search: {t_search*1000:.2f}ms | Total: {total_time:.2f}ms")
        
        # Output Matching Results
        if results:
            print(f"   🏆 Top Match: {results[0].filename} (Score: {results[0].score:.4f})")
            # Optional: Show 2nd match if close
            if len(results) > 1:
                 print(f"      2nd Match: {results[1].filename} (Score: {results[1].score:.4f})")
        else:
            print("   ❌ No results found.")
        print("-" * 50)

    db.close()

if __name__ == "__main__":
    main()
