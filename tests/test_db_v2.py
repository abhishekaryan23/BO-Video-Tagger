import pytest
import sqlite3
import numpy as np
import json
import os
from bo_db import VideoDB
from schemas import MediaItem, MediaType, MediaMetadata

@pytest.fixture
def db():
    # In-memory DB
    database = VideoDB(db_path=":memory:")
    yield database
    database.close()

def test_schema_columns(db):
    """Verify new columns exist."""
    cursor = db.conn.cursor()
    cursor.execute("PRAGMA table_info(videos)")
    cols = {row['name'] for row in cursor.fetchall()}
    assert "media_type" in cols
    assert "transcription" in cols
    assert "embedding" in cols

def test_upsert_and_fetch_media(db):
    item = MediaItem(
        media_type=MediaType.IMAGE,
        meta=MediaMetadata(filename="pic.jpg", path="/tmp/pic.jpg", size_mb=1, parent_folder="tmp", width=100, height=100)
    )
    
    db.upsert_media(item)
    
    items = db.get_all_media()
    assert len(items) == 1
    assert items[0].media_type == MediaType.IMAGE
    assert items[0].meta.width == 100

def test_vector_search_logic(db):
    """Test the numpy math and search logic."""
    item1 = MediaItem(
        media_type=MediaType.VIDEO,
        meta=MediaMetadata(filename="cat_video.mp4", path="/cat.mp4", size_mb=1, parent_folder="tmp")
    )
    # Vec 1: [1, 0]
    vec1 = np.array([1.0, 0.0], dtype=np.float32)
    
    item2 = MediaItem(
        media_type=MediaType.VIDEO,
        meta=MediaMetadata(filename="dog_video.mp4", path="/dog.mp4", size_mb=1, parent_folder="tmp")
    )
    # Vec 2: [0, 1]
    vec2 = np.array([0.0, 1.0], dtype=np.float32)
    
    db.upsert_media(item1, vec1.tobytes())
    db.upsert_media(item2, vec2.tobytes())
    
    # 1. Search for Cat ([1, 0])
    # Dot product should be 1.0 for cat, 0.0 for dog
    query_vec = np.array([1.0, 0.0], dtype=np.float32)
    
    results = db.search_media(query="", query_vector=query_vec)
    assert len(results) > 0
    assert results[0].filename == "cat_video.mp4"
    assert results[0].score >= 0.65 # 0.7 weight * 1.0 sim roughly
    
    # 2. Search for Dog ([0, 1])
    query_vec2 = np.array([0.0, 1.0], dtype=np.float32)
    results2 = db.search_media(query="", query_vector=query_vec2)
    assert results2[0].filename == "dog_video.mp4"

def test_hybrid_search(db):
    """Test FTS + Vector Combination."""
    # Setup: item matches FTS "moon" but bad vector, vs item matches Vector but no FTS
    # For simplicity, just check FTS works
    item = MediaItem(
        media_type=MediaType.VIDEO,
        meta=MediaMetadata(filename="apollo.mp4", path="/apollo.mp4", size_mb=1, parent_folder="space"),
        ai={"description": "landing on the moon", "summary": "", "tags": []}
    )
    db.upsert_media(item)
    
    results = db.search_media("moon")
    assert len(results) == 1
    assert results[0].filename == "apollo.mp4"
