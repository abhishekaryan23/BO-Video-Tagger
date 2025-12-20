import pytest
import sqlite3
import json
import os
from datetime import datetime
from bo_db import VideoDB

# Fixture for In-Memory DB
@pytest.fixture
def db():
    # Use :memory: for fast, isolated tests
    database = VideoDB(db_path=":memory:")
    yield database
    database.close()

def test_db_initialization(db):
    """Verify tables and WAL mode are set up."""
    cursor = db.conn.cursor()
    
    # Check WAL mode (In-Memory defaults to MEMORY usually)
    cursor.execute("PRAGMA journal_mode;")
    mode = cursor.fetchone()[0]
    assert mode.upper() in ["WAL", "MEMORY"]

    # Check Table Exists
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='videos';")
    assert cursor.fetchone() is not None

def test_upsert_video(db):
    """Test inserting and updating video metadata."""
    video_data = {
        "meta": {
            "path": "/tmp/test.mp4",
            "file": "test.mp4",
            "size_mb": 10.5,
            "duration_sec": 60.0,
            "resolution": "1920x1080"
        },
        "ai": {
            "tags": ["outdoors", "drone"],
            "summary": "A test video",
            "description": "Flying over a field."
        },
        "system": {
            "timestamp": datetime.now().isoformat()
        }
    }

    # 1. Insert
    db.upsert_video(video_data)
    
    # Verify
    videos = db.get_all_videos()
    assert len(videos) == 1
    v = videos[0]
    assert v['filename'] == "test.mp4"
    assert "drone" in v['tags']
    assert v['resolution'] == "1920x1080"

    # 2. Update (Upsert)
    video_data['ai']['summary'] = "Updated Summary"
    db.upsert_video(video_data)
    
    videos = db.get_all_videos()
    assert len(videos) == 1
    assert videos[0]['summary'] == "Updated Summary"

def test_search_videos(db):
    """Test full text search / fallback."""
    # Insert dummy data
    db.upsert_video({"meta": {"path": "/a.mp4", "file": "moon_landing.mp4"}, "ai": {"tags": ["space", "nasa"], "description": "Apollo 11"}} )
    db.upsert_video({"meta": {"path": "/b.mp4", "file": "cooking.mp4"}, "ai": {"tags": ["food", "kitchen"], "description": "Making pasta"}} )

    # Test exact match
    results = db.search_videos("nasa")
    assert len(results) == 1
    assert results[0]['filename'] == "moon_landing.mp4"

    # Test unknown
    results = db.search_videos("cars")
    assert len(results) == 0

def test_get_unique_folders(db):
    """Test folder extraction."""
    db.upsert_video({"meta": {"path": "/Users/Me/Movies/SciFi/a.mp4"}})
    db.upsert_video({"meta": {"path": "/Users/Me/Movies/Action/b.mp4"}})
    db.upsert_video({"meta": {"path": "/Users/Me/Movies/SciFi/c.mp4"}})

    folders = db.get_unique_folders()
    assert "SciFi" in folders
    assert "Action" in folders
    assert len(folders) == 2

def test_metadata_update(db):
    """Test manual metadata update."""
    path = "/update_test.mp4"
    db.upsert_video({"meta": {"path": path}, "ai": {"tags": ["old"], "description": "old desc"}})
    
    db.update_metadata(path, "new desc", ["new", "tags"])
    
    # Reload
    videos = db.get_all_videos()
    v = videos[0]
    assert v['description'] == "new desc"
    assert v['tags'] == "new,tags"

def test_parent_folder_optimization(db):
    """Test that parent_folder is correctly populated and queried."""
    db.upsert_video({"meta": {"path": "/Volumes/Drive/Movies/SciFi/alien.mp4"}})
    db.upsert_video({"meta": {"path": "/Volumes/Drive/Movies/Action/rambo.mp4"}})
    
    # Check if parent_folder was populated in DB
    cursor = db.conn.cursor()
    cursor.execute("SELECT filename, parent_folder FROM videos ORDER BY filename")
    rows = cursor.fetchall()
    
    assert rows[0]['parent_folder'] == "SciFi"
    assert rows[1]['parent_folder'] == "Action"
    
    # Check Optimized Get Unique Folders
    folders = db.get_unique_folders()
    assert folders == ["Action", "SciFi"]
    
    # Check Set lookup
    path_set = db.get_all_file_paths_set()
    assert "/Volumes/Drive/Movies/SciFi/alien.mp4" in path_set
