import sqlite3
import json
import os
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path

# Configure Logger
from bo_config import Settings, configure_logging

# Ensure logging is configured when DB is imported
configure_logging()
logger = logging.getLogger("VideoDB")

class VideoDB:
    def __init__(self):
        self.conn = sqlite3.connect(Settings.DB_PATH, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._init_db()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def _init_db(self):
        """Initialize the database schema and enable WAL mode."""
        cursor = self.conn.cursor()
        
        # ENABLE WAL MODE for Concurrency (Writer doesn't block Reader)
        cursor.execute("PRAGMA journal_mode=WAL;")
        cursor.execute("PRAGMA synchronous=NORMAL;") # Faster writes, slightly less safe on power loss
        
        # Main Videos Table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS videos (
                path TEXT PRIMARY KEY,
                filename TEXT NOT NULL,
                size_mb REAL,
                duration_sec REAL,
                resolution TEXT,
                tags TEXT,
                summary TEXT,
                description TEXT,
                metadata JSON,
                processed_at DATETIME
            )
        ''')
        
        # Full Text Search Index (FTS5)
        try:
            cursor.execute('CREATE VIRTUAL TABLE IF NOT EXISTS videos_search USING fts5(path, filename, tags, summary, description)')
            self.has_fts = True
        except sqlite3.OperationalError:
            logger.warning("FTS5 extension not available. Falling back to standard LIKE search.")
            self.has_fts = False
        
        # Performance Index for Gallery Sorting
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_videos_processed_at ON videos(processed_at);")

        self.conn.commit()

    def upsert_video(self, data: Dict[str, Any]):
        """Insert or Update a video record."""
        meta = data.get("meta", {})
        ai = data.get("ai", {})
        system = data.get("system", {})
        
        path = meta.get("path")
        if not path:
            return

        filename = meta.get("file", os.path.basename(path))
        size_mb = meta.get("size_mb", 0.0)
        duration = meta.get("duration_sec", 0.0)
        resolution = meta.get("resolution", "Unknown")
        
        tags = ",".join(ai.get("tags", []))
        summary = ai.get("summary", "")
        description = ai.get("description", "")
        processed_at = system.get("timestamp", datetime.now().isoformat())

        cursor = self.conn.cursor()
        
        try:
            # 1. Update Main Table
            cursor.execute('''
                INSERT INTO videos (path, filename, size_mb, duration_sec, resolution, tags, summary, description, metadata, processed_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(path) DO UPDATE SET
                    tags=excluded.tags,
                    summary=excluded.summary,
                    description=excluded.description,
                    metadata=excluded.metadata,
                    processed_at=excluded.processed_at
            ''', (path, filename, size_mb, duration, resolution, tags, summary, description, json.dumps(data), processed_at))
            
            # 2. Update Search Index (if FTS enabled)
            if self.has_fts:
                cursor.execute('DELETE FROM videos_search WHERE path = ?', (path,))
                cursor.execute('''
                    INSERT INTO videos_search (path, filename, tags, summary, description)
                    VALUES (?, ?, ?, ?, ?)
                ''', (path, filename, tags, summary, description))

            self.conn.commit()
        except Exception as e:
            logger.error(f"DB Insert Error: {e}")
            self.conn.rollback()

    def get_all_videos(self, limit: int = 100, offset: int = 0) -> List[dict]:
        """Retrieve all videos for gallery view."""
        cursor = self.conn.cursor()
        cursor.execute('SELECT * FROM videos ORDER BY processed_at DESC LIMIT ? OFFSET ?', (limit, offset))
        return [dict(row) for row in cursor.fetchall()]

    def search_videos(self, query: str) -> List[dict]:
        """Search videos using FTS or LIKE."""
        cursor = self.conn.cursor()
        
        if self.has_fts:
            fts_query = f"{query}*" 
            try:
                cursor.execute('''
                    SELECT v.* FROM videos v
                    JOIN videos_search s ON v.path = s.path
                    WHERE s.videos_search MATCH ?
                    ORDER BY rank
                ''', (fts_query,))
            except sqlite3.OperationalError:
                return self._search_like(query)
        else:
            return self._search_like(query)
            
        return [dict(row) for row in cursor.fetchall()]
    
    def _search_like(self, query: str) -> List[dict]:
        """Fallback search using LIKE."""
        cursor = self.conn.cursor()
        param = f"%{query}%"
        cursor.execute('''
            SELECT * FROM videos 
            WHERE filename LIKE ? OR tags LIKE ? OR summary LIKE ? OR description LIKE ?
        ''', (param, param, param, param))
        return [dict(row) for row in cursor.fetchall()]

    def update_metadata(self, path: str, description: str, tags: List[str]):
        """Update description and tags from UI Editor."""
        cursor = self.conn.cursor()
        
        try:
            cursor.execute('SELECT metadata FROM videos WHERE path = ?', (path,))
            row = cursor.fetchone()
            if not row:
                return
                
            full_data = json.loads(row['metadata'])
            full_data['ai']['description'] = description
            full_data['ai']['tags'] = tags
            
            tags_str = ",".join(tags)
            
            cursor.execute('''
                UPDATE videos 
                SET description = ?, tags = ?, metadata = ?
                WHERE path = ?
            ''', (description, tags_str, json.dumps(full_data), path))
            
            if self.has_fts:
                cursor.execute('DELETE FROM videos_search WHERE path = ?', (path,))
                cursor.execute('''
                    INSERT INTO videos_search (path, filename, tags, summary, description)
                    VALUES (?, ?, ?, ?, ?)
                ''', (path, os.path.basename(path), tags_str, full_data['ai'].get('summary', ''), description))
                
            self.conn.commit()
        except Exception as e:
            logger.error(f"Metadata Update Error: {e}")
            self.conn.rollback()

    def get_analytics_df(self):
        """Fetch DataFrame for analytics (Memory Optimized)."""
        import pandas as pd
        
        query = '''
            SELECT path, size_mb, duration_sec, tags, processed_at, metadata 
            FROM videos
        '''
        return pd.read_sql_query(query, self.conn)

    def close(self):
        self.conn.close()

    # --- Migration Logic ---
    def migrate_jsonl(self, folder_path: str):
        """Scans folder for .jsonl files and imports them to SQLite."""
        path_obj = Path(folder_path)
        if not path_obj.exists():
            return

        for file in path_obj.glob("*_video_tags_*.jsonl"):
            try:
                logger.info(f"Migrating {file}...")
                with open(file, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if not line: continue
                        try:
                            record = json.loads(line)
                            self.upsert_video(record)
                        except json.JSONDecodeError:
                            logger.error(f"Skipping invalid JSON line in {file}")
                
                # Rename after successful import
                file.rename(file.with_suffix('.jsonl.bak'))
                logger.info(f"Migration complete. Renamed to {file.name}.bak")
                
            except Exception as e:
                logger.error(f"Failed to migrate {file}: {e}")
