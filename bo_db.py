import sqlite3
import json
import os
import logging
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
import weakref

# Configure Logger
from bo_config import settings, configure_logging
from schemas import MediaItem, MediaType, MediaMetadata, AIResponse, TranscriptionData, SearchResponse

# Ensure logging is configured when DB is imported
configure_logging()
logger = logging.getLogger("VideoDB")

class VideoDB:
    def __init__(self, db_path: str = None):
        target_path = db_path or settings.DB_PATH
        self.conn = sqlite3.connect(target_path, check_same_thread=False, timeout=30.0)
        self.conn.row_factory = sqlite3.Row
        self._init_db()
        # Ensure connection is closed when object is garbage collected
        self._finalizer = weakref.finalize(self, self.conn.close)

        # Cache for Vector Search
        self._vector_cache: Dict[str, np.ndarray] = {}
        self._load_vector_cache()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def _init_db(self):
        """Initialize the database schema and enable WAL mode."""
        cursor = self.conn.cursor()
        
        # ENABLE WAL MODE for Concurrency (Writer doesn't block Reader)
        cursor.execute("PRAGMA journal_mode=WAL;")
        cursor.execute("PRAGMA synchronous=NORMAL;") 
        
        # Main Videos Table (Now General Media Table)
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
                processed_at DATETIME,
                parent_folder TEXT,
                media_type TEXT DEFAULT 'video',
                transcription TEXT,
                embedding BLOB
            )
        ''')

        # Schema Migration
        self._migrate_schema(cursor)
        
        # Full Text Search Index (FTS5)
        try:
            cursor.execute('CREATE VIRTUAL TABLE IF NOT EXISTS videos_search USING fts5(path, filename, tags, summary, description, transcription)')
            self.has_fts = True
        except sqlite3.OperationalError:
            logger.warning("FTS5 extension not available. Falling back to standard LIKE search.")
            self.has_fts = False
        
        # Performance Index
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_processed_at ON videos(processed_at);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_media_type ON videos(media_type);")

        self.conn.commit()

    def _migrate_schema(self, cursor):
        """Hands schema updates gracefully."""
        cursor.execute("PRAGMA table_info(videos)")
        columns = {row['name'] for row in cursor.fetchall()}
        
        # 1. Parent Folder
        if 'parent_folder' not in columns:
            self._add_column(cursor, "parent_folder", "TEXT")
            
        # 2. Media Type
        if 'media_type' not in columns:
            self._add_column(cursor, "media_type", "TEXT DEFAULT 'video'")
            
        # 3. Transcription
        if 'transcription' not in columns:
            self._add_column(cursor, "transcription", "TEXT")
            
        if 'embedding' not in columns:
            self._add_column(cursor, "embedding", "BLOB")
            
        # 5. Fix FTS Schema if needed
        # FTS5 tables are virtual and harder to alter. If 'transcription' is missing, likely old schema.
        try:
            cursor.execute("PRAGMA table_info(videos_search)")
            fts_columns = {row['name'] for row in cursor.fetchall()}
            if fts_columns and 'transcription' not in fts_columns:
                logger.warning("⚡ Migrating FTS Schema: Recreating videos_search with transcription support...")
                cursor.execute("DROP TABLE IF EXISTS videos_search")
                cursor.execute('CREATE VIRTUAL TABLE videos_search USING fts5(path, filename, tags, summary, description, transcription)')
                # Backfill
                cursor.execute('''
                    INSERT INTO videos_search(path, filename, tags, summary, description, transcription)
                    SELECT path, filename, tags, summary, description, transcription FROM videos
                ''')
                logger.info("✅ FTS Migration complete.")
        except Exception as e:
            logger.warning(f"FTS Migration check failed (ignoring): {e}")
        if 'embedding' not in columns:
            self._add_column(cursor, "embedding", "BLOB")

    def _add_column(self, cursor, col_name, col_type):
        try:
            logger.info(f"⚡ Migrating Schema: Adding {col_name}...")
            cursor.execute(f"ALTER TABLE videos ADD COLUMN {col_name} {col_type}")
        except Exception as e:
            logger.warning(f"Column {col_name} might already exist: {e}")

    def _load_vector_cache(self):
        """Loads all vectors into memory for fast search."""
        try:
            cursor = self.conn.cursor()
            cursor.execute("SELECT path, embedding FROM videos WHERE embedding IS NOT NULL")
            count = 0
            for row in cursor.fetchall():
                path, blob = row
                if blob:
                    # Convert bytes back to numpy array
                    vec = np.frombuffer(blob, dtype=np.float32)
                    self._vector_cache[path] = vec
                    count += 1
            logger.info(f"🧠 Loaded {count} vectors into memory cache.")
        except Exception as e:
            logger.error(f"Failed to load vector cache: {e}")

    def upsert_media(self, item: MediaItem, embedding_bytes: Optional[bytes] = None):
        """Insert or Update a media record."""
        path = item.meta.path
        if not path:
            return

        # Prepare Data
        tags_str = ",".join(item.ai.tags) if item.ai else ""
        summary = item.ai.summary if item.ai else ""
        description = item.ai.description if item.ai else ""
        
        transcription_text = item.transcription.full_text if item.transcription else ""
        
        # Serialize full object to JSON for 'metadata' column
        json_data = item.model_dump_json()
        
        # Timestamp
        processed_at = item.system.timestamp.isoformat() if item.system else datetime.now().isoformat()
        
        cursor = self.conn.cursor()
        
        try:
            # 1. Update Main Table
            cursor.execute('''
                INSERT INTO videos (
                    path, filename, size_mb, duration_sec, resolution, 
                    tags, summary, description, metadata, processed_at, parent_folder,
                    media_type, transcription, embedding
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(path) DO UPDATE SET
                    tags=excluded.tags,
                    summary=excluded.summary,
                    description=excluded.description,
                    metadata=excluded.metadata,
                    processed_at=excluded.processed_at,
                    parent_folder=excluded.parent_folder,
                    transcription=excluded.transcription,
                    embedding=excluded.embedding,
                    media_type=excluded.media_type
            ''', (
                path, item.meta.filename, item.meta.size_mb, item.meta.duration_sec, item.meta.resolution,
                tags_str, summary, description, json_data, processed_at, item.meta.parent_folder,
                item.media_type.value, transcription_text, embedding_bytes
            ))
            
            # 2. Update Search Index
            if self.has_fts:
                # Combine all text fields for FTS
                full_search_text = f"{tags_str} {summary} {description} {transcription_text}"
                cursor.execute('DELETE FROM videos_search WHERE path = ?', (path,))
                cursor.execute('''
                    INSERT INTO videos_search (path, filename, tags, summary, description, transcription)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (path, item.meta.filename, tags_str, summary, description, transcription_text))

            self.conn.commit()
            
            # 3. Update In-Memory Vector Cache
            if embedding_bytes:
                vec = np.frombuffer(embedding_bytes, dtype=np.float32)
                self._vector_cache[path] = vec
                
        except Exception as e:
            logger.error(f"DB Insert Error: {e}")
            self.conn.rollback()

    def get_all_media(
        self, 
        limit: int = 50, 
        offset: int = 0, 
        media_type: str = "all",
        sort_by: str = "date_desc",
        tag: Optional[str] = None,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None
    ) -> List[MediaItem]:
        """
        Retrieve media items with advanced filtering and sorting.
        sort_by: date_asc, date_desc, duration_asc, duration_desc
        """
        cursor = self.conn.cursor()
        
        query = "SELECT metadata FROM videos WHERE 1=1"
        params = []
        
        # 1. Filters
        if media_type != "all":
            query += " AND media_type = ?"
            params.append(media_type)
            
        if tag:
            query += " AND tags LIKE ?"
            params.append(f"%{tag}%")
            
        if date_from:
            query += " AND processed_at >= ?"
            params.append(date_from)
            
        if date_to:
            query += " AND processed_at <= ?"
            params.append(date_to)

        # 2. Sorting
        if sort_by == "date_asc":
            query += " ORDER BY processed_at ASC"
        elif sort_by == "duration_desc":
            query += " ORDER BY duration_sec DESC"
        elif sort_by == "duration_asc":
            query += " ORDER BY duration_sec ASC"
        else: # Default: date_desc
            query += " ORDER BY processed_at DESC"
            
        # 3. Pagination
        query += " LIMIT ? OFFSET ?"
        params.extend([limit, offset])
        
        # Execute
        cursor.execute(query, params)
        
        items = []
        for row in cursor.fetchall():
            try:
                data = json.loads(row[0])
                items.append(MediaItem(**data))
            except Exception:
                continue
        return items

    def get_total_count(self, media_type: str = "all", tag: Optional[str] = None) -> int:
        """Get total count for pagination headers."""
        cursor = self.conn.cursor()
        query = "SELECT COUNT(*) FROM videos WHERE 1=1"
        params = []
        
        if media_type != "all":
            query += " AND media_type = ?"
            params.append(media_type)
        
        if tag:
            query += " AND tags LIKE ?"
            params.append(f"%{tag}%")
            
        cursor.execute(query, params)
        return cursor.fetchone()[0]

    def get_media(self, path: str) -> Optional[Dict[str, Any]]:
        """Retrieves a specific media item (raw dict) by path for check."""
        try:
            cursor = self.conn.cursor()
            cursor.execute("SELECT * FROM videos WHERE path = ?", (path,))
            row = cursor.fetchone()
            if row:
                return dict(row)
            return None
        except Exception:
            # logger.error(f"Failed to get media path: {e}") 
            return None

    def search_media(self, query: str, query_vector: Optional[np.ndarray] = None, limit: int = 50) -> List[SearchResponse]:
        """Hybrid Search: FTS + Vector Cosine Similarity."""
        # 1. FTS Search (Keyword Match)
        fts_paths = set()
        fts_scores = {}
        
        if self.has_fts and query:
            cursor = self.conn.cursor()
            fts_query = f"{query}*" 
            try:
                # Use FTS rank if possible, or simple match
                cursor.execute('''
                    SELECT path, -rank as score 
                    FROM videos_search 
                    WHERE videos_search MATCH ? 
                    ORDER BY rank LIMIT ?
                ''', (fts_query, limit))
                
                for row in cursor.fetchall():
                    fts_paths.add(row[0])
                    fts_scores[row[0]] = 1.0  # Base score for keyword match
            except Exception:
                pass

        # 2. Vector Search (Semantic)
        vec_scores = {}
        if query_vector is not None and self._vector_cache:
            # Normalize query
            norm_q = np.linalg.norm(query_vector)
            if norm_q > 0:
                q_unit = query_vector / norm_q
                
                # Bulk Compute: Matrix Multiplication
                # Keys and Values lists
                paths = list(self._vector_cache.keys())
                vectors = list(self._vector_cache.values())
               
                if vectors:
                    # Stack: (N, D)
                    matrix = np.stack(vectors)
                    # Normalize Matrix (Pre-normalized in service usually, but let's be safe)
                    # Assuming stored vectors are normalized... check embedding.py (yes they are)
                    
                    # Dot Product: (N, D) @ (D,) -> (N,)
                    sims = matrix @ q_unit
                    
                    # Filter top K
                    # Get indices of top limit
                    if len(sims) > limit:
                        # argpartition is faster than sort
                        top_indices = np.argpartition(sims, -limit)[-limit:]
                    else:
                        top_indices = np.arange(len(sims))
                        
                    for idx in top_indices:
                        score = float(sims[idx])
                        if score > settings.SEARCH_CUTOFF: # Metadata cutoff
                            path = paths[idx]
                            vec_scores[path] = score

        # 3. Merge & Rank
        all_paths = fts_paths.union(vec_scores.keys())
        results = []
        
        for path in all_paths:
            # Hybrid Score (Adjustable via Config)
            s_fts = fts_scores.get(path, 0.0)
            s_vec = vec_scores.get(path, 0.0)
            
            final_score = (s_fts * settings.SEARCH_FTS_WEIGHT) + (s_vec * settings.SEARCH_VEC_WEIGHT)
            
            # Fetch Metadata (Light fetch)
            cursor = self.conn.cursor()
            cursor.execute("SELECT filename, summary, tags, media_type FROM videos WHERE path = ?", (path,))
            row = cursor.fetchone()
            if row:
                results.append(SearchResponse(
                    filename=row[0],
                    path=path,
                    score=round(final_score, 3),
                    media_type=row[3],
                    summary=row[1] or "",
                    tags=row[2].split(",") if row[2] else []
                ))

        # Sort by Final Score
        results.sort(key=lambda x: x.score, reverse=True)
        return results[:limit]

    def get_all_file_paths_set(self) -> set:
        """Returns a set of all indexed file paths for O(1) lookups."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT path FROM videos")
        return {row[0] for row in cursor.fetchall()}

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
                            # Adapt Legacy Dict to MediaItem
                            try:
                                item = MediaItem(**record)
                            except Exception:
                                # Best effort adaptation
                                item = MediaItem(
                                    media_type=MediaType.VIDEO, # Assume video for legacy
                                    meta=MediaMetadata(**record.get('meta', {})),
                                    ai=AIResponse(**record.get('ai', {})),
                                    system=None
                                )
                                
                            self.upsert_media(item)
                            
                        except Exception as e:
                            logger.error(f"Skipping invalid record in {file}: {e}")
                
                # Rename after successful import
                file.rename(file.with_suffix('.jsonl.bak'))
                logger.info(f"Migration complete. Renamed to {file.name}.bak")
                
            except Exception as e:
                logger.error(f"Failed to migrate {file}: {e}")

