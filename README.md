![BO Video Tagger Logo](assets/logo.png)

# BO Video Tagger (v2.0)
**Your Local-First Video Intelligence Asset Manager.**

---
## 🚀 Getting Started

### 1. Requirements
-   **System**: MacOS (Apple Silicon recommended) or Linux with NVIDIA GPU.
-   **Deps**: Python 3.10+, ffmpeg.

### 2. Run as API Server (Backend)
Start the headless FastAPI server.
```bash
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```
-   **Docs**: `http://localhost:8000/docs`
-   **Process**: `POST /process`
-   **Search**: `GET /search?q=...`

### 3. Run Batch Worker (CLI)
Process a folder of media files directly from the terminal.
```bash
python bo_worker.py --dir "/path/to/media" --workers 2 --mode smart
```
-   `--dir`: Folder to scan recursively.
-   `--workers`: Parallel processing threads (beware VRAM usage).
-   `--mode`: `smart` (fast) or `super` (accurate).

---
---

# BO Video Tagger API Features Map

## 1. Core Endpoints

### ⚙️ Process Media
**Endpoint**: `POST /process`
**Description**: Triggers the AI pipeline for a single file. Indexing is performed synchronously (blocking) to ensure immediate availability, though concurrent requests are handled via thread pooling.
**Features**:
-   **Smart Skipping**: Checks database first; skips if file usage hasn't changed (unless `force_reprocess: true`).
-   **Multi-Model Analysis**:
    -   **Vision**: Uses `SmolVLM2` to generate dense visual descriptions and tags.
    -   **Audio**: Uses `Faster-Whisper` for automatic speech recognition (ASR) / transcription.
    -   **Vector Auth**: Generates 384-d semantic vectors (`all-MiniLM-L6-v2`) for search.
-   **Automatic Metadata**: Extracts duration, resolution, size, and timestamp.

### 🔍 Search Media
**Endpoint**: `GET /search`
**Description**: Performs a Hybrid Search combining keyword matching and semantic understanding.
**Features**:
-   **Hybrid Logic**:
    -   **Full-Text Search (FTS5)**: Specific keyword matching against Tags, Summary, Description, and *Transcription*.
    -   **Semantic Search**: (If configured) Vector similarity search to find concepts even without exact keywords (e.g. "dog" finds "puppy").
-   **Scoring**: Result candidates are ranked by relevance score.
-   **Filtering**: Optional `media_type` filter (video/image).

### 📂 List Media (Library)
**Endpoint**: `GET /media`
**Description**: The "Library View". Retrieves paginated, sorted, and filtered lists of assets.
**Features**:
-   **Pagination**: standard `limit` & `offset`. Returns `X-Total-Count` headers.
-   **Sorting**:
    -   `date_desc` / `date_asc` (Time)
    -   `duration_desc` / `duration_asc` (Length)
-   **Filtering**:
    -   `tag`: Exact substring match on tags.
    -   `media_type`: `video` or `image`.
    -   `date_from` / `date_to`: ISO date range filtering.

### 🩺 Health Check
**Endpoint**: `GET /health`
**Description**: System status ping.

---

## 2. Data Models (Schemas)

### `MediaItem`
The central object returned by most endpoints.
-   **`meta`**: File stats (path, size, duration, resolution).
-   **`ai`**: Generated intelligence.
    -   `description`: Full paragraph description.
    -   `summary`: One-line tl;dr.
    -   `tags`: List of keywords.
-   **`transcription`**:
    -   `full_text`: Complete speech transcript.
    -   `language`: Detected language (e.g. 'en').
-   **`media_type`**: `video` | `image`.

### `ErrorResponse` (RFC 7807)
Standardized error reporting for all 4xx/5xx responses.
-   `type`, `title`, `status`, `detail`, `instance`.

---

## 3. Underlying Capabilities

### Database (SQLite + FTS5)
-   **WAL Mode**: Enabled for high concurrency (writers don't block readers).
-   **FTS5**: Virtual table backend for instant text search over millions of rows.
-   **Vector Cache**: In-memory numpy cache for fast cosine similarity updates.

### AI Engine (Processor)
-   **Lazy Loading**: Models are loaded only when needed to save RAM.
-   **GPU Acceleration**: Metal (MPS) / CUDA support detected automatically.
-   **Fallback**: Robust error handling for corrupt files or undecodable audio.
