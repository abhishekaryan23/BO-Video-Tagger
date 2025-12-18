# BO Video Tagger V2

## Overview
**BO Video Tagger V2** (`tag_my_videos_v2.py`) is an automated, local-first video analysis tool that uses the **SmolVLM2** Vision Language Model to generate searchable tags and description summaries for your video files. It processes videos locally on your machine, ensuring privacy and eliminating the need for cloud APIs.

## Features
- 🚀 **Local AI Processing**: Uses `llama.cpp` to run efficient GGUF models locally.
- 👁️ **Vision Enabled**: Downloads and uses the official multimodal projector for accurate video understanding.
- ⚙️ **Adaptive Tiers**: Two processing modes (Smart, Super) to balance quality vs. resources.
- 🛡️ **System Safety**: Built-in RAM checks to prevent system crashes before loading models.
- 📦 **Auto-Management**: Automatically downloads the specific model weights from the `ggml-org` repository.
- 🐛 **Debug Mode**: Verify what the AI "sees" by saving extracted frames to disk.

## Prerequisites
- **Python**: Version 3.8 or higher.
- **RAM**: Minimum ~2.5GB free for 'Smart' mode, up to ~4.0GB+ for 'Super' mode.
- **Storage**: ~3GB - 5GB of disk space for model weights.

## Installation

1. **Install Dependencies**
   The script checks for dependencies on launch, but you can pre-install them using the generated requirements file:
   ```bash
   pip install -r requirements.txt
   ```

   **🍎 MacOS / Apple Silicon Users (M1/M2/M3)**
   To enable GPU acceleration (Metal) for much faster processing, install `llama-cpp-python` with the following command:
   ```bash
   CMAKE_ARGS="-DGGML_METAL=on" pip install llama-cpp-python --upgrade --force-reinstall --no-cache-dir
   ```

## Usage

### Basic Usage
Run the script and provide the path to your folder containing videos:
```bash
python tag_my_videos_v2.py /path/to/your/video_folder
```
This runs in **Smart (Default)** mode.

### Command Line Arguments
You can control the behavior using flags:

```bash
# Run in 'Super' mode for maximum accuracy
python tag_my_videos_v2.py ./vacation_clips --mode super

# Run with Debugging enabled (saves frames to ./debug_frames/)
python tag_my_videos_v2.py ./vacation_clips --debug
```

### Processing Modes
| Mode | RAM Req | Model Type | Description |
|------|---------|------------|-------------|
| **Smart** | ~2.5 GB | Q8_0 | **Default**. High quality balanced with speed. (Repo: `ggml-org`) |
| **Super** | ~4.0 GB | F16 | Lossless (Full Precision). Highest accuracy, requires more RAM. |

### Compatibility Note (Google Drive / Cloud)
The script works with mounted Cloud Drives (like Google Drive for Desktop).
However, if streaming is slow, the script might read blank frames.
**Tip:** Use the `--debug` flag to check if the script is successfully extracting images from your cloud drive. If the images in `debug_frames` are valid, the AI will work.

## Output
The script generates a `video_tags.jsonl` (JSON Lines) file. Each line is a self-contained JSON object, which is generally safer and faster for large datasets.

**Example Output (One Line):**
```json
{"file": "beach.mp4", "path": "/videos/beach.mp4", "analysis": "Sunny beach...", "tier_used": "SmolVLM...", "processing_time_sec": 12.5}
```
[
  {
    "file": "beach_trip.mp4",
    "path": "/Users/videos/beach_trip.mp4",
    "analysis": "The video shows a sunny beach with waves crashing gently on the shore. People are walking in the distance.\n\nKeywords: beach, ocean, waves, sunny, relaxation",
    "tier_used": "SmolVLM2-500M-Video-Instruct-Q8_0.gguf",
    "processing_time": "12.42s"
  }
]
```

## Troubleshooting
- **Loops/Repetitive Text**: The updated V2 script uses valid vision projectors to solve this. Ensure you have internet access on the first run to download the new `mmproj` files.
- **"Missing dependencies" error**: Run `pip install -r requirements.txt`.
- **System crash / Out of Memory**: 
    - Ensure you have enough free RAM.
    - Close other memory-intensive applications (Chrome, Photoshop, etc.).
