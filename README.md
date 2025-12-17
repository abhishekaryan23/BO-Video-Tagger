# BO Video Tagger V2

## Overview
**BO Video Tagger V2** (`tag_my_videos_v2.py`) is an automated, local-first video analysis tool that uses the **SmolVLM2** Vision Language Model to generate searchable tags and summaries for your video files. It processes videos locally on your machine, ensuring privacy and eliminating the need for cloud APIs.

## Features
- 🚀 **Local AI Processing**: Uses `llama.cpp` to run efficient GGUF models locally.
- ⚙️ **Adaptive Tiers**: Three processing modes (Normal, Smart, Super) to balance speed vs. accuracy.
- 🛡️ **System Safety**: Built-in RAM checks to prevent system crashes before loading models.
- 📦 **Auto-Management**: Automatically downloads the necessary specific model weights from HuggingFace.
- 💾 **Resume-Safe**: Atomically updates `video_tags.json` after every video, so you don't lose progress if interrupted.

## Prerequisites
- **Python**: Version 3.8 or higher.
- **RAM**: Minimum 2GB free for 'Normal' mode, up to 4GB+ for 'Super' mode.
- **Storage**: ~1GB - 2GB of disk space for model weights.

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
The script will interactively ask you to select a processing mode (Normal, Smart, or Super).

### Command Line Arguments
You can bypass the interactive menu using the `--mode` flag:

```bash
# Run in 'Smart' mode without prompts
python tag_my_videos_v2.py ./vacation_clips --mode smart
```

### Processing Modes
| Mode | RAM Req | Model Type | Description |
|------|---------|------------|-------------|
| **Normal** | ~1.5 GB | Q4_K_M | Fastest, good for general tagging. Default option. |
| **Smart** | ~2.0 GB | Q8_0 | Balanced. Better detail retention than Normal. |
| **Super** | ~3.0 GB | F16 | Lossless (Full Precision). Highest accuracy, requires most RAM. |

## Output
The script generates a `video_tags.json` file in the directory where you ran the script.

**Example Output:**
```json
[
  {
    "file": "beach_trip.mp4",
    "path": "/Users/videos/beach_trip.mp4",
    "analysis": "Keywords: [beach, ocean, sunset, waves, sand], Summary: [A relaxing view of waves crashing on the beach during sunset.]",
    "tier_used": "SmolVLM2-500M-Video-Instruct.Q4_K_M.gguf",
    "processing_time": "8.42s"
  }
]
```

## Troubleshooting
- **"Missing dependencies" error**: Run the `pip install` command displayed in the error message.
- **Script runs slowly**: 
    - Ensure you are running on a machine with a decent CPU or GPU. 
    - For Apple Silicon, verify you installed `llama-cpp-python` with `GGML_METAL=on`.
- **System crash / Out of Memory**: 
    - Try the **Normal** mode first.
    - Close other memory-intensive applications (Chrome, Photoshop, etc.).
