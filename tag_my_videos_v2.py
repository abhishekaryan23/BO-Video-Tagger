import os
import sys
import json
import time
import argparse
import logging
from typing import List, Dict, Any

# --- Dependency Check & Auto-Install Recommendation ---
required_packages = {
    "cv2": "opencv-python",
    "huggingface_hub": "huggingface-hub",
    "llama_cpp": "llama-cpp-python",
    "tqdm": "tqdm",
    "psutil": "psutil"  # Added for system checks
}

missing_pkgs = []
for import_name, install_name in required_packages.items():
    try:
        __import__(import_name)
    except ImportError:
        missing_pkgs.append(install_name)

if missing_pkgs:
    print(f"❌ Missing dependencies. Please run:")
    print(f"pip install {' '.join(missing_pkgs)}")
    sys.exit(1)

# Imports after check
import cv2
import base64
import psutil
from huggingface_hub import hf_hub_download
from llama_cpp import Llama
from tqdm import tqdm

# --- Configuration & Tiers ---
REPO_ID = "mradermacher/SmolVLM2-500M-Video-Instruct-GGUF"

# Dictionary mapping friendly names to specific GGUF files and RAM requirements
MODEL_TIERS = {
    "normal": {
        "filename": "SmolVLM2-500M-Video-Instruct.Q4_K_M.gguf",
        "desc": "Fastest (Q4 Quantization)",
        "min_ram_gb": 1.5
    },
    "smart": {
        "filename": "SmolVLM2-500M-Video-Instruct.Q8_0.gguf",
        "desc": "Balanced (Q8 Quantization)",
        "min_ram_gb": 2.0
    },
    "super": {
        "filename": "SmolVLM2-500M-Video-Instruct.f16.gguf",
        "desc": "Max Precision (F16 - Lossless)",
        "min_ram_gb": 3.0
    }
}

CONTEXT_SIZE = 8192  # Video models need large context for multiple frames

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_system_resources(tier: str) -> bool:
    """
    Checks if the system has enough RAM to run the selected tier.
    Returns True if safe, False if risky (user can override).
    """
    # 1. Get System Stats
    mem = psutil.virtual_memory()
    total_ram_gb = mem.total / (1024 ** 3)
    available_ram_gb = mem.available / (1024 ** 3)
    
    req_ram = MODEL_TIERS[tier]['min_ram_gb']
    
    print(f"\n🖥️  System Check for '{tier.upper()}' mode:")
    print(f"   • Required RAM: ~{req_ram} GB")
    print(f"   • Available RAM: {available_ram_gb:.2f} GB (Total: {total_ram_gb:.2f} GB)")

    # 2. Heuristic Check
    if available_ram_gb < req_ram:
        print(f"⚠️  WARNING: You have {available_ram_gb:.2f} GB available, but this mode needs {req_ram} GB.")
        print("   The system might swap (slow) or crash.")
        response = input("   Do you want to continue anyway? (y/n): ").lower()
        return response == 'y'
    
    print("✅ System resources look good.")
    return True

class BRollTagger:
    def __init__(self, tier: str = "normal", interval: int = 10):
        self.interval = interval
        self.tier_config = MODEL_TIERS[tier]
        self.filename = self.tier_config['filename']
        
        # 1. Model Download Path
        # We store models in a local 'models' folder to keep things tidy
        self.model_dir = os.path.join(os.getcwd(), "models")
        os.makedirs(self.model_dir, exist_ok=True)
        self.model_path = os.path.join(self.model_dir, self.filename)

        # 2. Ensure Model Exists
        if not os.path.exists(self.model_path):
            print(f"\n⬇️  Downloading {tier.upper()} model ({self.filename})...")
            try:
                # Download to specific path
                self.model_path = hf_hub_download(
                    repo_id=REPO_ID, 
                    filename=self.filename,
                    local_dir=self.model_dir,
                    local_dir_use_symlinks=False
                )
                print(f"✅ Download complete: {self.model_path}")
            except Exception as e:
                logger.error(f"Failed to download model: {e}")
                sys.exit(1)

        # 3. Load Model
        print(f"🤖 Loading {tier.upper()} Neural Network...")
        try:
            self.llm = Llama(
                model_path=self.model_path,
                n_ctx=CONTEXT_SIZE,
                n_gpu_layers=-1, # Auto-offload to GPU if available
                verbose=False
            )
        except Exception as e:
            logger.error(f"Failed to initialize Llama: {e}")
            sys.exit(1)

    def extract_frames(self, video_path: str, max_frames: int = 5) -> List[str]:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return []

        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = int(fps * self.interval) if fps > 0 else 30
        
        base64_frames = []
        count = 0
        
        while cap.isOpened() and len(base64_frames) < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            if count % frame_interval == 0:
                # Resize to 384x384 (SmolVLM standard sweet spot)
                resized = cv2.resize(frame, (384, 384))
                _, buffer = cv2.imencode('.jpg', resized, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                b64_str = base64.b64encode(buffer).decode('utf-8')
                base64_frames.append(f"data:image/jpeg;base64,{b64_str}")
            
            count += 1
            
        cap.release()
        return base64_frames

    def tag_video(self, video_path: str) -> Dict[str, Any]:
        start_time = time.time()
        try:
            frames = self.extract_frames(video_path)
            if not frames:
                return {"file": os.path.basename(video_path), "error": "No video stream found"}

            # Prompt Construction
            content = [{"type": "image_url", "image_url": {"url": img}} for img in frames]
            content.append({
                "type": "text", 
                "text": "Describe this video for a search engine. Provide 5 keywords and a 1-sentence summary. Format: Keywords: [a, b, c], Summary: [text]"
            })

            response = self.llm.create_chat_completion(
                messages=[{"role": "user", "content": content}],
                max_tokens=150,
                temperature=0.1
            )
            
            return {
                "file": os.path.basename(video_path),
                "path": video_path,
                "analysis": response['choices'][0]['message']['content'],
                "tier_used": self.filename,
                "processing_time": f"{time.time() - start_time:.2f}s"
            }

        except Exception as e:
            return {"file": os.path.basename(video_path), "error": str(e)}

def select_mode():
    print("\nSelect Processing Mode:")
    print("1. [Normal] - Fast, Low RAM (Q4_K_M) [Default]")
    print("2. [Smart]  - Better Detail (Q8_0)")
    print("3. [Super]  - Max Precision (F16) - *High RAM Required*")
    
    choice = input("Enter choice (1-3): ").strip()
    
    if choice == "2": return "smart"
    if choice == "3": return "super"
    return "normal"

def main():
    parser = argparse.ArgumentParser(description="BO Video Tagger V2")
    parser.add_argument("folder", help="Path to video folder")
    parser.add_argument("--mode", choices=["normal", "smart", "super"], help="Override interactive mode selection")
    args = parser.parse_args()

    # 1. Mode Selection
    mode = args.mode if args.mode else select_mode()
    
    # 2. System Check
    if not check_system_resources(mode):
        print("❌ Aborted by user due to system requirements.")
        return

    # 3. Initialize
    tagger = BRollTagger(tier=mode)
    
    # 4. Scan & Process
    video_files = [
        os.path.join(args.folder, f) for f in os.listdir(args.folder) 
        if f.lower().endswith(('.mp4', '.mov', '.avi', '.mkv'))
    ]
    
    if not video_files:
        print("❌ No video files found.")
        return

    print(f"\n📂 Processing {len(video_files)} videos in '{mode.upper()}' mode...")
    results = []
    
    with tqdm(total=len(video_files), unit="vid") as pbar:
        for vid in video_files:
            res = tagger.tag_video(vid)
            results.append(res)
            
            # Atomic Write (Prevents corruption)
            with open("video_tags.json", 'w') as f:
                json.dump(results, f, indent=2)
            
            pbar.set_postfix(file=os.path.basename(vid)[:10])
            pbar.update(1)

    print("\n✅ Done! Check 'video_tags.json' for results.")

if __name__ == "__main__":
    main()