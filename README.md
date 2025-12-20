<div align="center">
  <img src="assets/logo.png" alt="BO Video Tagger Logo" width="200" />
  <h1>BO Video Tagger (v2.0)</h1>
  <p><strong>Your Local-First Video Intelligence Asset Manager.</strong></p>
</div>

---

### The Problem
You have terabytes of video footage. 
You know you filmed a "drone shot of a forest at sunset" three years ago. 
But finding it involves manually scrubbing through hundreds of files named `DJI_0049.MOV` or `C0012.MP4`.

### The Solution
**BO Video Tagger** watches your videos so you don't have to. 
It creates a searchable, intelligent index of your entire library—tags, descriptions, and summaries—saved locally on your machine.

No cloud uploads. No monthly fees. Just you, your data, and an AI that works for you.

---

## 🎯 Who Is This For?

| Role | Why You Need This |
| :--- | :--- |
| **Video Editors** | Find b-roll instantly. Search for "happy couple laughing" and drag it into your timeline. |
| **Archivists** | Standardize metadata across decades of footage without lifting a finger. |
| **Content Creators** | Repurpose old content. Ask your library: *"Show me all clips discussing AI from 2023."* |
| **Data Hoarders** | Finally understand what's actually taking up space on your NAS. |

---

## 🚀 Performance
Speed matters. We built the **Titanium Engine** to respect your time and your hardware.

![Performance Chart](assets/performance_chart.png)

-   **O(1) Smart Skip**: Re-scanning a 10TB library takes seconds. We only process new files.
-   **Local Acceleration**: Optimized for Apple Silicon (Metal) and NVIDIA GPUs (CUDA).
-   **Zero Latency Search**: 100k+ assets? Search results appear in <50ms thanks to FTS5 SQLite indexing.

---

## 🛠 Features

### 🧠 **Local Intelligence**
Running on **SmolVLM2**, the tagger understands visual context, text on screen (OCR), and complex actions. It doesn't just see "dog"; it sees "Golden Retriever catching a frisbee in a park."

### ⚡ **Control Deck**
You have full control over the indexing process.
-   **Tier Selection**: Choose `Smart` (Speed) or `Super` (Precision) models.
-   **Force Reprocess**: One-click re-indexing for updated models.
-   **Live Analytics**: Monitor health, storage usage, and tag density in real-time.

### 🔒 **Privacy by Design**
-   **100% Offline**: Unplug your ethernet cable. It still works.
-   **Verifiable Integrity**: All models are SHA256 checksummed before execution.

---

## 🖼 Interface
**Clean. Dark. Data-Dense.**

### Library
*Filter, Search, and Inspect.*
![Library View](assets/library_blurred.png)

### Analytics
*Know your data.*
![Analytics Dashboard](assets/analytics_blurred.png)

---

## ⚙️ Quick Start

**1. Install**
```bash
git clone https://github.com/abhishekaryan23/BO-Video-Tagger.git
cd BO-Video-Tagger
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**2. Launch**
```bash
streamlit run app.py
```

**3. Index**
Point the sidebar to your footage folder (e.g., `/Users/studio/Footage`) and click **Start Indexing**.

---

*Engineered with precision by the BO Video Tagger Team.*
