import streamlit as st
import time
from bo_config import Settings, configure_logging
from bo_db import VideoDB
from bo_worker import VideoWorker, WorkerSignals

# Component Imports
from components.sidebar import render_sidebar
from components.gallery import render_gallery
from components.analytics import render_analytics
from components.inspector import check_inspector_state

# --- 1. CONFIGURATION & STYLING ---
st.set_page_config(
    page_title="BO Video Tagger",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for "Netflix-Style" Dark Theme
STYLING = """
<style>
    /* Global Reset & Dark Theme */
    [data-testid="stAppViewContainer"] {
        background-color: #0e1117;
    }
    
    /* Remove standard Streamlit padding */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 95% !important;
    }
    
    /* Sidebar Styling (Glassmorphism) */
    [data-testid="stSidebar"] {
        background-color: #161b22;
        border-right: 1px solid #30363d;
    }
    
    /* Card/Gallery Styling */
    .movie-card {
        background-color: #1e2127;
        border-radius: 8px;
        padding: 0;
        margin-bottom: 20px;
        transition: transform 0.2s;
        border: 1px solid #30363d;
        overflow: hidden;
        cursor: pointer;
    }
    .movie-card:hover {
        transform: scale(1.02);
        border-color: #58a6ff;
        box-shadow: 0 10px 30px rgba(0,0,0,0.5);
    }
    .card-img {
        width: 100%;
        aspect-ratio: 16 / 9;
        object-fit: cover;
        opacity: 0.9;
    }
    .card-img:hover {
        opacity: 1.0;
    }
    .tag-pill {
        display: inline-block;
        background: linear-gradient(90deg, #238636 0%, #2ea043 100%);
        color: white;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 0.7rem;
        margin-right: 4px;
        margin-bottom: 4px;
    }
    
    /* Hide Streamlit Elements */
    /* Hide Streamlit Elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    /* header {visibility: hidden;}  <-- This hides the sidebar toggle! */
    [data-testid="stHeader"] {
        background-color: rgba(0,0,0,0);
    }
</style>
"""
st.markdown(STYLING, unsafe_allow_html=True)

# --- 2. STATE MANAGEMENT ---
if 'db' not in st.session_state:
    st.session_state.db = VideoDB()

if 'worker' not in st.session_state:
    st.session_state.worker = VideoWorker(st.session_state.db)

# --- 3. PROGRESS MONITOR (FRAGMENT) ---
# This is the "Holy Grail" fix. It re-runs ONLY this function every 1s.
try:
    @st.fragment(run_every=1)
    def monitor_progress():
        if st.session_state.worker.is_running:
            container = st.container()
            with container:
                st.info("⚡ Neural Engine Active")
                
                # Check Queue (Non-blocking)
                try:
                    while True:
                        signal, data = st.session_state.worker.queue.get_nowait()
                        if signal == WorkerSignals.PROGRESS:
                            curr, total, name = data
                            st.session_state['progress_float'] = curr / total
                            st.session_state['progress_text'] = f"Processing: {name} ({curr}/{total})"
                        elif signal == WorkerSignals.STATUS:
                            st.session_state['progress_text'] = data
                        elif signal == WorkerSignals.DONE:
                            st.session_state['progress_float'] = 1.0
                            st.session_state['progress_text'] = "Done!"
                            st.rerun() # Force full refresh once done to show new videos
                        elif signal == WorkerSignals.ERROR:
                            st.error(data)
                except Exception:
                    pass

                # Render State
                curr_progress = st.session_state.get('progress_float', 0.0)
                curr_text = st.session_state.get('progress_text', "Initializing...")
                st.progress(curr_progress)
                st.text(curr_text)
except AttributeError:
    # Fallback for older Streamlit versions without fragment
    # Fallback for older Streamlit versions without fragment
    def monitor_progress():
        if st.session_state.worker.is_running:
            # Poll queue once per render
            try:
                while True:
                    signal, data = st.session_state.worker.queue.get_nowait()
                    if signal == WorkerSignals.PROGRESS:
                        curr, total, name = data
                        st.session_state['progress_float'] = curr / total
                        st.session_state['progress_text'] = f"Processing: {name} ({curr}/{total})"
                    elif signal == WorkerSignals.STATUS:
                        st.session_state['progress_text'] = data
                    elif signal == WorkerSignals.DONE:
                        st.session_state['progress_float'] = 1.0
                        st.session_state['progress_text'] = "Done!"
                        st.rerun()
                    elif signal == WorkerSignals.ERROR:
                        st.error(data)
            except Exception:
                pass

            curr_progress = st.session_state.get('progress_float', 0.0)
            curr_text = st.session_state.get('progress_text', "Processing (Background)...")
            st.progress(curr_progress)
            st.text(curr_text)
            
            # Suggest update but keep working
            st.caption("⚠️ Install `streamlit>=1.37` for smoother animation.")

# --- 4. ORCHESTRATION ---

# Render Sidebar
render_sidebar(st.session_state.worker)

# Render Progress (Top)
monitor_progress()

# Main Content
tab_library, tab_analytics = st.tabs(["🎥 Library", "📊 Intelligence Analytics"])

with tab_library:
    render_gallery(st.session_state.db)

with tab_analytics:
    render_analytics(st.session_state.db)

# Inspector Overlay
check_inspector_state(st.session_state.db)

# Final Cleanup (Close DB on exit? Streamlit handles lifecycle poorly, rely on context manager usage if possible, 
# but session_state DB persists. We rely on SQLite handling connection closing on process exit or explicit close if we added a hook.)

