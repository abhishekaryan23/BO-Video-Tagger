import streamlit as st
import os
from bo_config import Settings
from bo_worker import VideoWorker

def render_sidebar(worker: VideoWorker):
    """Renders the Control Deck sidebar."""
    with st.sidebar:
        if os.path.exists("assets/logo.png"):
            st.image("assets/logo.png", width=150)
        else:
            st.title("BO-View")
        
        st.markdown("### Control Deck")
        
        st.header("⚙️ Controls")
        
        def close_inspector():
            if 'selected_video' in st.session_state:
                st.session_state.selected_video = None

        # Tier Selection
        from bo_video_tagger import MODEL_TIERS
        tier_choice = st.radio(
            "Intelligence Tier", 
            options=list(MODEL_TIERS.keys()),
            format_func=lambda x: f"{x.upper()} - {MODEL_TIERS[x].desc} (RAM >{MODEL_TIERS[x].min_ram_gb}GB)",
            help="Select the AI model capability.",
            on_change=close_inspector
        )
        
        # Path Selection
        target_dir = st.text_input("Target Folder", value="/Users/public", help="Absolute path to video folder.", on_change=close_inspector)
        
        st.divider()
        
        col1, col2 = st.columns(2)
        interval = col1.number_input("Interval (s)", value=10, min_value=1, on_change=close_inspector)
        debug_mode = col2.checkbox("Debug Frames", value=False, on_change=close_inspector)
        
        force_reprocess = st.checkbox("Force Reprocess All (Slow)", value=False, help="If checked, will re-analyze ALL videos even if already indexed.", on_change=close_inspector)
        
        # Action Button
        if st.button("Start Indexing", type="primary", width="stretch"):
            if not os.path.isdir(target_dir):
                st.error("Invalid Directory!")
            else:
                worker.start_processing(tier_choice, target_dir, interval, debug_mode, force_reprocess)
                st.rerun()

        # Progress Monitor Placeholder
        # The main app orchestrator handles the placement, or we can assume it's here.
        # But per plan, we might separate progress. Ideally sidebar contains controls.
        st.markdown("---")
