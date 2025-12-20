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
        
        # Tier Selection
        from bo_video_tagger import MODEL_TIERS
        tier_choice = st.radio(
            "Intelligence Tier", 
            list(MODEL_TIERS.keys()), 
            format_func=lambda x: f"{x.upper()} - {MODEL_TIERS[x].desc}"
        )
        
        # Path Selection
        default_path = os.path.expanduser("~/Movies" if os.path.exists(os.path.expanduser("~/Movies")) else ".")
        target_dir = st.text_input("Target Directory", value=default_path)
        
        col1, col2 = st.columns(2)
        interval = col1.number_input("Interval (s)", value=10, min_value=1)
        debug_mode = col2.checkbox("Debug Frames", value=False)
        
        # Action Button
        if st.button("Start Indexing", type="primary", width="stretch"):
            if not os.path.isdir(target_dir):
                st.error("Invalid Directory!")
            else:
                worker.start_processing(tier_choice, target_dir, interval, debug_mode)
                st.rerun()

        # Progress Monitor Placeholder
        # The main app orchestrator handles the placement, or we can assume it's here.
        # But per plan, we might separate progress. Ideally sidebar contains controls.
        st.markdown("---")
