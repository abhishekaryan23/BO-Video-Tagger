import streamlit as st
import os
import time
from bo_db import VideoDB

@st.dialog("Inspect Video", width="large")
def show_inspector_dialog(vid: dict, db: VideoDB):
    """Renders the Inspector Dialog with File Path Playback."""
    try:
        col_player, col_meta = st.columns([1.5, 1])
        
        with col_player:
            # Video Player
            # FIX: Use path directly, do NOT read bytes into RAM
            try:
                if os.path.exists(vid['path']):
                    st.markdown(f"**{os.path.basename(vid['path'])}**")
                    st.video(vid['path'])
                else:
                    st.error("File not found on disk.")
            except Exception as e:
                st.error(f"Playback Error: {e}")
        
        with col_meta:
            st.markdown("### Metadata")
            
            # Editable Form
            new_desc = st.text_area("Description", value=vid['description'], height=200)
            new_tags_str = st.text_input("Tags (comma separated)", value=vid['tags'])
            
            if st.button("Save Changes", type="primary"):
                clean_tags = [t.strip() for t in new_tags_str.split(",") if t.strip()]
                db.update_metadata(vid['path'], new_desc, clean_tags)
                st.success("Saved!")
                time.sleep(0.5)
                st.session_state.selected_video = None # Close
                st.rerun()
                
    except Exception as e:
        st.error(f"Inspector Error: {e}")

def check_inspector_state(db: VideoDB):
    """Checks session state and opens dialog if needed."""
    if 'selected_video' in st.session_state and st.session_state.selected_video:
        # We need to render the dialog function
        # Note: st.dialog functions must be called to open.
        # But we defined it as a decorated function above.
        # The pattern is: call the decorated function.
        show_inspector_dialog(st.session_state.selected_video, db)
