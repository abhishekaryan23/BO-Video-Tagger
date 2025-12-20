import streamlit as st
import os
import hashlib
import html
from bo_config import Settings
from bo_db import VideoDB

def render_gallery(db: VideoDB):
    """Renders the Video Gallery Grid."""
    try:
        # B. GALLERY & SEARCH
        def clear_inspector():
            if 'selected_video' in st.session_state:
                st.session_state.selected_video = None

        search_q = st.text_input("Search Library", placeholder="Type to search objects, tags, or descriptions...", label_visibility="collapsed", on_change=clear_inspector)

        # Fetch Data
        if search_q:
            videos = db.search_videos(search_q)
        else:
            videos = db.get_all_videos(limit=100)

        if not videos:
            st.warning("No videos found. Point to a folder and click 'Start Indexing'.")
        else:
            st.markdown(f"**{len(videos)} Assets Found**")
            
            # Masonry Grid Logic
            cols = st.columns(Settings.GRID_COLUMNS)
            
            for idx, video in enumerate(videos):
                col = cols[idx % Settings.GRID_COLUMNS]
                
                # Thumbnail Path logic
                vid_path = video['path']
                vid_hash = hashlib.md5(vid_path.encode()).hexdigest()
                thumb_path = os.path.join(Settings.THUMBS_DIR, f"{vid_hash}.jpg")
                
                # Fallback to placeholder if not exists
                real_thumb = thumb_path if os.path.exists(thumb_path) else "https://placehold.co/600x400/1e2127/FFF?text=Processing..."
                
                with col:
                    st.image(real_thumb, width="stretch")
                    st.markdown(f"**{video['filename']}**")
                    
                    # Tags (XSS FIXED)
                    tags = video['tags'].split(",")[:3] if video['tags'] else []
                    # Escape HTML to prevent injection
                    tag_html = "".join([f"<span class='tag-pill'>{html.escape(t.strip())}</span>" for t in tags])
                    st.markdown(tag_html, unsafe_allow_html=True)
                    
                    if st.button("Inspect", key=f"btn_{idx}"):
                        st.session_state.selected_video = video
                        st.rerun()
    except Exception as e:
        st.error(f"Gallery Component Error: {e}")
        import logging
        logging.error(f"Gallery Error: {e}")
