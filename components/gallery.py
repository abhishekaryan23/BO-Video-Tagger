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

        col_filter, col_pg = st.columns([3, 1])
        
        # Callback to reset page on filter change
        def reset_page():
            st.session_state.gallery_page = 0
            if 'selected_video' in st.session_state:
                st.session_state.selected_video = None

        with col_filter:
            # Folder Filter
            all_folders = ["All"] + db.get_unique_folders()
            selected_folder = st.selectbox("📂 Filter by Source", all_folders, on_change=reset_page)

        # Pagination State
        if 'gallery_page' not in st.session_state:
            st.session_state.gallery_page = 0
            
        PAGE_SIZE = 50
        offset = st.session_state.gallery_page * PAGE_SIZE

        # Fetch Data
        if search_q:
            videos = db.search_videos(search_q) # Search ignores pagination for now to keep it simple
            is_search = True
        else:
            videos = db.get_all_videos(limit=PAGE_SIZE, offset=offset, folder_filter=selected_folder)
            is_search = False

        if not videos and st.session_state.gallery_page == 0:
            st.warning("No videos found. Point to a folder and click 'Start Indexing'.")
        else:
            # Pagination Controls (Only show if not searching)
            if not is_search:
                with col_pg:
                    c_prev, c_next = st.columns(2)
                    if c_prev.button("◀", disabled=st.session_state.gallery_page == 0):
                        st.session_state.gallery_page -= 1
                        st.rerun()
                    if c_next.button("▶", disabled=len(videos) < PAGE_SIZE):
                        st.session_state.gallery_page += 1
                        st.rerun()

            st.markdown(f"**Showing {len(videos)} Assets**")
            
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
                    
                    # Display Logic: Show Folder name if in 'All' view to avoid confusion
                    if selected_folder == "All":
                        parent = os.path.basename(os.path.dirname(vid_path))
                        # Use Help tooltip for full path
                        st.markdown(f"**{video['filename']}**", help=vid_path)
                        st.caption(f"📁 {parent}")
                    else:
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
