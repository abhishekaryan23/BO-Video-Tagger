import streamlit as st
import pandas as pd
import altair as alt
import json
import os
from bo_db import VideoDB

@st.cache_data(ttl=60)
def fetch_analytics_data(_db):
    """Cached data fetcher. _db is unhashed/underscored to exclude from cache hash if needed, 
    but for this simple case we assume DB is stable-ish or ttl handles it."""
    # Note: Streamlit can't hash SQLite connection objects easily.
    # Better pattern: Pass the method result? Or use a singleton DB?
    # We will assume _db is passed for method access.
    return _db.get_analytics_df()

def render_analytics(db: VideoDB):
    """Renders the Analytics Dashboard."""
    try:
        df = fetch_analytics_data(db)
        
        if df.empty:
            st.info("Index some videos to see intelligence analytics.")
            return

        # --- DATA PROCESSING ---
        # 1. Folder Extraction
        df['folder'] = df['path'].apply(lambda p: os.path.basename(os.path.dirname(p)))
        
        # 2. Extract AI Time safely
        def get_ai_time(meta_str):
            try:
                data = json.loads(meta_str)
                return data.get('system', {}).get('processing_time_sec', 0)
            except:
                return 0
        df['ai_time'] = df['metadata'].apply(get_ai_time)

        # --- KPI ROW ---
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Total Assets", len(df))
        k2.metric("Total Storage", f"{df['size_mb'].sum() / 1024:.2f} GB")
        k3.metric("Total Duration", f"{df['duration_sec'].sum() / 3600:.1f} Hrs")
        k4.metric("Avg AI Speed", f"{df['ai_time'].mean():.1f} s/vid")
        
        st.markdown("---")
        
        # --- CHARTS ---
        c1, c2 = st.columns(2)
        
        with c1:
            st.markdown("### 📂 Storage by Folder")
            # Group by folder
            folder_stats = df.groupby('folder')['size_mb'].sum().reset_index()
            
            base = alt.Chart(folder_stats).encode(
                theta=alt.Theta("size_mb", stack=True),
                color=alt.Color("folder", legend=None),
                tooltip=["folder", "size_mb"]
            )
            pie = base.mark_arc(outerRadius=120)
            text = base.mark_text(radius=140).encode(
                text="folder",
                order=alt.Order("size_mb", sort="descending") 
            )
            st.altair_chart(pie + text, use_container_width=True, theme="streamlit")
            
        with c2:
            st.markdown("### 🏷️ Tag Ecosystem (Top 20)")
            # Explosion Logic for Tags
            # Handle empty tags first
            tag_df = df.assign(tags=df['tags'].str.astype(str).str.split(',')).explode('tags')
            tag_df['tags'] = tag_df['tags'].str.strip()
            tag_df = tag_df[tag_df['tags'] != ''] # Cleanup
            tag_df = tag_df[tag_df['tags'] != 'nan'] # Cleanup string nan
            tag_df = tag_df[tag_df['tags'].notna()]
            
            top_tags = tag_df['tags'].value_counts().head(20).reset_index()
            top_tags.columns = ['tag', 'count']
            
            bars = alt.Chart(top_tags).mark_bar().encode(
                x='count:Q',
                y=alt.Y('tag:N', sort='-x'),
                color=alt.Color('count:Q', scale=alt.Scale(scheme='greens')),
                tooltip=['tag', 'count']
            ).properties(height=300)
            st.altair_chart(bars, use_container_width=True, theme="streamlit")
        
        st.markdown("### 🚀 Processing Health")
        scatter = alt.Chart(df).mark_circle(size=60).encode(
            x=alt.X('duration_sec', title='Video Duration (s)'),
            y=alt.Y('ai_time', title='Processing Time (s)'),
            color=alt.Color('folder', legend=alt.Legend(title="Folder")),
            tooltip=['path', 'duration_sec', 'ai_time']
        ).interactive()
        st.altair_chart(scatter, use_container_width=True, theme="streamlit")
        
    except Exception as e:
        st.error(f"Analytics Error: {e}")
        import logging
        logging.error(f"Analytics Component Failed: {e}", exc_info=True)
