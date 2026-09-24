# ===== STREAMLIT USER INTERFACE =====

import os
import sys
import streamlit as st

# Set page config with wide layout and custom title
st.set_page_config(
    page_title="Multilingual Hybrid YouTube Video Summarizer",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Ensure src directory is in sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(BASE_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

# Import the unified pipeline orchestrator
from pipeline import run_hybrid_pipeline

# Custom CSS for modern glassmorphism aesthetic and typography
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&family=Inter:wght@400;500;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Outfit', 'Inter', sans-serif;
    }

    .main-header {
        background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 50%, #06b6d4 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.4rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }

    .sub-header {
        color: #94a3b8;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }

    .glass-card {
        background: rgba(30, 41, 59, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 1.25rem;
        margin-bottom: 1.25rem;
        backdrop-filter: blur(10px);
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.2);
    }

    .timestamp-badge {
        background-color: #2563eb;
        color: #ffffff;
        padding: 0.25rem 0.6rem;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.85rem;
        margin-right: 0.5rem;
    }

    .entity-tag {
        display: inline-block;
        background: rgba(59, 130, 246, 0.15);
        color: #60a5fa;
        border: 1px solid rgba(59, 130, 246, 0.3);
        border-radius: 20px;
        padding: 0.2rem 0.7rem;
        font-size: 0.85rem;
        margin: 0.2rem;
    }
</style>
""", unsafe_allow_html=True)


def main():
    # Header Banner
    st.markdown('<div class="main-header">🎬 Multilingual Hybrid YouTube Summarizer</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Classical NLP extraction pipeline (Part A) coupled with '
        'deep pretrained neural models (Part B) for English & Hindi videos.</div>',
        unsafe_allow_html=True
    )

    # Sidebar Controls
    with st.sidebar:
        st.header("⚙️ Configuration")

        target_lang = st.selectbox(
            "Output Language",
            options=["English", "Hindi", "French (Bonus)", "German (Bonus)"],
            index=0,
            help="Language in which the final summary will be presented."
        )
        lang_code_map = {
            "English": "en",
            "Hindi": "hi",
            "French (Bonus)": "fr",
            "German (Bonus)": "de"
        }
        chosen_lang_code = lang_code_map[target_lang]

        length_option = st.selectbox(
            "Summary Length",
            options=["Small (~100 words)", "Medium (~250 words)", "Long (~500 words)"],
            index=1
        )
        length_key_map = {
            "Small (~100 words)": "small",
            "Medium (~250 words)": "medium",
            "Long (~500 words)": "long"
        }
        chosen_length_key = length_key_map[length_option]

        mode = st.radio(
            "Summarization Engine Mode",
            options=["Fast Mode (Single Model)", "Consensus Mode (Multi-Model Fusion)"],
            index=0,
            help="Fast mode uses T5-Small; Consensus mode runs multi-model agreement voting."
        )
        is_fast = (mode == "Fast Mode (Single Model)")

        st.markdown("---")
        st.caption("Developed as an end-to-end NLP Capstone Project.")

    # Main Input Card
    with st.container():
        st.subheader("1. Enter Video Source")
        col_input, col_demos = st.columns([3, 2])

        if "video_input_val" not in st.session_state:
            st.session_state["video_input_val"] = "en_short_01"

        with col_input:
            video_input = st.text_input(
                "YouTube Video URL or Demo ID:",
                value=st.session_state["video_input_val"],
                placeholder="https://www.youtube.com/watch?v=... or en_short_01",
                help="Paste any YouTube video link or use one of the offline demo samples below."
            )

        with col_demos:
            st.write("Or pick a pre-cached offline sample:")
            demo_cols = st.columns(3)
            if demo_cols[0].button("EN Short (3m)"):
                st.session_state["video_input_val"] = "en_short_01"
                st.rerun()
            if demo_cols[1].button("EN Lecture"):
                st.session_state["video_input_val"] = "en_medium_01"
                st.rerun()
            if demo_cols[2].button("HI Talk (5m)"):
                st.session_state["video_input_val"] = "hi_clean_01"
                st.rerun()

        summarize_clicked = st.button("🚀 Summarize Video", type="primary", use_container_width=True)

    # Process and Output
    if summarize_clicked and video_input:
        with st.spinner("Executing 15-step classical extraction & neural summarization pipeline..."):
            try:
                # Run the pipeline with verbose=False for web performance
                results = run_hybrid_pipeline(
                    video_input=video_input.strip(),
                    target_language=chosen_lang_code,
                    summary_length=chosen_length_key,
                    fast_mode=is_fast,
                    verbose=False
                )

                st.success("✅ Summary Generated Successfully!")

                video_id = results.get("video_id", "")
                base_yt_url = f"https://www.youtube.com/watch?v={video_id}" if video_id and not video_id.startswith("en_") and not video_id.startswith("hi_") else ""

                # 1. TITLE ROW: video title, duration, detected language
                st.markdown("---")
                st.markdown(f"### 🎬 {results.get('video_title', 'Video Summary')}")
                t_col1, t_col2, t_col3, t_col4 = st.columns(4)
                t_col1.metric("Duration", results.get("duration_formatted", "N/A"))
                t_col2.metric("Detected Language", results.get("detected_language", "en").upper())
                t_col3.metric("Output Language", results.get("target_language", "en").upper())
                t_col4.metric("Word Count", f"{results.get('word_count', 0)} words")

                st.markdown("---")

                # 2. TL;DR: 2-3 sentences that cover the whole video
                st.markdown("### 📌 TL;DR")
                st.info(results.get("tldr", ""))

                # 3. CHAPTERS (the main part): One block per chapter in time order
                if chosen_length_key != "small" and results.get("chapters_summary"):
                    st.markdown("### 📑 Chapters & Discussion")
                    for ch in results.get("chapters_summary", []):
                        timestamp_str = ch["timestamp"]
                        secs = ch.get("seconds", 0)
                        yt_link = f"{base_yt_url}&t={secs}" if base_yt_url else f"#{timestamp_str}"
                        title_str = ch["title"]

                        # Clickable timestamp link in Title Case
                        ch_header = f"<a href='{yt_link}' target='_blank' style='text-decoration:none; color:#3b82f6; font-weight:700;'>[{timestamp_str}]</a> **{title_str}**"
                        st.markdown(ch_header, unsafe_allow_html=True)

                        if chosen_length_key == "long":
                            st.write(ch.get("paragraph", ""))
                        else:
                            for bullet in ch.get("bullets", []):
                                st.markdown(f"- {bullet}")
                        st.write("")

                # 4. KEY TAKEAWAYS: 5-8 bullets, each starting with its timestamp [MM:SS]
                st.markdown("### 💡 Key Takeaways")
                for tk in results.get("key_takeaways", []):
                    timestamp_str = tk["timestamp"]
                    secs = tk.get("seconds", 0)
                    yt_link = f"{base_yt_url}&t={secs}" if base_yt_url else f"#{timestamp_str}"
                    takeaway_line = f"<a href='{yt_link}' target='_blank' style='text-decoration:none; color:#2563eb; font-weight:600;'>[{timestamp_str}]</a> {tk['text']}"
                    st.markdown(f"- {takeaway_line}", unsafe_allow_html=True)

                st.markdown("---")

                # 5. KEYWORDS AND ENTITIES
                kw_col, ent_col = st.columns(2)
                with kw_col:
                    st.markdown("### 🏷️ Top Keywords")
                    if results.get("top_keywords"):
                        kw_tags = "".join([f"<span class='entity-tag'>#{kw}</span>" for kw, _ in results["top_keywords"][:12]])
                        st.markdown(kw_tags, unsafe_allow_html=True)
                    else:
                        st.write("No dominant keywords extracted.")

                with ent_col:
                    st.markdown("### 👥 Named Entities")
                    raw_entities = results.get("entities", {})
                    # Group as People, Organizations, Places
                    grouped = {
                        "People": raw_entities.get("PERSON", []),
                        "Organizations": raw_entities.get("ORG", []) + raw_entities.get("ORGANIZATION", []),
                        "Places": raw_entities.get("GPE", []) + raw_entities.get("LOC", []) + raw_entities.get("LOCATION", [])
                    }

                    has_any_entity = False
                    for group_name, items in grouped.items():
                        unique_items = list(dict.fromkeys(items))
                        if unique_items:
                            has_any_entity = True
                            st.markdown(f"**{group_name}**:")
                            tags_html = "".join([f"<span class='entity-tag'>{e}</span>" for e in unique_items[:8]])
                            st.markdown(tags_html, unsafe_allow_html=True)

                    if not has_any_entity:
                        st.write("No distinct named entities detected.")

                st.markdown("---")

                # 6. DOWNLOAD: Summary as .txt and .pdf
                st.markdown("### 💾 Download Summary")
                d_col1, d_col2 = st.columns(2)

                # TXT Report
                txt_summary_lines = [
                    f"TITLE: {results.get('video_title')}",
                    f"DURATION: {results.get('duration_formatted')} | DETECTED LANGUAGE: {results.get('detected_language', 'en').upper()} | OUTPUT: {results.get('target_language', 'en').upper()}",
                    "=" * 60,
                    "\nTL;DR:",
                    results.get("tldr", ""),
                    "\n" + "=" * 60
                ]

                if chosen_length_key != "small" and results.get("chapters_summary"):
                    txt_summary_lines.append("\nCHAPTERS:")
                    for ch in results.get("chapters_summary", []):
                        txt_summary_lines.append(f"\n[{ch['timestamp']}] {ch['title']}")
                        if chosen_length_key == "long":
                            txt_summary_lines.append(f"  {ch.get('paragraph', '')}")
                        else:
                            for b in ch.get("bullets", []):
                                txt_summary_lines.append(f"  - {b}")

                txt_summary_lines.append("\n" + "=" * 60)
                txt_summary_lines.append("\nKEY TAKEAWAYS:")
                for tk in results.get("key_takeaways", []):
                    txt_summary_lines.append(f"- [{tk['timestamp']}] {tk['text']}")

                txt_content = "\n".join(txt_summary_lines)

                with d_col1:
                    st.download_button(
                        label="📄 Download Summary (.txt)",
                        data=txt_content.encode("utf-8"),
                        file_name=f"summary_{video_id if video_id else 'video'}.txt",
                        mime="text/plain",
                        use_container_width=True
                    )

                # PDF Report
                def build_pdf_bytes():
                    from fpdf import FPDF
                    pdf = FPDF()
                    pdf.add_page()
                    pdf.set_auto_page_break(auto=True, margin=15)
                    pdf.set_font("Helvetica", "B", 15)
                    safe_title = results.get("video_title", "Video Summary").encode("latin-1", "replace").decode("latin-1")
                    pdf.multi_cell(0, 8, safe_title)
                    pdf.ln(2)

                    pdf.set_font("Helvetica", "I", 10)
                    meta_str = f"Duration: {results.get('duration_formatted')} | Detected Lang: {results.get('detected_language', 'en').upper()} | Mode: {chosen_length_key.title()}"
                    pdf.cell(0, 6, meta_str.encode("latin-1", "replace").decode("latin-1"), ln=True)
                    pdf.ln(3)

                    # TL;DR
                    pdf.set_font("Helvetica", "B", 12)
                    pdf.cell(0, 6, "TL;DR", ln=True)
                    pdf.set_font("Helvetica", "", 10)
                    safe_tldr = results.get("tldr", "").encode("latin-1", "replace").decode("latin-1")
                    pdf.multi_cell(0, 5, safe_tldr)
                    pdf.ln(3)

                    # Chapters
                    if chosen_length_key != "small" and results.get("chapters_summary"):
                        pdf.set_font("Helvetica", "B", 12)
                        pdf.cell(0, 6, "Chapters", ln=True)
                        for ch in results.get("chapters_summary", []):
                            pdf.set_font("Helvetica", "B", 10)
                            ch_line = f"[{ch['timestamp']}] {ch['title']}".encode("latin-1", "replace").decode("latin-1")
                            pdf.cell(0, 5, ch_line, ln=True)
                            pdf.set_font("Helvetica", "", 9)
                            if chosen_length_key == "long":
                                pdf.multi_cell(0, 5, ch.get("paragraph", "").encode("latin-1", "replace").decode("latin-1"))
                            else:
                                for b in ch.get("bullets", []):
                                    pdf.multi_cell(0, 5, f"  - {b}".encode("latin-1", "replace").decode("latin-1"))
                            pdf.ln(1)

                    # Key Takeaways
                    pdf.set_font("Helvetica", "B", 12)
                    pdf.cell(0, 6, "Key Takeaways", ln=True)
                    pdf.set_font("Helvetica", "", 9)
                    for tk in results.get("key_takeaways", []):
                        tk_line = f"- [{tk['timestamp']}] {tk['text']}".encode("latin-1", "replace").decode("latin-1")
                        pdf.multi_cell(0, 5, tk_line)

                    return pdf.output(dest="S").encode("latin-1", errors="replace")

                with d_col2:
                    try:
                        pdf_data = build_pdf_bytes()
                        st.download_button(
                            label="📕 Download Summary (.pdf)",
                            data=pdf_data,
                            file_name=f"summary_{video_id if video_id else 'video'}.pdf",
                            mime="application/pdf",
                            use_container_width=True
                        )
                    except Exception as pdf_err:
                        st.caption(f"PDF generator notice: {pdf_err}")

                st.markdown("---")

                # 7. "Compare methods" expander, collapsed by default
                with st.expander("🔍 Compare methods (Hybrid / NLP Pipeline Only / Models Only)", expanded=False):
                    c_tab1, c_tab2, c_tab3 = st.tabs([
                        "✨ Hybrid (Part A + B)",
                        "📑 NLP Pipeline Only (Part A Extractive)",
                        "🤖 Models Only (Direct Neural Baseline)"
                    ])
                    with c_tab1:
                        st.markdown("**Hybrid Result (Primary Output):**")
                        st.info(results.get("hybrid_summary", ""))
                    with c_tab2:
                        st.markdown("**Part A Classical Extractive Baseline:**")
                        st.write(results.get("extractive_summary", ""))
                    with c_tab3:
                        st.markdown("**Part B Direct Neural Baseline:**")
                        st.write(results.get("model_only_summary", ""))

            except Exception as e:
                st.error(f"❌ Error processing video: {str(e)}")
                st.info("Tip: Try using one of the pre-cached demo IDs (e.g., `en_short_01` or `hi_clean_01`) if offline.")


if __name__ == "__main__":
    main()
