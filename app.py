import html
import os
import re
from pathlib import Path

from PIL import Image
import streamlit as st

import config
from rag_agent import MAX_FILE_BYTES, SUPPORTED_EXTENSIONS, RagAgent


BASE_DIR = Path(__file__).resolve().parent
KNOWLEDGE_DIR = Path(os.getenv("RAG_KNOWLEDGE_DIR", BASE_DIR / "knowledge"))
DATA_DIR = Path(os.getenv("RAG_DATA_DIR", BASE_DIR / ".rag_data"))

st.set_page_config(page_title="Knowledge Assistant", page_icon="✨", layout="wide")

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

    :root {
        --ink: #e2e8f0;
        --muted: #94a3b8;
        --accent: #10b981;
        --accent-glow: #2dd4bf;
        --cyan-glow: #38bdf8;
    }

    /* Ambient Luxury Dark Canvas with Floating Mesh Glow */
    html, body, [data-testid="stAppViewContainer"], .stApp {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
        background: radial-gradient(circle at 10% 12%, rgba(16, 185, 129, 0.20) 0%, transparent 42%),
                    radial-gradient(circle at 90% 15%, rgba(14, 165, 233, 0.18) 0%, transparent 38%),
                    radial-gradient(circle at 50% 85%, rgba(99, 102, 241, 0.12) 0%, transparent 45%),
                    #080c0a !important;
        background-attachment: fixed !important;
        color: #e2e8f0 !important;
    }

    [data-testid="stHeader"] {
        background: transparent !important;
    }
    [data-testid="stDecoration"] {
        background-image: linear-gradient(90deg, #10b981 0%, #2dd4bf 50%, #38bdf8 100%) !important;
        height: 3px !important;
    }

    /* Glossy Hero Header Banner */
    .glossy-hero {
        display: flex;
        justify-content: space-between;
        align-items: center;
        flex-wrap: wrap;
        gap: 14px;
        padding: 22px 28px;
        margin-bottom: 22px;
        background: linear-gradient(135deg, rgba(20, 35, 30, 0.72) 0%, rgba(10, 18, 15, 0.82) 100%);
        backdrop-filter: blur(20px) saturate(180%);
        -webkit-backdrop-filter: blur(20px) saturate(180%);
        border: 1px solid rgba(255, 255, 255, 0.12);
        border-radius: 18px;
        box-shadow: inset 0 1px 1px rgba(255, 255, 255, 0.2), 0 16px 40px rgba(0, 0, 0, 0.55);
    }
    .hero-left {
        display: flex;
        align-items: center;
        gap: 18px;
    }
    .hero-icon {
        font-size: 1.85rem;
        width: 52px;
        height: 52px;
        display: flex;
        align-items: center;
        justify-content: center;
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.45) 0%, rgba(14, 165, 233, 0.35) 100%);
        border: 1px solid rgba(45, 212, 191, 0.5);
        border-radius: 14px;
        box-shadow: inset 0 1px 1px rgba(255, 255, 255, 0.3), 0 0 22px rgba(45, 212, 191, 0.28);
    }
    .hero-title {
        font-size: 1.7rem;
        font-weight: 800;
        background: linear-gradient(135deg, #ffffff 0%, #a7f3d0 55%, #38bdf8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: -0.03em;
        line-height: 1.2;
    }
    .hero-subtitle {
        font-size: 0.88rem;
        color: #94a3b8;
        margin-top: 4px;
        font-weight: 400;
        letter-spacing: 0.01em;
    }
    .hero-status-pill {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 6px 15px;
        background: rgba(16, 185, 129, 0.15);
        border: 1px solid rgba(16, 185, 129, 0.4);
        border-radius: 20px;
        font-size: 0.80rem;
        font-weight: 600;
        color: #34d399;
        box-shadow: inset 0 1px 1px rgba(255, 255, 255, 0.1);
    }
    .pulse-dot {
        width: 8px;
        height: 8px;
        background: #34d399;
        border-radius: 50%;
        box-shadow: 0 0 8px #34d399;
        display: inline-block;
        animation: pulseGlow 2s infinite;
    }
    @keyframes pulseGlow {
        0% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(52, 211, 153, 0.7); }
        70% { transform: scale(1.15); box-shadow: 0 0 0 6px rgba(52, 211, 153, 0); }
        100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(52, 211, 153, 0); }
    }

    /* Glossy Sidebar */
    [data-testid="stSidebar"] {
        background: rgba(8, 13, 11, 0.85) !important;
        backdrop-filter: blur(24px) saturate(180%) !important;
        -webkit-backdrop-filter: blur(24px) saturate(180%) !important;
        border-right: 1px solid rgba(255, 255, 255, 0.08) !important;
        box-shadow: 4px 0 30px rgba(0, 0, 0, 0.5) !important;
    }
    [data-testid="stSidebar"] * {
        color: #cbd5e1;
    }
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
        color: #f8fafc !important;
        font-weight: 700 !important;
        letter-spacing: -0.02em;
    }
    [data-testid="stSidebar"] .stCaption, [data-testid="stSidebar"] small {
        color: #94a3b8 !important;
    }

    /* Glossy Metric Cards with Specular Highlight */
    [data-testid="stMetric"] {
        background: linear-gradient(135deg, rgba(22, 36, 30, 0.75) 0%, rgba(11, 18, 15, 0.82) 100%) !important;
        backdrop-filter: blur(18px) saturate(180%) !important;
        -webkit-backdrop-filter: blur(18px) saturate(180%) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 14px !important;
        padding: 16px 20px !important;
        box-shadow: inset 0 1px 1px 0 rgba(255, 255, 255, 0.16), 0 10px 30px -8px rgba(0, 0, 0, 0.55) !important;
        transition: all 0.28s cubic-bezier(0.4, 0, 0.2, 1) !important;
        position: relative;
        overflow: hidden;
    }
    [data-testid="stMetric"]::before {
        content: "";
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.35), transparent);
    }
    [data-testid="stMetric"]:hover {
        transform: translateY(-2px) !important;
        border-color: rgba(45, 212, 191, 0.45) !important;
        box-shadow: inset 0 1px 1px 0 rgba(255, 255, 255, 0.25), 0 16px 36px -8px rgba(16, 185, 129, 0.32) !important;
    }
    [data-testid="stMetricLabel"], [data-testid="stMetricLabel"] > div {
        color: #94a3b8 !important;
        font-size: 0.80rem !important;
        text-transform: uppercase !important;
        letter-spacing: 0.06em !important;
        font-weight: 600 !important;
    }
    [data-testid="stMetricValue"], [data-testid="stMetricValue"] > div {
        color: #f8fafc !important;
        font-size: 1.75rem !important;
        font-weight: 800 !important;
        text-shadow: 0 0 20px rgba(45, 212, 191, 0.25) !important;
        font-family: 'Plus Jakarta Sans', sans-serif !important;
    }
    [data-testid="stMetricDelta"] svg {
        fill: #2dd4bf !important;
    }
    [data-testid="stMetricDelta"], [data-testid="stMetricDelta"] > div {
        color: #2dd4bf !important;
        font-weight: 600 !important;
        font-size: 0.84rem !important;
    }

    /* Glossy Chat Bubbles */
    [data-testid="stChatMessage"] {
        background: linear-gradient(135deg, rgba(18, 30, 25, 0.68) 0%, rgba(10, 16, 14, 0.75) 100%) !important;
        backdrop-filter: blur(18px) saturate(180%) !important;
        -webkit-backdrop-filter: blur(18px) saturate(180%) !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-radius: 16px !important;
        padding: 16px 22px !important;
        margin-bottom: 14px !important;
        box-shadow: inset 0 1px 1px 0 rgba(255, 255, 255, 0.12), 0 8px 30px rgba(0, 0, 0, 0.42) !important;
        transition: border-color 0.2s ease !important;
    }
    [data-testid="stChatMessage"]:hover {
        border-color: rgba(255, 255, 255, 0.15) !important;
    }
    [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] * {
        color: #e2e8f0 !important;
        line-height: 1.65 !important;
    }
    [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] strong {
        color: #f8fafc !important;
        font-weight: 700 !important;
    }

    /* Floating Glossy Chat Input */
    [data-testid="stChatInput"] {
        padding-bottom: 18px !important;
    }
    [data-testid="stChatInput"] > div {
        background: rgba(14, 22, 19, 0.88) !important;
        backdrop-filter: blur(24px) saturate(180%) !important;
        -webkit-backdrop-filter: blur(24px) saturate(180%) !important;
        border: 1px solid rgba(255, 255, 255, 0.14) !important;
        border-radius: 18px !important;
        box-shadow: inset 0 1px 1px rgba(255, 255, 255, 0.16), 0 14px 38px rgba(0, 0, 0, 0.55) !important;
        transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1) !important;
    }
    [data-testid="stChatInput"] > div:focus-within {
        border-color: rgba(45, 212, 191, 0.7) !important;
        box-shadow: inset 0 1px 1px rgba(255, 255, 255, 0.25), 0 0 32px rgba(45, 212, 191, 0.3) !important;
    }
    [data-testid="stChatInput"] textarea {
        background: transparent !important;
        color: #f8fafc !important;
        -webkit-text-fill-color: #f8fafc !important;
        caret-color: #2dd4bf !important;
        font-size: 0.95rem !important;
    }
    [data-testid="stChatInput"] textarea::placeholder {
        color: #64748b !important;
        opacity: 1 !important;
    }

    /* Glossy Gradient Buttons */
    .stButton > button, button[kind="primary"] {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%) !important;
        color: #ffffff !important;
        border: 1px solid rgba(255, 255, 255, 0.25) !important;
        border-radius: 10px !important;
        padding: 8px 18px !important;
        font-weight: 600 !important;
        font-size: 0.88rem !important;
        box-shadow: inset 0 1px 1px rgba(255, 255, 255, 0.35), 0 4px 18px rgba(16, 185, 129, 0.35) !important;
        transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
    }
    .stButton > button:hover, button[kind="primary"]:hover {
        transform: translateY(-1px) !important;
        border-color: rgba(45, 212, 191, 0.9) !important;
        box-shadow: inset 0 1px 1px rgba(255, 255, 255, 0.5), 0 8px 24px rgba(45, 212, 191, 0.45) !important;
    }

    /* Glossy Token Usage Bar & Pills */
    .glossy-token-bar {
        display: flex;
        flex-wrap: wrap;
        align-items: center;
        gap: 8px;
        margin: 12px 0 10px 0;
        padding: 10px 14px;
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.04) 0%, rgba(255, 255, 255, 0.01) 100%);
        backdrop-filter: blur(14px);
        -webkit-backdrop-filter: blur(14px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 14px;
        box-shadow: inset 0 1px 1px rgba(255, 255, 255, 0.12);
    }
    .glossy-pill {
        display: inline-flex;
        align-items: center;
        gap: 5px;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.80rem;
        font-weight: 500;
        letter-spacing: 0.01em;
        backdrop-filter: blur(8px);
        transition: all 0.2s ease;
    }
    .glossy-pill:hover {
        transform: translateY(-1px);
    }
    .pill-total {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.45) 0%, rgba(13, 82, 68, 0.55) 100%);
        color: #5eead4 !important;
        border: 1px solid rgba(45, 212, 191, 0.45);
        box-shadow: inset 0 1px 1px rgba(255, 255, 255, 0.2), 0 0 14px rgba(45, 212, 191, 0.2);
        font-weight: 700;
    }
    .pill-sub {
        background: rgba(255, 255, 255, 0.05);
        color: #cbd5e1 !important;
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: inset 0 1px 1px rgba(255, 255, 255, 0.06);
    }
    .pill-model {
        background: linear-gradient(135deg, rgba(14, 165, 233, 0.25) 0%, rgba(3, 105, 161, 0.3) 100%);
        color: #7dd3fc !important;
        border: 1px solid rgba(56, 189, 248, 0.35);
        box-shadow: inset 0 1px 1px rgba(255, 255, 255, 0.15), 0 0 12px rgba(14, 165, 233, 0.15);
    }
    .pill-id {
        background: rgba(255, 255, 255, 0.03);
        color: #94a3b8 !important;
        border: 1px solid rgba(255, 255, 255, 0.08);
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.75rem;
    }

    /* Glossy Citation Cards & Expanders */
    [data-testid="stExpander"] {
        background: linear-gradient(135deg, rgba(16, 26, 22, 0.75) 0%, rgba(10, 16, 14, 0.8) 100%) !important;
        backdrop-filter: blur(16px) !important;
        -webkit-backdrop-filter: blur(16px) !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-radius: 14px !important;
        box-shadow: inset 0 1px 1px rgba(255, 255, 255, 0.08), 0 8px 24px rgba(0, 0, 0, 0.35) !important;
        margin-top: 10px !important;
    }
    [data-testid="stExpander"] summary {
        color: #f1f5f9 !important;
        font-weight: 600 !important;
        font-size: 0.90rem !important;
        padding: 10px 14px !important;
    }
    [data-testid="stExpander"] summary:hover {
        color: #2dd4bf !important;
    }
    [data-testid="stExpander"] summary svg {
        fill: #2dd4bf !important;
    }
    .glossy-citation-card {
        background: linear-gradient(135deg, rgba(20, 32, 28, 0.7) 0%, rgba(11, 18, 15, 0.8) 100%);
        backdrop-filter: blur(14px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-left: 3px solid #2dd4bf;
        border-radius: 10px;
        padding: 12px 16px;
        margin-bottom: 12px;
        box-shadow: inset 0 1px 1px rgba(255, 255, 255, 0.08), 0 6px 20px rgba(0, 0, 0, 0.35);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .glossy-citation-card:hover {
        transform: translateX(2px);
        border-left-color: #5eead4;
        border-color: rgba(45, 212, 191, 0.25);
    }
    .glossy-citation-header {
        display: flex;
        align-items: center;
        gap: 8px;
        margin-bottom: 4px;
    }
    .citation-badge {
        background: linear-gradient(135deg, #10b981 0%, #0d9488 100%);
        color: #ffffff !important;
        font-weight: 700;
        font-size: 0.78rem;
        padding: 2px 8px;
        border-radius: 6px;
        border: 1px solid rgba(255, 255, 255, 0.2);
        box-shadow: inset 0 1px 1px rgba(255, 255, 255, 0.25);
    }
    .citation-filename {
        color: #f1f5f9 !important;
        font-weight: 600;
        font-size: 0.92rem;
    }
    .citation-chunk {
        color: #94a3b8 !important;
        font-size: 0.80rem;
        background: rgba(255, 255, 255, 0.06);
        padding: 1px 7px;
        border-radius: 4px;
    }
    .glossy-citation-meta {
        font-size: 0.76rem;
        color: #64748b !important;
        font-family: 'JetBrains Mono', monospace;
        margin-bottom: 8px;
    }
    .glossy-citation-text {
        font-size: 0.86rem;
        color: #cbd5e1 !important;
        background: rgba(0, 0, 0, 0.35);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-left: 2px solid rgba(45, 212, 191, 0.5);
        padding: 10px 14px;
        border-radius: 6px;
        line-height: 1.55;
        white-space: pre-wrap;
        max-height: 240px;
        overflow-y: auto;
        font-family: 'Plus Jakarta Sans', sans-serif;
    }

    /* File uploader & Select Dropdowns */
    [data-testid="stFileUploader"] {
        background: rgba(18, 28, 24, 0.6) !important;
        border: 1px dashed rgba(45, 212, 191, 0.35) !important;
        border-radius: 12px !important;
        padding: 12px !important;
        backdrop-filter: blur(10px) !important;
    }
    [data-testid="stFileUploader"] [data-testid="stFileUploaderDropzone"] {
        background: transparent !important;
        border: none !important;
    }
    [data-baseweb="select"] > div {
        background: rgba(18, 28, 24, 0.75) !important;
        border: 1px solid rgba(255, 255, 255, 0.12) !important;
        border-radius: 10px !important;
        color: #f8fafc !important;
    }
    [data-baseweb="popover"] {
        background: #0d1512 !important;
        border: 1px solid rgba(255, 255, 255, 0.12) !important;
        border-radius: 10px !important;
        box-shadow: 0 12px 36px rgba(0, 0, 0, 0.6) !important;
    }
    [data-baseweb="menu"] {
        background: #0d1512 !important;
    }
    [role="option"] {
        color: #e2e8f0 !important;
    }
    [role="option"]:hover, [aria-selected="true"] {
        background: rgba(16, 185, 129, 0.25) !important;
        color: #5eead4 !important;
    }

    /* Sleek Frosted Scrollbars */
    ::-webkit-scrollbar {
        width: 7px;
        height: 7px;
    }
    ::-webkit-scrollbar-track {
        background: rgba(0, 0, 0, 0.2);
    }
    ::-webkit-scrollbar-thumb {
        background: rgba(255, 255, 255, 0.15);
        border-radius: 4px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: rgba(45, 212, 191, 0.4);
    }
    hr {
        border-color: rgba(255, 255, 255, 0.08) !important;
        margin: 18px 0 !important;
    }

    /* Glossy Image Container & Gallery */
    .glossy-image-section {
        margin: 16px 0 10px 0;
        padding: 10px 16px;
        background: linear-gradient(135deg, rgba(20, 36, 30, 0.6) 0%, rgba(12, 22, 18, 0.75) 100%);
        border: 1px solid rgba(45, 212, 191, 0.28);
        border-radius: 12px;
        box-shadow: 0 4px 18px rgba(0, 0, 0, 0.35);
    }
    .glossy-image-section-header {
        display: flex;
        align-items: center;
        gap: 10px;
    }
    .gallery-icon {
        font-size: 1.15rem;
    }
    .gallery-title {
        font-weight: 700;
        font-size: 0.90rem;
        color: #f1f5f9;
        letter-spacing: -0.01em;
    }
    .gallery-count {
        margin-left: auto;
        font-size: 0.75rem;
        background: rgba(45, 212, 191, 0.18);
        color: #2dd4bf;
        border: 1px solid rgba(45, 212, 191, 0.35);
        padding: 2px 9px;
        border-radius: 12px;
        font-family: 'JetBrains Mono', monospace;
    }
    [data-testid="stImage"] {
        border-radius: 12px;
        overflow: hidden;
        transition: transform 0.22s ease, box-shadow 0.22s ease;
    }
    [data-testid="stImage"]:hover {
        transform: translateY(-2px);
    }
    [data-testid="stImage"] img {
        border-radius: 12px !important;
        border: 1px solid rgba(255, 255, 255, 0.12) !important;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.45) !important;
        background: rgba(10, 16, 14, 0.7) !important;
    }
    [data-testid="stImageCaption"] {
        font-family: 'JetBrains Mono', monospace !important;
        color: #94a3b8 !important;
        font-size: 0.80rem !important;
        margin-top: 6px !important;
    }
    .citation-img-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        font-size: 0.78rem;
        color: #2dd4bf;
        background: rgba(45, 212, 191, 0.12);
        border: 1px solid rgba(45, 212, 191, 0.25);
        padding: 3px 10px;
        border-radius: 6px;
        margin: 8px 0 6px 0;
        font-family: 'JetBrains Mono', monospace;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Loading local embedding model and vector database...")
def get_agent(
    api_key: str | None,
    persist_directory: Path,
    embedding_model: str,
    text_model: str,
    vision_model: str,
    cache_version: str = "v3_media_loaded",
) -> RagAgent:
    return RagAgent(
        api_key=api_key,
        persist_directory=persist_directory,
        embedding_model=embedding_model,
        text_model=text_model,
        vision_model=vision_model,
    )


def save_uploads(uploaded_files: list) -> list[Path]:
    KNOWLEDGE_DIR.mkdir(parents=True, exist_ok=True)
    saved_paths = []
    for uploaded_file in uploaded_files:
        filename = Path(uploaded_file.name).name
        if Path(filename).suffix.lower() not in SUPPORTED_EXTENSIONS:
            st.error(f"{filename} has an unsupported file type.")
            continue
        destination = KNOWLEDGE_DIR / filename
        if uploaded_file.size > MAX_FILE_BYTES:
            st.error(
                f"{filename} exceeds the {MAX_FILE_BYTES // (1024 * 1024)} MB file limit.")
            continue
        destination.write_bytes(uploaded_file.getbuffer())
        saved_paths.append(destination)
    return saved_paths


def show_ingestion_results(results: dict) -> None:
    for item in results.get("files", []):
        if item.get("status") == "indexed":
            st.write(f"Indexed {item['file']} ({item['chunks']} passages).")
        for warning in item.get("warnings", []):
            st.warning(f"{item['file']}: {warning}")
    for error in results.get("errors", []):
        st.error(f"{error['file']}: {error['error']}")


def resolve_message_images(
    images: list[str] | None,
    content: str,
    citations: list[dict],
) -> list[str]:
    found_images: list[str] = []

    # 1. Any images explicitly passed in the message / response
    if images:
        for img in images:
            p = Path(img)
            if p.is_file() and str(p.resolve()) not in found_images:
                found_images.append(str(p.resolve()))

    # 2. Any images in citations
    for c in citations:
        for img in c.get("images", []):
            p = Path(img)
            if p.is_file() and str(p.resolve()) not in found_images:
                found_images.append(str(p.resolve()))
        # Check citation text for explicit [image: ...] or Embedded image ...
        c_text = c.get("text", "")
        for match in re.findall(r"\[image:\s*([^\]]+)\]", c_text):
            p = Path(match.strip())
            if p.is_file() and str(p.resolve()) not in found_images:
                found_images.append(str(p.resolve()))
        c_source = Path(c.get("path", c.get("source", "")))
        for name_match in re.findall(
            r"Embedded image\s+([a-zA-Z0-9_\-\.]+\.(?:png|jpe?g|webp|gif|bmp))",
            c_text,
            re.IGNORECASE,
        ):
            candidate = DATA_DIR / "media" / c_source.name / name_match
            if candidate.is_file() and str(candidate.resolve()) not in found_images:
                found_images.append(str(candidate.resolve()))

    # 3. Check assistant content text for mentioned images (e.g. "image2.png", "image1.jpeg")
    mentioned_names = re.findall(
        r"\b([a-zA-Z0-9_\-\.]+\.(?:png|jpe?g|webp|gif|bmp))\b",
        content,
        re.IGNORECASE,
    )
    if mentioned_names:
        media_root = DATA_DIR / "media"
        if media_root.is_dir():
            for name in mentioned_names:
                for c in citations:
                    c_src = Path(c.get("path", c.get("source", ""))).name
                    cand = media_root / c_src / name
                    if cand.is_file() and str(cand.resolve()) not in found_images:
                        found_images.append(str(cand.resolve()))
                for cand in media_root.rglob(name):
                    if cand.is_file() and str(cand.resolve()) not in found_images:
                        found_images.append(str(cand.resolve()))

    return found_images


def render_images_gallery(images: list[str], title: str = "🖼️ Visual References & Extracted Media") -> None:
    if not images:
        return
    valid_images = [str(Path(img).resolve()) for img in images if Path(img).is_file()]
    if not valid_images:
        return

    st.markdown(
        f"""
        <div class="glossy-image-section">
            <div class="glossy-image-section-header">
                <span class="gallery-icon">🖼️</span>
                <span class="gallery-title">{title}</span>
                <span class="gallery-count">{len(valid_images)} item{'s' if len(valid_images) > 1 else ''}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if len(valid_images) == 1:
        img_p = valid_images[0]
        try:
            with Image.open(img_p) as pil_img:
                st.image(pil_img, caption=f"📄 {Path(img_p).name}", use_container_width=True)
        except Exception:
            st.image(str(img_p), caption=f"📄 {Path(img_p).name}", use_container_width=True)
    elif len(valid_images) == 2:
        cols = st.columns(2)
        for i, img_p in enumerate(valid_images):
            with cols[i]:
                try:
                    with Image.open(img_p) as pil_img:
                        st.image(pil_img, caption=f"📄 {Path(img_p).name}", use_container_width=True)
                except Exception:
                    st.image(str(img_p), caption=f"📄 {Path(img_p).name}", use_container_width=True)
    else:
        cols = st.columns(min(3, len(valid_images)))
        for i, img_p in enumerate(valid_images):
            with cols[i % len(cols)]:
                try:
                    with Image.open(img_p) as pil_img:
                        st.image(pil_img, caption=f"📄 {Path(img_p).name}", use_container_width=True)
                except Exception:
                    st.image(str(img_p), caption=f"📄 {Path(img_p).name}", use_container_width=True)


def render_citations_and_usage(
    citations: list[dict],
    usage: dict | None,
    query_id: str | None = None,
    model: str | None = None,
) -> None:
    if usage:
        prompt_t = usage.get("prompt_tokens", 0)
        comp_t = usage.get("completion_tokens", 0)
        total_t = usage.get("total_tokens", prompt_t + comp_t)
        model_name = model or config.TEXT_MODEL
        qid = query_id or "n/a"
        st.markdown(
            f"""
            <div class="glossy-token-bar">
                <span class="glossy-pill pill-total">⚡ {total_t:,} tokens</span>
                <span class="glossy-pill pill-sub">📥 {prompt_t:,} prompt</span>
                <span class="glossy-pill pill-sub">📤 {comp_t:,} completion</span>
                <span class="glossy-pill pill-model">🤖 {html.escape(model_name)}</span>
                <span class="glossy-pill pill-id">#{html.escape(str(qid))}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    if citations:
        with st.expander(f"📚 Citations & Referenced Sources ({len(citations)})", expanded=False):
            for citation in citations:
                c_num = citation.get("number", 1)
                c_source = citation.get("source", "Source")
                c_chunk = citation.get("chunk", 0)
                c_path = Path(citation.get("path", c_source))
                try:
                    rel_path = c_path.relative_to(BASE_DIR)
                except ValueError:
                    rel_path = c_path

                c_text = citation.get("text", "")
                text_html = ""
                if c_text:
                    escaped_text = html.escape(c_text.strip())
                    text_html = f'<div class="glossy-citation-text">{escaped_text}</div>'

                escaped_source = html.escape(str(c_source))
                escaped_path = html.escape(str(rel_path))

                st.markdown(
                    f"""
                    <div class="glossy-citation-card">
                        <div class="glossy-citation-header">
                            <span class="citation-badge">[{c_num}]</span>
                            <span class="citation-filename">{escaped_source}</span>
                            <span class="citation-chunk">Chunk {c_chunk}</span>
                        </div>
                        <div class="glossy-citation-meta">Path: {escaped_path}</div>
                        {text_html}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                # Show any images tied to this specific citation
                c_images = [img for img in citation.get("images", []) if Path(img).is_file()]
                if not c_images:
                    c_source = Path(citation.get("path", citation.get("source", "")))
                    for name_match in re.findall(
                        r"Embedded image\s+([a-zA-Z0-9_\-\.]+\.(?:png|jpe?g|webp|gif|bmp))",
                        citation.get("text", ""),
                        re.IGNORECASE,
                    ):
                        candidate = DATA_DIR / "media" / c_source.name / name_match
                        if candidate.is_file() and str(candidate.resolve()) not in c_images:
                            c_images.append(str(candidate.resolve()))

                if c_images:
                    for c_img in c_images:
                        st.markdown(
                            f"""
                            <div class="citation-img-badge">
                                <span>🖼️ Embedded visual:</span> <b>{html.escape(Path(c_img).name)}</b>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                        try:
                            with Image.open(c_img) as c_pil:
                                st.image(c_pil, width=280)
                        except Exception:
                            st.image(str(c_img), width=280)


agent = get_agent(
    config.GEMINI_API_KEY,
    DATA_DIR / "chroma",
    config.EMBEDDING_MODEL,
    config.TEXT_MODEL,
    config.VISION_MODEL,
    cache_version="v3_media_loaded",
)
KNOWLEDGE_DIR.mkdir(parents=True, exist_ok=True)

if "messages" not in st.session_state:
    st.session_state.messages = []

# Glossy Hero Header Banner
st.markdown(
    """
    <div class="glossy-hero">
        <div class="hero-left">
            <div class="hero-icon">✨</div>
            <div>
                <div class="hero-title">Knowledge Assistant</div>
                <div class="hero-subtitle">Interactive RAG • Persistent Vectors • Real-time Citations & Token Analytics</div>
            </div>
        </div>
        <div class="hero-badges">
            <span class="hero-status-pill">
                <span class="pulse-dot"></span> Online & Syncing
            </span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.subheader("Knowledge library")
    st.caption(f"Watched folder: `{KNOWLEDGE_DIR}`")
    st.caption(f"Query log: `{DATA_DIR / 'rag_queries.log'}`")
    accepted_types = sorted(extension.lstrip(".")
                            for extension in SUPPORTED_EXTENSIONS)
    uploaded_files = st.file_uploader(
        "Add knowledge files",
        type=accepted_types,
        accept_multiple_files=True,
        max_upload_size=MAX_FILE_BYTES // (1024 * 1024),
        help=f"Files are stored in the watched folder. Maximum size: {MAX_FILE_BYTES // (1024 * 1024)} MB each.",
    )
    if uploaded_files and st.button("Add selected files", type="primary", use_container_width=True):
        saved_paths = save_uploads(uploaded_files)
        if saved_paths:
            results = {"files": [], "errors": [], "indexed": 0, "unchanged": 0}
            with st.spinner("Extracting content and updating the index..."):
                for path in saved_paths:
                    try:
                        outcome = agent.sync_file(path, force=True)
                        results["files"].append({"file": path.name, **outcome})
                        results["indexed"] += outcome["status"] == "indexed"
                    except Exception as error:
                        results["errors"].append(
                            {"file": path.name, "error": str(error)})
            show_ingestion_results(results)

    source_records = agent.sources()
    if source_records:
        st.divider()
        st.caption("Indexed sources")
        selected_source = st.selectbox(
            "Choose a source to remove",
            source_records,
            format_func=lambda item: item["filename"],
            label_visibility="collapsed",
        )
        if st.button("Remove selected source", use_container_width=True):
            agent.remove_source(selected_source["source"])
            st.rerun()

    warning_sources = agent.sources_with_warnings()
    if warning_sources and st.button(
        f"Retry image extraction ({len(warning_sources)})",
        use_container_width=True,
    ):
        retry_results = {"files": [], "errors": []}
        with st.spinner("Retrying image extraction and updating vectors..."):
            for item in warning_sources:
                try:
                    outcome = agent.sync_file(Path(item["source"]), force=True)
                    retry_results["files"].append(
                        {"file": item["filename"], **outcome})
                except Exception as error:
                    retry_results["errors"].append(
                        {"file": item["filename"], "error": str(error)})
        show_ingestion_results(retry_results)

    if not config.GEMINI_API_KEY:
        st.warning(
            "Set GEMINI_API_KEY in `.env` to enable answers and image understanding.")

    # Sidebar token usage & analytics
    st.divider()
    st.subheader("⚡ Token usage")
    session_prompt = sum(
        m.get("usage", {}).get("prompt_tokens", 0)
        for m in st.session_state.messages
        if m.get("role") == "assistant" and m.get("usage")
    )
    session_completion = sum(
        m.get("usage", {}).get("completion_tokens", 0)
        for m in st.session_state.messages
        if m.get("role") == "assistant" and m.get("usage")
    )
    session_total = session_prompt + session_completion
    query_count = sum(
        1 for m in st.session_state.messages
        if m.get("role") == "assistant" and m.get("usage")
    )

    t_col1, t_col2 = st.columns(2)
    t_col1.metric("Prompt", f"{session_prompt:,}")
    t_col2.metric("Completion", f"{session_completion:,}")
    st.metric("Total tokens", f"{session_total:,}")
    st.caption(f"Queries: **{query_count}** · Model: `{config.TEXT_MODEL}`")

    if st.session_state.messages:
        if st.button("Clear chat & reset tokens", use_container_width=True):
            st.session_state.messages = []
            st.rerun()


@st.fragment(run_every="20s")
def auto_sync() -> None:
    result = agent.sync_directory(KNOWLEDGE_DIR)
    initial_file = BASE_DIR / "Knowledge.txt"
    if initial_file.is_file():
        try:
            outcome = agent.sync_file(initial_file)
            result["indexed"] += outcome["status"] == "indexed"
            result["unchanged"] += outcome["status"] == "unchanged"
            result["files"].append({"file": initial_file.name, **outcome})
        except Exception as error:
            error_result = {"file": initial_file.name, "error": str(error)}
            result["errors"].append(error_result)
            result["files"].append({"file": initial_file.name, "status": "error",
                                   "chunks": 0, "warnings": [], "error": str(error)})
    if result["indexed"]:
        st.toast(
            f"Automatically indexed {result['indexed']} new or changed file(s).")
    if result["indexed"] or result["errors"] or any(item.get("warnings") for item in result["files"]):
        with st.expander("Latest ingestion results", expanded=bool(result["errors"])):
            show_ingestion_results(result)
    st.caption(
        f"Folder scan complete. {len(agent.sources())} source(s) indexed.")


auto_sync()

source_count = len(agent.sources())
chunk_count = agent.collection.count()

session_prompt_tokens = sum(
    m.get("usage", {}).get("prompt_tokens", 0)
    for m in st.session_state.messages
    if m.get("role") == "assistant" and m.get("usage")
)
session_completion_tokens = sum(
    m.get("usage", {}).get("completion_tokens", 0)
    for m in st.session_state.messages
    if m.get("role") == "assistant" and m.get("usage")
)
session_total_tokens = session_prompt_tokens + session_completion_tokens

last_assistant_msg = next(
    (m for m in reversed(st.session_state.messages)
     if m.get("role") == "assistant" and m.get("usage")),
    None,
)
last_query_tokens = (
    last_assistant_msg.get("usage", {}).get("total_tokens", 0)
    if last_assistant_msg
    else 0
)

metric_columns = st.columns(4)
metric_columns[0].metric("Sources", source_count)
metric_columns[1].metric("Searchable passages", chunk_count)
metric_columns[2].metric(
    "Session tokens",
    f"{session_total_tokens:,}",
    delta=f"+{last_query_tokens:,} last query" if last_query_tokens else None,
)
metric_columns[3].metric("Folder sync", "Every 20 sec")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant":
            imgs_to_show = resolve_message_images(
                message.get("images"),
                message.get("content", ""),
                message.get("citations", []),
            )
            if imgs_to_show:
                render_images_gallery(imgs_to_show)
            render_citations_and_usage(
                message.get("citations", []),
                message.get("usage"),
                message.get("query_id"),
                message.get("model", config.TEXT_MODEL),
            )

question = st.chat_input("Ask a question about your knowledge files")
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        with st.spinner("Searching and preparing an answer..."):
            try:
                response = agent.answer(question)
                st.markdown(response["answer"])
                imgs_to_show = resolve_message_images(
                    response.get("images"),
                    response.get("answer", ""),
                    response.get("citations", []),
                )
                if imgs_to_show:
                    render_images_gallery(imgs_to_show)
                render_citations_and_usage(
                    response.get("citations", []),
                    response.get("usage"),
                    response.get("query_id"),
                    response.get("model", config.TEXT_MODEL),
                )
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": response["answer"],
                        "images": imgs_to_show,
                        "citations": response.get("citations", []),
                        "usage": response.get("usage"),
                        "query_id": response.get("query_id"),
                        "model": response.get("model", config.TEXT_MODEL),
                    }
                )
                st.rerun()
            except Exception as error:
                st.error(f"The request failed: {error}")
