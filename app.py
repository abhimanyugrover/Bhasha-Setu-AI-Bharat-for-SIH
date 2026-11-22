"""
Bhasha Setu v5 — Any-to-Any Multilingual AI Video Dubbing
Author: Abhimanyu | J.C. Bose University YMCA, Faridabad
Multilingual AI Video Dubbing Platform
"""

import os, json, tempfile, subprocess, logging, html, requests
from datetime import datetime
from typing import Optional

# Ensure FFmpeg is available on PATH
try:
    import static_ffmpeg
    static_ffmpeg.add_paths()
except Exception:
    pass

_custom_ffmpeg = os.environ.get("FFMPEG_PATH")
if _custom_ffmpeg and os.path.exists(_custom_ffmpeg) and _custom_ffmpeg not in os.environ.get("PATH", ""):
    os.environ["PATH"] = _custom_ffmpeg + os.pathsep + os.environ.get("PATH", "")

import streamlit as st

from auth import (
    init_db,
    get_current_user,
    logout_user,
    render_login_page,
    save_user_history,
    get_user_history,
    clear_user_history,
    check_rate_limit,
    log_audit_event,
    track_event,
    get_analytics_summary,
)
from pipeline.concurrency import get_queue_status

# ── Localization ──────────────────────────────────────────────────────────────
try:
    from i18n import t, UI_LANGUAGES, get_ui_lang
    I18N_READY = True
except ImportError:
    I18N_READY = False
    UI_LANGUAGES = {"English": "en"}
    def t(key: str, **kwargs) -> str: return key
    def get_ui_lang() -> str: return "en"

log = logging.getLogger(__name__)

try:
    from llm_summarizer import summarize_transcript
except ImportError:
    summarize_transcript = None

# Bridge st.secrets to os.environ (AWS credentials, GROQ_API_KEY, S3_BUCKET, etc.)
try:
    for _sec_key, _sec_val in st.secrets.items():
        if isinstance(_sec_val, str) and _sec_val.strip():
            os.environ[_sec_key] = _sec_val.strip()
except Exception:
    pass

st.set_page_config(
    page_title="Bhasha Setu — AI Bharat Multilingual Video Dubbing",
    page_icon="🪷",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── SEO, Social Meta & Enforce HTTPS ──────────────────────────────────────────
st.markdown(
    """
    <head>
      <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=5.0">
      <meta name="description" content="Production-Grade Multilingual AI Video Dubbing platform supporting 12+ Indian regional languages powered by Sarvam AI, AWS Polly, and Edge-TTS.">
      <meta name="keywords" content="AI video dubbing, Sarvam AI, Bharat, Indic languages, speech synthesis, Bhasha Setu, multilingual">
      <meta property="og:title" content="Bhasha Setu — AI Bharat Multilingual Video Dubbing">
      <meta property="og:description" content="Dub and translate videos across 12+ Indian regional languages with neural speech synthesis and high-fidelity lip-sync.">
      <meta property="og:type" content="website">
      <meta property="og:url" content="https://bhashasetu.ai">
      <meta property="og:image" content="https://bhashasetu.ai/assets/social-preview.png">
      <meta property="og:image:width" content="1200">
      <meta property="og:image:height" content="630">
      <meta name="twitter:card" content="summary_large_image">
      <meta name="twitter:title" content="Bhasha Setu — AI Bharat Multilingual Video Dubbing">
      <meta name="twitter:description" content="Production-Grade Multilingual AI Video Dubbing platform supporting 12+ Indian regional languages.">
      <meta name="twitter:image" content="https://bhashasetu.ai/assets/social-preview.png">
      <meta name="theme-color" content="#FF6B35">
      <link rel="icon" href="data:image/svg+xml,<svg xmlns=%22http://www.w3.org/2000/svg%22 viewBox=%220 0 100 100%22><text y=%22.9em%22 font-size=%2290%22>🪷</text></svg>">
      <meta http-equiv="Content-Security-Policy" content="upgrade-insecure-requests">
    </head>
    """,
    unsafe_allow_html=True,
)

# ── Pipeline & AI Model imports ───────────────────────────────────────────────
try:
    from pipeline.config import LANGUAGES, OUTPUT_DIR, SOURCE_LANGUAGES
    from pipeline.main   import run_pipeline, run_transcribe_and_translate, run_tts_and_mux
    from pipeline.downloader import get_video_info
    from pipeline.sarvam import is_sarvam_available, SARVAM_VOICES, SARVAM_LANGUAGES
    PIPELINE_READY = True
except ImportError:
    PIPELINE_READY = False
    get_video_info = None
    is_sarvam_available = lambda: False
    SARVAM_VOICES = {}
    SARVAM_LANGUAGES = {}
    SOURCE_LANGUAGES = {
        "Auto-detect": None, "English": "en-US", "Hindi": "hi-IN",
        "Tamil": "ta-IN", "Telugu": "te-IN", "Kannada": "kn-IN",
        "Malayalam": "ml-IN", "French": "fr-FR", "German": "de-DE",
        "Spanish": "es-US", "Japanese": "ja-JP",
    }
    LANGUAGES = {
        "Hindi":     {"flag":"IN","native_name":"हिन्दी",   "tts":"polly"},
        "Tamil":     {"flag":"IN","native_name":"தமிழ்",    "tts":"edge"},
        "Telugu":    {"flag":"IN","native_name":"తెలుగు",   "tts":"edge"},
        "Kannada":   {"flag":"IN","native_name":"ಕನ್ನಡ",   "tts":"edge"},
        "Malayalam": {"flag":"IN","native_name":"മലയാളം",  "tts":"edge"},
        "Bengali":   {"flag":"IN","native_name":"বাংলা",    "tts":"edge"},
        "Marathi":   {"flag":"IN","native_name":"मराठी",    "tts":"edge"},
        "Gujarati":  {"flag":"IN","native_name":"ગુજરાતી", "tts":"edge"},
        "Punjabi":   {"flag":"IN","native_name":"ਪੰਜਾਬੀ",  "tts":"edge"},
        "Urdu":      {"flag":"IN","native_name":"اردو",     "tts":"edge"},
        "Odia":      {"flag":"IN","native_name":"ଓଡ଼ିଆ",    "tts":"gtts"},
        "Assamese":  {"flag":"IN","native_name":"অসমীয়া",  "tts":"gtts"},
    }
    OUTPUT_DIR = "output"

try:
    from deep_translator import GoogleTranslator as _GT
    TRANSLATOR_READY = True
except ImportError:
    TRANSLATOR_READY = False

# ── Constants ──────────────────────────────────────────────────────────────────
LANG_CODES = {
    "Auto-detect":"auto",
    "English":"en","Hindi":"hi","Tamil":"ta","Telugu":"te","Kannada":"kn",
    "Malayalam":"ml","Bengali":"bn","Marathi":"mr","Gujarati":"gu",
    "Punjabi":"pa","Urdu":"ur","Odia":"or","Assamese":"as",
    "French":"fr","German":"de","Spanish":"es","Japanese":"ja",
    "Chinese":"zh-CN","Arabic":"ar","Russian":"ru","Portuguese":"pt",
}

LANG_COLOR = {
    "Hindi":"#E05C1A","Tamil":"#C0392B","Telugu":"#7B2FBE","Kannada":"#0891B2",
    "Malayalam":"#059669","Bengali":"#D97706","Marathi":"#BE185D","Gujarati":"#0D9488",
    "Punjabi":"#4F46E5","Urdu":"#65A30D","Odia":"#0284C7","Assamese":"#EA580C",
}
LANG_SHORT = {
    "Hindi":"HI","Tamil":"TA","Telugu":"TE","Kannada":"KN","Malayalam":"ML",
    "Bengali":"BN","Marathi":"MR","Gujarati":"GU","Punjabi":"PB","Urdu":"UR",
    "Odia":"OD","Assamese":"AS",
}

HISTORY_FILE = "bhasha_setu_history.json"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Session state defaults ─────────────────────────────────────────────────────
_DEFAULTS = {
    "dark_mode": True, "sel_lang": "Hindi", "sel_src_lang": "Auto-detect",
    "batch_langs": [], "result": None, "batch_results": [],
    "pcts": [0.0]*5, "msgs": [""]*5, "cur_stage": 0,
    "hil_phase": "idle", "hil_data": None, "hil_tmp": "",
    "hil4_phase": "idle", "hil4_data": None, "hil4_tmp": "",
    "url_text": "", "url_info": None,
    "tr_src": "English", "tr_tgt": "Hindi", "tr_input": "", "tr_output": "",
    "chat_msgs": [],
    "ui_lang": "en",
}
for k, v in _DEFAULTS.items():
    if k not in st.session_state: st.session_state[k] = v

DM = st.session_state.dark_mode

# ══════════════════════════════════════════════════════════════════════════════
#  DESIGN SYSTEM (Figma & Snitch Modern Multilingual UI)
# ══════════════════════════════════════════════════════════════════════════════
DARK_CSS = """
:root {
  --bg: #080B14;
  --bg1: #0D1322;
  --bg2: #121A2D;
  --bg3: #18223B;
  --sur: rgba(255, 255, 255, 0.04);
  --sur2: rgba(255, 255, 255, 0.075);
  --sur3: rgba(255, 255, 255, 0.11);
  --bdr: rgba(255, 255, 255, 0.09);
  --bdr2: rgba(255, 255, 255, 0.18);
  --tx1: #FFFFFF;
  --tx2: #CBD5E1;
  --tx3: #8E9EB6;
  --tx4: #475569;
  --p1: #FF6B35;
  --p2: #FB923C;
  --pg: linear-gradient(135deg, #FF6B35 0%, #EA580C 100%);
  --pg2: linear-gradient(135deg, #FF6B35 0%, #F97316 40%, #EA580C 100%);
  --a1: #22D3EE;
  --ag: linear-gradient(135deg, #22D3EE, #818CF8);
  --g1: #34D399;
  --gs: rgba(52, 211, 153, 0.09);
  --gb: rgba(52, 211, 153, 0.22);
  --am1: #FBBF24;
  --ams: rgba(251, 191, 36, 0.09);
  --amb: rgba(251, 191, 36, 0.22);
  --r1: #F87171;
  --rs: rgba(248, 113, 113, 0.09);
  --rb: rgba(248, 113, 113, 0.22);
  --ps: rgba(255, 107, 53, 0.10);
  --pb: rgba(255, 107, 53, 0.24);
  --glass: rgba(18, 26, 45, 0.78);
  --glass2: rgba(24, 34, 59, 0.85);
  --trk: rgba(255, 255, 255, 0.08);
  --inp: #131C2E;
  --inpb: rgba(255, 255, 255, 0.16);
  --sh: 0 2px 14px rgba(0, 0, 0, 0.55), 0 8px 32px rgba(0, 0, 0, 0.40);
  --sh2: 0 6px 28px rgba(0, 0, 0, 0.68), 0 16px 48px rgba(0, 0, 0, 0.48);
  --psh: 0 4px 20px rgba(255, 107, 53, 0.35), 0 8px 40px rgba(255, 107, 53, 0.18);
  --rad: 28px;
  --rad-lg: 34px;
  --rad-sm: 9999px;
  --rad-pill: 9999px;
}
"""
LIGHT_CSS = """
:root {
  --bg: #F8FAFC;
  --bg1: #FFFFFF;
  --bg2: #EEF2F6;
  --bg3: #E2E8F0;
  --sur: #FFFFFF;
  --sur2: #F1F5F9;
  --sur3: rgba(234, 88, 12, 0.08);
  --bdr: rgba(15, 23, 42, 0.09);
  --bdr2: rgba(15, 23, 42, 0.16);
  --tx1: #0F172A;
  --tx2: #334155;
  --tx3: #64748B;
  --tx4: #94A3B8;
  --p1: #EA580C;
  --p2: #F97316;
  --pg: linear-gradient(135deg, #FF6B35 0%, #EA580C 100%);
  --pg2: linear-gradient(135deg, #FF6B35 0%, #F97316 45%, #EA580C 100%);
  --a1: #0284C7;
  --ag: linear-gradient(135deg, #0284C7, #4F46E5);
  --g1: #059669;
  --gs: rgba(5, 150, 105, 0.09);
  --gb: rgba(5, 150, 105, 0.24);
  --am1: #D97706;
  --ams: rgba(217, 119, 6, 0.09);
  --amb: rgba(217, 119, 6, 0.24);
  --r1: #DC2626;
  --rs: rgba(220, 38, 38, 0.09);
  --rb: rgba(220, 38, 38, 0.24);
  --ps: rgba(234, 88, 12, 0.09);
  --pb: rgba(234, 88, 12, 0.24);
  --glass: #FFFFFF;
  --glass2: #F8FAFC;
  --trk: #E2E8F0;
  --inp: #FFFFFF;
  --inpb: #CBD5E1;
  --sh: 0 2px 10px rgba(15, 23, 42, 0.04), 0 12px 28px rgba(15, 23, 42, 0.05);
  --sh2: 0 4px 16px rgba(15, 23, 42, 0.06), 0 20px 44px rgba(15, 23, 42, 0.08);
  --psh: 0 4px 18px rgba(234, 88, 12, 0.35), 0 8px 32px rgba(234, 88, 12, 0.15);
  --rad: 28px;
  --rad-lg: 34px;
  --rad-sm: 9999px;
  --rad-pill: 9999px;
}
"""

BASE_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=Plus+Jakarta+Sans:ital,wght@0,400;0,500;0,600;0,700;0,800&display=swap');

@keyframes fadeUp  {from{opacity:0;transform:translateY(18px)}to{opacity:1;transform:none}}
@keyframes fadeIn  {from{opacity:0}to{opacity:1}}
@keyframes slideIn {from{opacity:0;transform:translateX(-14px)}to{opacity:1;transform:none}}
@keyframes pulse   {0%,100%{opacity:1}50%{opacity:.35}}
@keyframes shimmer {0%{background-position:-200% 0}100%{background-position:200% 0}}
@keyframes float   {0%,100%{transform:translateY(0)}50%{transform:translateY(-7px)}}
@keyframes gradmov {0%,100%{background-position:0% 50%}50%{background-position:100% 50%}}
@keyframes scaleIn {from{opacity:0;transform:scale(.95)}to{opacity:1;transform:scale(1)}}

/* ROOT APPLICATION CANVAS & FULL THEME APPLICATION */
html,
body,
.stApp,
[data-testid="stApp"],
[data-testid="stAppViewContainer"],
[data-testid="stAppViewContainer"] > .main,
section.main,
[data-testid="stBottom"],
.block-container {
  background-color: var(--bg) !important;
  color: var(--tx1) !important;
  font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
  transition: background-color .28s ease, color .28s ease !important;
}

[data-testid="stAppViewContainer"] > .main {
  background-color: var(--bg) !important;
  background-image: radial-gradient(ellipse 65% 45% at 15% -5%, rgba(244, 114, 43, 0.08) 0%, transparent 60%),
                    radial-gradient(ellipse 55% 40% at 88% 105%, rgba(59, 130, 246, 0.06) 0%, transparent 55%) !important;
  min-height: 100vh !important;
}

.block-container {
  padding-top: 14px !important;
  padding-bottom: 80px !important;
  max-width: 1360px !important;
}

/* HIDE STREAMLIT CHROME ARTIFACTS */
[data-testid="stHeader"],
[data-testid="stDecoration"],
[data-testid="stToolbar"],
header[data-testid="stHeader"],
div[data-testid="stDecoration"] {
  display: none !important;
  height: 0 !important;
  visibility: hidden !important;
}
footer, #MainMenu {
  visibility: hidden !important;
}

/* TOP NAVBAR FLOATING CAPSULE BAR (Full Circular Curve 9999px) */
div[data-testid="stVerticalBlockBorderWrapper"]:has(button[key="mode_toggle"]),
.top-navbar-card {
  background: var(--glass) !important;
  backdrop-filter: blur(24px) !important;
  -webkit-backdrop-filter: blur(24px) !important;
  border: 1px solid var(--bdr) !important;
  border-radius: 9999px !important;
  padding: 8px 24px !important;
  margin: 6px 0 20px 0 !important;
  box-shadow: var(--sh) !important;
  transition: all .2s ease !important;
}
div[data-testid="stVerticalBlockBorderWrapper"]:has(button[key="mode_toggle"]):hover,
.top-navbar-card:hover {
  border-color: var(--bdr2) !important;
  box-shadow: var(--sh2) !important;
}

/* CIRCULAR ACTION CONTROLS */
button[key="mode_toggle"] {
  min-height: 42px !important;
  height: 42px !important;
  width: 42px !important;
  min-width: 42px !important;
  border-radius: 50% !important;
  padding: 0 !important;
  font-size: 18px !important;
  display: inline-flex !important;
  align-items: center !important;
  justify-content: center !important;
  margin: 0 !important;
  background: var(--sur2) !important;
  border: 1px solid var(--bdr) !important;
  box-shadow: var(--sh) !important;
  transition: all .2s ease !important;
}
button[key="mode_toggle"]:hover {
  transform: scale(1.08) !important;
  border-color: var(--p1) !important;
  background: var(--ps) !important;
}
button[key="logout_btn"] {
  min-height: 42px !important;
  height: 42px !important;
  border-radius: 9999px !important;
  padding: 8px 22px !important;
  font-size: 12.5px !important;
  font-weight: 700 !important;
  display: inline-flex !important;
  align-items: center !important;
  justify-content: center !important;
  margin: 0 !important;
  background: var(--sur2) !important;
  border: 1px solid var(--bdr) !important;
  box-shadow: var(--sh) !important;
  transition: all .2s ease !important;
}
button[key="logout_btn"]:hover {
  transform: translateY(-1px) !important;
  border-color: var(--r1) !important;
  color: var(--r1) !important;
  background: var(--rs) !important;
}

/* ── CIRCULAR CAPSULE PILL TABS DOCK (Figma & Snitch Pill Dock) ── */
div[data-testid="stTabs"],
.stTabs {
  border: none !important;
  background: transparent !important;
}

div[data-testid="stTabs"] > div:first-child,
div[data-testid="stTabs"] [data-baseweb="tab-list"],
.stTabs [data-baseweb="tab-list"],
div[role="tablist"] {
  background: var(--bg2) !important;
  border-radius: 9999px !important;
  padding: 6px 8px !important;
  border: 1px solid var(--bdr2) !important;
  border-bottom: none !important;
  gap: 6px !important;
  display: inline-flex !important;
  flex-direction: row !important;
  flex-wrap: wrap !important;
  box-shadow: var(--sh) !important;
  align-items: center !important;
  margin: 4px 0 22px 0 !important;
}

div[data-testid="stTabs"] button,
div[data-testid="stTabs"] button[data-baseweb="tab"],
div[data-testid="stTabs"] button[role="tab"],
.stTabs [data-baseweb="tab"],
button[role="tab"] {
  border-radius: 9999px !important;
  padding: 10px 22px !important;
  font-family: 'Plus Jakarta Sans', sans-serif !important;
  font-size: 13.5px !important;
  font-weight: 600 !important;
  color: var(--tx2) !important;
  border: none !important;
  outline: none !important;
  background: transparent !important;
  transition: all .2s cubic-bezier(0.16, 1, 0.3, 1) !important;
  height: auto !important;
  min-height: 42px !important;
  margin: 0 !important;
  overflow: hidden !important;
  box-shadow: none !important;
}

div[data-testid="stTabs"] button *,
div[data-testid="stTabs"] button[data-baseweb="tab"] *,
.stTabs [data-baseweb="tab"] * {
  border-radius: 9999px !important;
  color: inherit !important;
}

div[data-testid="stTabs"] button:hover,
div[data-testid="stTabs"] button[data-baseweb="tab"]:hover,
.stTabs [data-baseweb="tab"]:hover {
  color: var(--tx1) !important;
  background: var(--sur2) !important;
  border-radius: 9999px !important;
  transform: translateY(-1px) !important;
}

/* ACTIVE SELECTED TAB — SMOOTH CAPSULE PILL */
div[data-testid="stTabs"] button[aria-selected="true"],
div[data-testid="stTabs"] button[data-baseweb="tab"][aria-selected="true"],
div[data-testid="stTabs"] button[role="tab"][aria-selected="true"],
.stTabs [data-baseweb="tab"][aria-selected="true"],
button[role="tab"][aria-selected="true"] {
  background: var(--pg) !important;
  color: #FFFFFF !important;
  font-weight: 700 !important;
  border-radius: 9999px !important;
  box-shadow: 0 4px 18px rgba(244,114,43,0.45), 0 1px 3px rgba(0,0,0,0.12) !important;
  transform: translateY(-1px) !important;
  border: none !important;
}
div[data-testid="stTabs"] button[aria-selected="true"] *,
div[data-testid="stTabs"] button[data-baseweb="tab"][aria-selected="true"] * {
  color: #FFFFFF !important;
  font-weight: 700 !important;
}

/* COMPLETELY HIDE BASEWEB HIGHLIGHT LINE & UNDERLINE BORDER */
div[data-testid="stTabs"] [data-baseweb="tab-border"],
div[data-testid="stTabs"] [data-baseweb="tab-highlight"],
.stTabs [data-baseweb="tab-border"],
.stTabs [data-baseweb="tab-highlight"],
[data-baseweb="tab-border"],
[data-baseweb="tab-highlight"] {
  display: none !important;
  opacity: 0 !important;
  visibility: hidden !important;
  height: 0 !important;
  width: 0 !important;
  border: none !important;
}
div[data-testid="stTabs"] [data-baseweb="tab-panel"],
.stTabs [data-baseweb="tab-panel"]{
  padding-top:16px!important;
  border:none!important;
}

/* CARDS & CONTAINERS: GENEROUS MODERN CURVATURE (28px) */
div[data-testid="stVerticalBlockBorderWrapper"],
[data-testid="stVerticalBlockBorderWrapper"] {
  background: var(--glass) !important;
  backdrop-filter: blur(20px) !important;
  -webkit-backdrop-filter: blur(20px) !important;
  border: 1px solid var(--bdr) !important;
  border-radius: var(--rad) !important;
  box-shadow: var(--sh) !important;
  padding: 24px 26px !important;
  margin-bottom: 18px !important;
  transition: all .22s ease !important;
}
div[data-testid="stVerticalBlockBorderWrapper"]:hover {
  border-color: var(--bdr2) !important;
  box-shadow: var(--sh2) !important;
}

/* CAPSULE PILL BUTTONS (Circular Edges) */
.stButton>button{
  font-family:'Plus Jakarta Sans',sans-serif!important;font-weight:700!important;
  font-size:13.5px!important;border-radius:9999px!important;
  padding:11px 26px!important;min-height:44px!important;
  border:1px solid var(--bdr)!important;background:var(--sur2)!important;
  color:var(--tx1)!important;box-shadow:var(--sh)!important;
  transition:transform .18s cubic-bezier(0.16, 1, 0.3, 1),box-shadow .18s,background .18s,border-color .18s!important;
  letter-spacing:-.1px!important;
}
.stButton>button:hover{
  transform:translateY(-2px)!important;
  border-color:var(--bdr2)!important;
  box-shadow:var(--sh2)!important;
  color:var(--p1)!important;
}
.stButton>button:active{transform:translateY(0)!important;}
.stButton>button[kind="primary"]{
  background:var(--pg)!important;color:#fff!important;border:none!important;
  border-radius:9999px!important;
  box-shadow:var(--psh)!important;
}
.stButton>button[kind="primary"]:hover{
  box-shadow:0 6px 28px rgba(244,114,43,.50),0 2px 8px rgba(0,0,0,.2)!important;
  color:#fff!important;
  transform:translateY(-2px)!important;
}

/* SEGMENTED RADIO TOGGLES (Snitch & Figma modern segmented bar) */
div[data-testid="stRadio"] > div[role="radiogroup"]{
  display:inline-flex!important;
  flex-direction:row!important;
  background:var(--bg2)!important;
  border:1px solid var(--bdr)!important;
  border-radius:9999px!important;
  padding:4px!important;
  gap:4px!important;
  align-items:center!important;
}
div[data-testid="stRadio"] > div[role="radiogroup"] > label{
  display:inline-flex!important;
  align-items:center!important;
  justify-content:center!important;
  padding:8px 22px!important;
  border-radius:9999px!important;
  margin:0!important;
  cursor:pointer!important;
  transition:all .2s cubic-bezier(0.16, 1, 0.3, 1)!important;
  background:transparent!important;
  font-size:13px!important;
  font-weight:600!important;
  color:var(--tx2)!important;
}
div[data-testid="stRadio"] > div[role="radiogroup"] > label:hover{
  color:var(--tx1)!important;
  background:var(--sur2)!important;
}
div[data-testid="stRadio"] > div[role="radiogroup"] > label:has(input:checked){
  background:var(--pg)!important;
  color:#FFFFFF!important;
  font-weight:700!important;
  box-shadow:0 3px 12px rgba(244,114,43,.35)!important;
}
div[data-testid="stRadio"] > div[role="radiogroup"] > label:has(input:checked) span,
div[data-testid="stRadio"] > div[role="radiogroup"] > label:has(input:checked) p{
  color:#FFFFFF!important;
}
div[data-testid="stRadio"] > div[role="radiogroup"] input[type="radio"]{
  display:none!important;
}
div[data-testid="stRadio"] > div[role="radiogroup"] div[data-testid="stMarkdownContainer"]{
  margin:0!important;
  padding:0!important;
}

/* CURATED CHECKBOX SWITCHES */
div[data-testid="stCheckbox"]{
  background:var(--sur)!important;
  border:1px solid var(--bdr)!important;
  border-radius:16px!important;
  padding:10px 16px!important;
  transition:all .2s ease!important;
  margin-bottom:8px!important;
}
div[data-testid="stCheckbox"]:hover{
  border-color:var(--bdr2)!important;
  background:var(--sur2)!important;
  box-shadow:var(--sh)!important;
}
div[data-testid="stCheckbox"] label{
  cursor:pointer!important;
  font-size:13px!important;
  font-weight:600!important;
  color:var(--tx2)!important;
}
div[data-testid="stCheckbox"] span[data-baseweb="checkbox"]{
  border-radius:6px!important;
}

/* INPUTS: FULL CIRCULAR PILL */
[data-testid="stTextInput"] input{
  background:var(--inp)!important;border:1.5px solid var(--inpb)!important;
  border-radius:9999px!important;color:var(--tx1)!important;
  font-family:'Plus Jakarta Sans',sans-serif!important;font-size:13.5px!important;
  font-weight:500!important;padding:10px 22px!important;min-height:46px!important;
  transition:border-color .18s,box-shadow .18s!important;
}
[data-testid="stTextInput"] input:focus{
  border-color:var(--p1)!important;box-shadow:0 0 0 3px rgba(244,114,43,.18)!important;
}
[data-testid="stTextInput"] input::placeholder{color:var(--tx3)!important;}

[data-testid="stTextArea"] textarea{
  background:var(--inp)!important;border:1.5px solid var(--inpb)!important;
  border-radius:24px!important;color:var(--tx1)!important;
  font-family:'Plus Jakarta Sans',sans-serif!important;font-size:13.5px!important;
  font-weight:500!important;padding:14px 20px!important;
  transition:border-color .18s,box-shadow .18s!important;
}
[data-testid="stTextArea"] textarea:focus{
  border-color:var(--p1)!important;box-shadow:0 0 0 3px rgba(244,114,43,.18)!important;
}
[data-testid="stTextArea"] textarea::placeholder{color:var(--tx3)!important;}

/* SELECTBOXES: FULL CIRCULAR CAPSULE WITH CRISP CONTRAST */
[data-testid="stSelectbox"] > div > div,
[data-baseweb="select"] > div,
div[data-baseweb="select"],
div[data-baseweb="select"] > div:first-child,
div[data-baseweb="select"] input {
  background: var(--inp) !important;
  background-color: var(--inp) !important;
  border: 1.5px solid var(--inpb) !important;
  border-radius: 9999px !important;
  color: var(--tx1) !important;
  min-height: 46px !important;
  font-size: 13.5px !important;
  font-weight: 500 !important;
  transition: border-color .18s, box-shadow .18s !important;
}
[data-testid="stSelectbox"] > div > div:hover,
[data-baseweb="select"] > div:hover {
  border-color: var(--p1) !important;
}
[data-baseweb="select"] > div:focus-within {
  border-color: var(--p1) !important;
  box-shadow: 0 0 0 3px rgba(244, 114, 43, 0.22) !important;
}
[data-testid="stSelectbox"] *,
[data-baseweb="select"] * {
  color: var(--tx1) !important;
}
[data-testid="stSelectbox"] svg,
[data-baseweb="select"] svg {
  fill: var(--tx2) !important;
  color: var(--tx2) !important;
}

label[data-testid="stWidgetLabel"],
label[data-testid="stWidgetLabel"] p,
label[data-testid="stWidgetLabel"] span {
  color: var(--tx1) !important;
  font-weight: 700 !important;
  font-size: 12.5px !important;
  letter-spacing: 0.3px !important;
}

ul[data-baseweb="menu"],
div[data-baseweb="popover"],
div[data-baseweb="popover"] > div {
  background: var(--bg1) !important;
  background-color: var(--bg1) !important;
  border: 1px solid var(--bdr2) !important;
  border-radius: 20px !important;
  box-shadow: var(--sh2) !important;
  padding: 8px !important;
}
li[data-baseweb="menu-item"] {
  border-radius: 12px !important;
  padding: 9px 16px !important;
  font-size: 13.5px !important;
  color: var(--tx1) !important;
  font-weight: 600 !important;
  background: transparent !important;
}
li[data-baseweb="menu-item"]:hover {
  background: var(--sur2) !important;
  color: var(--p1) !important;
}

/* SLIDERS */
div[data-testid="stSlider"] [data-baseweb="slider"] div[role="slider"]{
  background-color:var(--p1)!important;
  border:2px solid #FFFFFF!important;
  box-shadow:0 2px 8px rgba(244,114,43,.45)!important;
  border-radius:50%!important;
}
div[data-testid="stSlider"] [data-baseweb="slider"] div{
  border-radius:9999px!important;
}

/* FILE UPLOADER */
[data-testid="stFileUploader"]>div>div{
  background:var(--sur2)!important;border:2px dashed var(--bdr2)!important;
  border-radius:24px!important;padding:24px!important;transition:border-color .18s,background .18s!important;
}
[data-testid="stFileUploader"]>div>div:hover{border-color:var(--p1)!important;background:var(--ps)!important;}
[data-testid="stFileUploader"] button{border-radius:9999px!important;}

/* EXPANDERS & ALERTS */
.stExpander{
  border:1px solid var(--bdr)!important;
  border-radius:24px!important;
  background:var(--sur)!important;
  transition:border-color .2s,box-shadow .2s!important;
  overflow:hidden!important;
  box-shadow:var(--sh)!important;
}
.stExpander:hover{border-color:var(--bdr2)!important;box-shadow:var(--sh2)!important;}
.stExpander details summary{
  padding:14px 20px!important;font-size:13.5px!important;font-weight:700!important;color:var(--tx1)!important;
}
.stAlert{background:var(--sur)!important;border:1px solid var(--bdr)!important;border-radius:20px!important;}
code{color:var(--p1)!important;background:var(--ps)!important;padding:2px 7px;border-radius:8px;font-size:12px;font-family:monospace;}
[data-testid="stVideo"] video{border-radius:24px!important;box-shadow:0 8px 36px rgba(0,0,0,.55)!important;border:1px solid var(--bdr2)!important;}
.stMarkdown p,.stMarkdown li{color:var(--tx2)!important;font-size:14px!important;line-height:1.8!important;}
::-webkit-scrollbar{width:5px;height:5px;}
::-webkit-scrollbar-track{background:transparent;}
::-webkit-scrollbar-thumb{background:var(--pg);border-radius:9999px;}

/* TYPOGRAPHY */
.h-xl{font-family:'Syne',sans-serif;font-size:clamp(32px,5vw,60px);
      font-weight:800;letter-spacing:-2px;line-height:1.08;color:var(--tx1);}
.h-xl .gr{background:var(--pg2);background-size:200% 200%;
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;
  background-clip:text;animation:gradmov 4s ease infinite;}
.h-sec{font-family:'Syne',sans-serif;font-size:clamp(20px,3vw,32px);
       font-weight:800;letter-spacing:-1px;color:var(--tx1);margin-bottom:10px;}
.h-card{font-family:'Syne',sans-serif;font-size:15px;font-weight:700;
        color:var(--tx1);letter-spacing:-.3px;margin-bottom:10px;}
.eyebrow{font-size:11px;font-weight:700;color:var(--p1);
         letter-spacing:1.1px;text-transform:uppercase;margin-bottom:8px;}
.body{font-size:14px;color:var(--tx2);line-height:1.8;}
.label{font-size:11px;font-weight:700;color:var(--tx3);
       letter-spacing:.8px;text-transform:uppercase;margin-bottom:7px;}

/* HERO */
.hero{padding:68px 20px 48px;text-align:center;position:relative;}
.hero-badge{display:inline-flex;align-items:center;gap:6px;
  background:var(--ps);border:1px solid var(--pb);color:var(--p1);
  font-size:12px;font-weight:700;padding:6px 18px;border-radius:9999px;
  margin-bottom:20px;letter-spacing:.3px;animation:fadeUp .5s ease both;}
.hero-sub{font-size:clamp(14px,2vw,18px);color:var(--tx2);line-height:1.8;
  max-width:540px;margin:14px auto 32px;animation:fadeUp .55s ease both;animation-delay:.07s;}

/* STATS BAR */
.stats-bar{display:flex;justify-content:center;flex-wrap:wrap;
  border:1px solid var(--bdr);border-radius:9999px;overflow:hidden;
  max-width:680px;margin:0 auto;background:var(--glass);backdrop-filter:blur(16px);
  box-shadow:var(--sh);animation:fadeUp .6s ease both;animation-delay:.14s;}
.stat-cell{flex:1;min-width:100px;text-align:center;padding:18px 10px;
           border-right:1px solid var(--bdr);transition:background .18s;}
.stat-cell:last-child{border-right:none;}
.stat-cell:hover{background:var(--sur2);}
.stat-n{font-family:'Syne',sans-serif;font-size:26px;font-weight:800;letter-spacing:-.8px;
        background:var(--pg2);background-size:200% 200%;
        -webkit-background-clip:text;-webkit-text-fill-color:transparent;
        background-clip:text;animation:gradmov 4s ease infinite;}
.stat-l{font-size:11px;color:var(--tx3);font-weight:600;margin-top:3px;}

/* SOURCE/TARGET LANG STRIP */
.lang-strip{display:flex;align-items:center;gap:10px;
  background:var(--sur2);border:1px solid var(--bdr2);border-radius:9999px;
  padding:12px 20px;margin-bottom:14px;}
.lang-arrow{color:var(--p1);font-size:18px;font-weight:700;flex-shrink:0;}
.lang-pill{display:inline-flex;align-items:center;gap:6px;
  background:var(--ps);border:1px solid var(--pb);color:var(--p1);
  font-size:12px;font-weight:700;padding:5px 14px;border-radius:9999px;}

/* PIPELINE PROGRESS */
.pdash{background:var(--glass);backdrop-filter:blur(16px);
       border:1px solid var(--bdr);border-radius:var(--rad);padding:22px;box-shadow:var(--sh);}
.pdash-hdr{display:flex;align-items:center;justify-content:space-between;
           margin-bottom:14px;font-family:'Syne',sans-serif;
           font-size:13.5px;font-weight:700;color:var(--tx1);}
.pdash-ov{font-family:'Syne',sans-serif;font-size:22px;font-weight:800;
          background:var(--pg2);background-size:200% 200%;
          -webkit-background-clip:text;-webkit-text-fill-color:transparent;
          background-clip:text;animation:gradmov 3s ease infinite;}
.srow{display:flex;align-items:center;gap:11px;margin-bottom:11px;}
.srow:last-child{margin-bottom:0;}
.si{width:38px;height:38px;border-radius:50%;background:var(--sur2);
    border:1px solid var(--bdr);display:flex;align-items:center;
    justify-content:center;font-size:15px;flex-shrink:0;transition:all .2s;}
.si.act{background:var(--ps);border-color:var(--pb);animation:pulse 1.4s ease infinite;}
.sinfo{flex:1;min-width:0;}
.slr{display:flex;justify-content:space-between;align-items:center;margin-bottom:4px;}
.sn{font-size:12px;font-weight:600;color:var(--tx2);}
.sp{font-size:11.5px;font-weight:700;color:var(--tx3);}
.st-track{background:var(--trk);border-radius:9999px;height:6px;overflow:hidden;}
.sf{height:100%;border-radius:9999px;transition:width .6s cubic-bezier(.4,0,.2,1);}
.f1{background:linear-gradient(90deg,#F4722B,#E53E3E);}
.f2{background:linear-gradient(90deg,#22D3EE,#818CF8);}
.f3{background:linear-gradient(90deg,#34D399,#22D3EE);}
.f4{background:linear-gradient(90deg,#FBBF24,#F87171);}
.f5{background:linear-gradient(90deg,#C084FC,#818CF8);}

/* CHIPS */
.chip{display:inline-flex;align-items:center;gap:5px;background:var(--sur2);
      border:1px solid var(--bdr2);color:var(--tx2);font-size:11px;
      font-weight:600;padding:4px 12px;border-radius:9999px;}
.chip-g{background:var(--gs)!important;border-color:var(--gb)!important;color:var(--g1)!important;}
.chip-p{background:var(--ps)!important;border-color:var(--pb)!important;color:var(--p1)!important;}
.chip-a{background:rgba(34,211,238,.08)!important;border-color:rgba(34,211,238,.22)!important;color:var(--a1)!important;}

/* RESULT */
.res-ok{display:flex;align-items:center;gap:12px;padding:16px 22px;
        background:var(--gs);border:1px solid var(--gb);border-radius:24px;
        margin-bottom:14px;animation:scaleIn .3s ease both;}
.res-ico{width:44px;height:44px;border-radius:50%;background:rgba(52,211,153,.18);
         display:flex;align-items:center;justify-content:center;font-size:20px;flex-shrink:0;}
.res-ttl{font-family:'Syne',sans-serif;font-size:15px;font-weight:700;color:var(--g1);}
.res-sub{font-size:11.5px;color:var(--tx3);margin-top:2px;}

/* HISTORY */
.hcard{background:var(--sur);border:1px solid var(--bdr);border-radius:20px;
       padding:14px 20px;margin-bottom:8px;transition:all .18s;box-shadow:var(--sh);}
.hcard:hover{border-color:var(--bdr2);transform:translateX(3px);box-shadow:var(--sh2);}
.met{background:var(--glass);backdrop-filter:blur(10px);border:1px solid var(--bdr);
     border-radius:var(--rad);padding:22px;text-align:center;transition:all .18s;box-shadow:var(--sh);}
.met:hover{border-color:var(--bdr2);box-shadow:var(--sh2);}
.met-n{font-family:'Syne',sans-serif;font-size:26px;font-weight:800;
       background:var(--pg2);background-size:200% 200%;
       -webkit-background-clip:text;-webkit-text-fill-color:transparent;
       background-clip:text;animation:gradmov 4s ease infinite;}
.met-l{font-size:11.5px;color:var(--tx3);font-weight:600;margin-top:4px;}

/* ROADMAP */
.rm{background:var(--sur);border:1px solid var(--bdr);border-radius:20px;
    padding:16px 22px;margin-bottom:10px;transition:all .2s;box-shadow:var(--sh);}
.rm:hover{border-color:var(--bdr2);transform:translateX(4px);box-shadow:var(--sh2);}
.rm-tag{font-size:9.5px;font-weight:700;color:var(--tx3);text-transform:uppercase;letter-spacing:.8px;margin-bottom:3px;}
.rm-ttl{font-family:'Syne',sans-serif;font-size:13.5px;font-weight:700;color:var(--tx1);margin-bottom:4px;}
.rm-desc{font-size:12px;color:var(--tx2);line-height:1.62;margin-bottom:6px;}
.pill{display:inline-flex;align-items:center;gap:4px;font-size:10px;font-weight:700;padding:3px 10px;border-radius:9999px;}
.p-live{background:var(--gs);color:var(--g1);border:1px solid var(--gb);}
.p-soon{background:var(--ams);color:var(--am1);border:1px solid var(--amb);}
.p-plan{background:var(--sur2);color:var(--tx3);border:1px solid var(--bdr);}

/* MISC */
.tp{font-size:12.5px;color:var(--tx2);line-height:1.8;background:var(--bg2);
    border-radius:20px;padding:16px;max-height:200px;overflow-y:auto;border:1px solid var(--bdr);}
.sdiv{height:1px;background:linear-gradient(90deg,transparent,var(--pb),transparent);border:none;margin:28px 0;}
.idiv{height:1px;background:var(--bdr);border:none;margin:16px 0;}
.bar-row{display:flex;align-items:center;gap:10px;margin-bottom:8px;}
.bar-lbl{font-size:12px;color:var(--tx2);min-width:70px;font-weight:500;}
.bar-track{flex:1;background:var(--trk);border-radius:9999px;height:6px;overflow:hidden;}
.bar-fill{height:100%;border-radius:9999px;background:var(--pg);transition:width .8s ease;}
.bar-val{font-size:11px;color:var(--tx3);min-width:22px;text-align:right;}

/* ── MOBILE RESPONSIVENESS ── */
@media (max-width: 768px) {
  .block-container { padding-left: 14px !important; padding-right: 14px !important; }
  .h-xl { font-size: 28px !important; letter-spacing: -1px !important; }
  .h-sec { font-size: 20px !important; }
  .hero { padding: 36px 10px 24px !important; }
  .hero-sub { font-size: 13.5px !important; }
  .stats-bar { flex-direction: column !important; width: 100% !important; max-width: 100% !important; border-radius: 24px !important; }
  .stat-cell { border-right: none !important; border-bottom: 1px solid var(--bdr) !important; padding: 10px !important; min-width: 100% !important; }
  .stat-cell:last-child { border-bottom: none !important; }
  div[data-testid="stTabs"] [data-baseweb="tab-list"],
  .stTabs [data-baseweb="tab-list"] {
    display: flex !important;
    overflow-x: auto !important;
    white-space: nowrap !important;
    -webkit-overflow-scrolling: touch !important;
    padding: 6px !important;
    border-radius: 9999px !important;
  }
  div[data-testid="stTabs"] [data-baseweb="tab"],
  .stTabs [data-baseweb="tab"] {
    flex-shrink: 0 !important;
    font-size: 11.5px !important;
    padding: 7px 14px !important;
  }
  div[data-testid="column"] {
    width: 100% !important;
    flex: 1 1 100% !important;
    min-width: 100% !important;
  }
  .lang-strip { flex-wrap: wrap !important; border-radius: 20px !important; }
  .stButton > button { min-height: 44px !important; width: 100% !important; border-radius: 9999px !important; }
  [data-testid="stVideo"] video { width: 100% !important; height: auto !important; }
}

@media (max-width: 480px) {
  .h-xl { font-size: 23px !important; }
  .hero-badge { font-size: 11px !important; padding: 4px 10px !important; }
}
"""

st.markdown(f"<style>{(DARK_CSS if DM else LIGHT_CSS)}{BASE_CSS}</style>", unsafe_allow_html=True)

# ── Database & Authentication Gate ─────────────────────────────────────────────
init_db()
current_user = get_current_user()
if not current_user:
    render_login_page()
    st.stop()

# ── Custom 404 & Route Verification (Checklist: Custom 404 page) ───────────────
if st.query_params.get("view") == "404" or st.query_params.get("error") == "404":
    st.markdown(
        """
        <div style="text-align:center;padding:70px 20px 40px;max-width:600px;margin:0 auto;animation:fadeUp .4s ease both;">
            <div style="font-size:56px;margin-bottom:12px;">🧭</div>
            <div style="font-family:'Syne',sans-serif;font-size:36px;font-weight:800;color:var(--tx1);margin-bottom:8px">
                404 &middot; Page Not Found
            </div>
            <div style="font-size:14px;color:var(--tx2);line-height:1.7;margin-bottom:28px">
                The requested URL route or dubbing resource could not be found. Don't worry, your dubbing projects and session are completely safe.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    col1, col2, col3 = st.columns([1, 1.4, 1])
    with col2:
        if st.button("← Return to Bhasha Setu Dashboard", type="primary", use_container_width=True, key="btn_404_back"):
            st.query_params.clear()
            st.rerun()
    st.stop()


# ══════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def load_history():
    user = get_current_user()
    if not user:
        return []
    is_admin = (user.get("role") == "admin")
    return get_user_history(user["id"], is_admin=is_admin)

def save_history(entry):
    user = get_current_user()
    if user:
        save_user_history(user["id"], entry)

def clip_preview(path, secs):
    base, ext = os.path.splitext(path)
    out = base + "_prev" + ext
    try:
        subprocess.run(["ffmpeg","-y","-i",path,"-t",str(secs),"-c","copy",out],
                       check=True, capture_output=True)
        return out if os.path.exists(out) else None
    except Exception: return None

def lbadge(lname, size=40, radius=10):
    s = LANG_SHORT.get(lname, lname[:2].upper())
    c = LANG_COLOR.get(lname, "#6366F1")
    return (f'<div style="background:{c};width:{size}px;height:{size}px;border-radius:{radius}px;'
            f'display:flex;align-items:center;justify-content:center;'
            f'font-family:Syne,sans-serif;font-size:{int(size*.28)}px;font-weight:800;color:#fff">{s}</div>')

def do_translate(text: str, src_lang: str, tgt_lang: str, engine: str = "auto") -> Tuple[Optional[str], Optional[str]]:
    if not text or not text.strip():
        return "", None
    try:
        s = LANG_CODES.get(src_lang, "en")
        t = LANG_CODES.get(tgt_lang, "hi")
        if s == t:
            return text, None
        from pipeline.translate import translate_text
        res = translate_text(text, target_lang_code=t, source_lang_code=s, translate_engine=engine)
        if res and res.strip():
            return res, None
        if TRANSLATOR_READY:
            return _GT(source=s, target=t).translate(text[:4500]), None
        return text, None
    except Exception as e:
        log.warning(f"Translation engine error: {e}. Trying fallback.")
        try:
            if TRANSLATOR_READY:
                s_code = LANG_CODES.get(src_lang, "en")
                t_code = LANG_CODES.get(tgt_lang, "hi")
                return _GT(source=s_code, target=t_code).translate(text[:4500]), None
        except Exception as e2:
            return None, str(e2)
        return None, str(e)

def _call_groq_rest(messages: list, max_tokens: int = 512, temperature: float = 0.5) -> str:
    """Call Groq REST API directly without C-extension or DLL dependencies."""
    key = os.environ.get("GROQ_API_KEY", "").strip()
    if not key:
        try: key = st.secrets.get("GROQ_API_KEY", "").strip()
        except Exception: key = ""
    if not key:
        return t("err_groq_not_set") if "err_groq_not_set" in UI_LANGUAGES else "⚠️ GROQ_API_KEY not set in secrets.toml"

    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    for m_name in ["qwen/qwen3.8-27b", "openai/gpt-oss-20b", "llama-3.1-8b-instant"]:
        try:
            resp = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers=headers,
                json={
                    "model": m_name,
                    "messages": messages,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                },
                timeout=30,
            )
            if resp.status_code == 200:
                data = resp.json()
                return data["choices"][0]["message"]["content"].strip()
            elif resp.status_code == 404:
                continue
            else:
                log.warning(f"Groq REST {m_name} status {resp.status_code}: {resp.text[:120]}")
        except Exception as ex:
            log.warning(f"Groq REST call failed for {m_name}: {ex}")
            continue
    return "⚠️ Groq AI service is currently unavailable. Please verify API key and internet connectivity."


def do_chat(user_msg, history, resp_lang):
    lang_instr = f"Always respond in {resp_lang} language." if resp_lang != "English" else "Respond in English."
    transcript_ctx = ""
    if st.session_state.get("result"):
        res = st.session_state.result
        tr = res.get("transcript", "")
        tx = res.get("translation", "")
        tgt_l = res.get("language", resp_lang)
        if tr:
            transcript_ctx = (
                f"\n\n--- DUBBED VIDEO CONTEXT ---\n"
                f"Target Language: {tgt_l}\n"
                f"Original Transcript (first 3500 chars):\n{tr[:3500]}\n"
            )
            if tx:
                transcript_ctx += f"Translated Text ({tgt_l}) (first 3500 chars):\n{tx[:3500]}\n"
            transcript_ctx += (
                f"--- END DUBBED VIDEO CONTEXT ---\n"
                f"You are the official Video Intelligence Assistant. You can answer any questions, "
                f"provide deep summaries, or explain concepts from this video in {resp_lang}."
            )

    system = (
        f"You are Bhasha Setu AI — an advanced multilingual assistant for Indian regional languages and video intelligence. "
        f"Help with language nuances, translation quality, video content questions, education, and general knowledge. "
        f"Be helpful, culturally aware, articulate, and friendly. {lang_instr}{transcript_ctx}"
    )

    msgs = [{"role": "system", "content": system}]
    for m in history[-6:]:
        msgs.append({"role": "user" if m["role"] == "user" else "assistant", "content": m["content"]})

    return _call_groq_rest(msgs, max_tokens=650, temperature=0.6)

def pdash_html(pcts, msgs, cur, label="Running…", done=False, err=False):
    overall = int(sum(pcts)/5)
    col  = "#34D399" if done else ("#F87171" if err else "var(--p1)")
    ico  = "✅" if done else ("❌" if err else "⚙️")
    SM   = [("📤","Upload","f1"),("🎙️","Transcribe","f2"),("🌐","Translate","f3"),
            ("🔊","Synthesize","f4"),("🎬","Mux","f5")]
    rows = ""
    esc_label = html.escape(str(label))
    for i,(em,nm,fc) in enumerate(SM):
        p   = min(100, int(pcts[i]))
        act = "act" if (i+1)==cur and not done and not err else ""
        msg_txt = html.escape(str(msgs[i])) if msgs[i] else "Waiting…"
        rows += (f'<div class="srow">'
                 f'<div class="si {act}">{em}</div>'
                 f'<div class="sinfo">'
                 f'<div class="slr"><span class="sn">{nm}</span><span class="sp">{p}%</span></div>'
                 f'<div class="st-track"><div class="sf {fc}" style="width:{p}%"></div></div>'
                 f'<div style="font-size:10.5px;color:var(--tx3);margin-top:3px;'
                 f'white-space:nowrap;overflow:hidden;text-overflow:ellipsis">'
                 f'{msg_txt}</div>'
                 f'</div></div>')
    return (f'<div class="pdash">'
            f'<div class="pdash-hdr"><span>{ico} {esc_label}</span>'
            f'<span class="pdash-ov">{overall}%</span></div>{rows}</div>')


# ══════════════════════════════════════════════════════════════════════════════
#  NAVBAR (Figma & Snitch Floating Header System)
# ══════════════════════════════════════════════════════════════════════════════
with st.container(border=True):
    nav1, nav2, nav3 = st.columns([2.5, 1.3, 2.4], vertical_alignment="center")
    with nav1:
        st.markdown(
            '<div style="display:flex;align-items:center;gap:12px;padding:3px 0;">'
            '<div style="font-size:28px;line-height:1;filter:drop-shadow(0 2px 8px rgba(244,114,43,0.35));">🪷</div>'
            '<div>'
            '<div style="font-family:\'Syne\',sans-serif;font-size:17px;font-weight:800;letter-spacing:-0.3px;'
            'background:var(--pg2);background-size:200%;-webkit-background-clip:text;'
            '-webkit-text-fill-color:transparent;background-clip:text;line-height:1.2;">Bhasha Setu</div>'
            '<div style="font-size:9.5px;font-weight:700;color:var(--tx3);letter-spacing:0.8px;">AI BHARAT DUBBING</div>'
            '</div>'
            '<div style="display:inline-flex;align-items:center;gap:6px;padding:4px 12px;border-radius:9999px;'
            'background:var(--sur2);border:1px solid var(--bdr);font-size:11px;font-weight:600;color:var(--tx2);margin-left:8px;">'
            '<span style="width:7px;height:7px;border-radius:50%;background:#10B981;display:inline-block;box-shadow:0 0 8px #10B981;"></span>'
            'Production Engine v5.2</div>'
            '</div>',
            unsafe_allow_html=True,
        )
    with nav2:
        _lang_opts = list(UI_LANGUAGES.keys())
        _cur_lang_name = [k for k, v in UI_LANGUAGES.items() if v == st.session_state.ui_lang]
        _cur_idx = _lang_opts.index(_cur_lang_name[0]) if _cur_lang_name else 0
        _sel_ui_lang = st.selectbox(
            t("ui_language"), _lang_opts, index=_cur_idx,
            key="ui_lang_sel", label_visibility="collapsed")
        _new_code = UI_LANGUAGES.get(_sel_ui_lang, "en")
        if _new_code != st.session_state.ui_lang:
            st.session_state.ui_lang = _new_code
            st.rerun()

    with nav3:
        user_display = html.escape(str(current_user.get("name") or current_user.get("username", "User")))
        user_role = html.escape(str(current_user.get("role", "user")).capitalize())
        avatar_url = current_user.get("avatar_url", "")
        is_google = current_user.get("auth_provider") == "google"

        uc1, uc2, uc3 = st.columns([1.5, 0.45, 0.65], vertical_alignment="center")
        with uc1:
            if avatar_url:
                esc_ava = html.escape(avatar_url)
                ava_img = f'<img src="{esc_ava}" alt="User Profile Avatar" style="width:24px;height:24px;border-radius:50%;vertical-align:middle;margin-right:6px;border:1.5px solid #4285F4">'
            else:
                ava_img = '<span style="font-size:16px;margin-right:4px;">👤</span>'
            gbadge = '<span class="chip chip-a" style="font-size:9px;padding:2px 6px">G-ID</span> ' if is_google else ''
            st.markdown(
                f'<div style="display:flex;align-items:center;justify-content:flex-end;gap:5px;padding:2px 0;">'
                f'{ava_img}'
                f'<span style="font-size:12px;font-weight:700;color:var(--tx1);white-space:nowrap;overflow:hidden;text-overflow:ellipsis;max-width:110px;">{user_display}</span>'
                f'{gbadge}'
                f'<span class="chip chip-p" style="font-size:9.5px;padding:2px 7px;">{user_role}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )
        with uc2:
            if st.button("🌙" if DM else "☀️", key="mode_toggle", help=t("nav_toggle_dark") if DM else t("nav_toggle_light"), use_container_width=True):
                st.session_state.dark_mode = not DM
                st.rerun()
        with uc3:
            if st.button("Sign Out", key="logout_btn", help="Sign out of your account", use_container_width=True):
                logout_user()

# ══════════════════════════════════════════════════════════════════════════════
#  TABS
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state.get("switch_tab") is not None:
    target_tab = st.session_state.pop("switch_tab")
    import streamlit.components.v1 as _comp
    _comp.html(
        f"""
        <script>
            setTimeout(() => {{
                try {{
                    const tabs = window.parent.document.querySelectorAll('button[data-baseweb="tab"]');
                    if (tabs && tabs[{target_tab}]) {{
                        tabs[{target_tab}].click();
                        tabs[{target_tab}].scrollIntoView({{ behavior: 'smooth', block: 'nearest' }});
                    }}
                }} catch (e) {{}}
            }}, 60);
        </script>
        """,
        height=0,
        width=0,
    )

t_home, t_dub, t_batch, t_hil, t_tr, t_chat, t_hist, t_graph, t_road = st.tabs([
    t("tab_home"), t("tab_dub"), t("tab_batch"), t("tab_review"),
    t("tab_translate"), t("tab_chat"), t("tab_history"), "🕸️ Knowledge Graph", t("tab_roadmap"),
])

# ══════════════════════════════════════════════════════════════════════════════
#  HOME
# ══════════════════════════════════════════════════════════════════════════════
with t_home:
    st.markdown(
        """
        <div style="text-align:center;margin-top:6px;margin-bottom:14px;">
            <div style="display:inline-flex;align-items:center;gap:8px;padding:7px 22px;border-radius:9999px;background:rgba(244,114,43,0.08);border:1px solid rgba(244,114,43,0.24);font-size:12px;font-weight:700;color:var(--p1);letter-spacing:0.6px;box-shadow:var(--sh);">
                <span style="font-size:14px;">✨</span>
                <span>PRODUCTION MULTILINGUAL AI &middot; ENTERPRISE SUITE</span>
                <span style="opacity:0.4">&bull;</span>
                <span style="color:var(--tx2);font-weight:600">Sarvam AI Bulbul:v3 &amp; Mayura:v1 Neural Core</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f'<div class="hero">'
        f'<div class="hero-badge">{t("hero_badge")}</div>'
        f'<div class="h-xl">{t("hero_title_1")}<br><span class="gr">{t("hero_title_2")}</span></div>'
        f'<div class="hero-sub">{t("hero_subtitle")}</div>'
        f'<div class="stats-bar">'
        f'<div class="stat-cell"><div class="stat-n">12+</div><div class="stat-l">Official Indic Languages</div></div>'
        f'<div class="stat-cell"><div class="stat-n">200+</div><div class="stat-l">Translation Pairs</div></div>'
        f'<div class="stat-cell"><div class="stat-n">24 kHz</div><div class="stat-l">HD Studio Voice</div></div>'
        f'<div class="stat-cell"><div class="stat-n">&lt; 3 min</div><div class="stat-l">Automated Dubbing</div></div>'
        f'</div></div>',
        unsafe_allow_html=True)

    # ── Interactive Primary CTA with Automatic Tab Switching ─────────────
    st.markdown('<div style="height: 10px;"></div>', unsafe_allow_html=True)
    cta_col1, cta_col2, cta_col3 = st.columns([1, 1.8, 1])
    with cta_col2:
        if st.button("🚀 Start Dubbing Video Now (Studio) →", type="primary", use_container_width=True, key="hero_primary_cta"):
            st.session_state["switch_tab"] = 1
            st.rerun()

        st.markdown(
            """
            <div style="text-align: center; margin-top: 8px; margin-bottom: 22px;">
                <div style="display: inline-flex; align-items: center; justify-content: center; flex-wrap: wrap; gap: 12px; font-size: 12px; color: var(--tx2);">
                    <span><strong style="color: #10B981;">✓</strong> Instant Zero Setup</span>
                    <span style="color: var(--tx3);">&bull;</span>
                    <span><strong style="color: #10B981;">✓</strong> 12+ Indian Regional Languages</span>
                    <span style="color: var(--tx3);">&bull;</span>
                    <span><strong style="color: #10B981;">✓</strong> Sarvam AI Bulbul:v3 &amp; Mayura:v1</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ── Quick Launch Navigation Pills ────────────────────────────────────
    st.markdown(
        """
        <div style="text-align:center;margin-bottom:24px;">
            <div class="label" style="margin-bottom:6px;">⚡ QUICK EXPLORE HUBS</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    q_c1, q_c2, q_c3, q_c4 = st.columns(4)
    with q_c1:
        if st.button("📦 Batch Dub 12+ Langs", key="quick_batch_btn", use_container_width=True):
            st.session_state["switch_tab"] = 2
            st.rerun()
    with q_c2:
        if st.button("🔬 Review & Dub Studio", key="quick_hil_btn", use_container_width=True):
            st.session_state["switch_tab"] = 3
            st.rerun()
    with q_c3:
        if st.button("🌐 Text Translator", key="quick_tr_btn", use_container_width=True):
            st.session_state["switch_tab"] = 4
            st.rerun()
    with q_c4:
        if st.button("💬 Video AI Assistant", key="quick_chat_btn", use_container_width=True):
            st.session_state["switch_tab"] = 5
            st.rerun()

    # ── HIGH-MODERN TECH TOOLS & ENTERPRISE ARCHITECTURE ECOSYSTEM ──────
    st.markdown('<div class="idiv" style="margin:30px 0 20px;"></div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div style="text-align:center;margin-bottom:22px;">
            <div class="eyebrow">MODERN ENTERPRISE TECH STACK &amp; SOFTWARE ARCHITECTURE</div>
            <div class="h-sec">Powered by World-Class AI Models &amp; Cloud Systems</div>
            <div style="font-size:13.5px;color:var(--tx2);max-width:680px;margin:0 auto;line-height:1.7;">
                Bhasha Setu integrates high-modern AI tools and infrastructure for studio-grade multilingual dubbing, sub-word synchronization, and contextual Indic translation.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    t_r1_c1, t_r1_c2, t_r1_c3 = st.columns(3, gap="medium")
    with t_r1_c1:
        with st.container(border=True):
            st.markdown(
                """
                <div style="display:flex;align-items:center;gap:10px;margin-bottom:10px;">
                    <span style="font-size:26px;">🌟</span>
                    <div>
                        <div style="font-family:'Syne',sans-serif;font-size:15px;font-weight:700;color:var(--tx1);">Sarvam AI Indic Suite</div>
                        <div style="font-size:11px;font-weight:700;color:var(--p1);">Bulbul:v3 &amp; Mayura:v1 Core</div>
                    </div>
                </div>
                <div style="font-size:12.5px;color:var(--tx2);line-height:1.7;margin-bottom:12px;">
                    10 Indic Native Speakers (Priya, Aditya, Rahul, Simran, etc.) with regional cadence calibration and context-aware NMT preserving cultural idioms and honorifics.
                </div>
                <div style="display:flex;flex-wrap:wrap;gap:6px;">
                    <span class="chip chip-p" style="font-size:10px;">10 Speakers</span>
                    <span class="chip chip-p" style="font-size:10px;">Idiom Preservation</span>
                    <span class="chip chip-p" style="font-size:10px;">24kHz Studio</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
    with t_r1_c2:
        with st.container(border=True):
            st.markdown(
                """
                <div style="display:flex;align-items:center;gap:10px;margin-bottom:10px;">
                    <span style="font-size:26px;">☁️</span>
                    <div>
                        <div style="font-family:'Syne',sans-serif;font-size:15px;font-weight:700;color:var(--tx1);">AWS Cloud Computing</div>
                        <div style="font-size:11px;font-weight:700;color:var(--a1);">S3 + Transcribe + Polly</div>
                    </div>
                </div>
                <div style="font-size:12.5px;color:var(--tx2);line-height:1.7;margin-bottom:12px;">
                    High-throughput cloud storage with token authorization, automated speech language identification (99.2% accuracy), and microsecond word timestamps for lip-sync.
                </div>
                <div style="display:flex;flex-wrap:wrap;gap:6px;">
                    <span class="chip chip-a" style="font-size:10px;">99.2% Acc</span>
                    <span class="chip chip-a" style="font-size:10px;">Word Timestamps</span>
                    <span class="chip chip-a" style="font-size:10px;">S3 Streaming</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
    with t_r1_c3:
        with st.container(border=True):
            st.markdown(
                """
                <div style="display:flex;align-items:center;gap:10px;margin-bottom:10px;">
                    <span style="font-size:26px;">⚡</span>
                    <div>
                        <div style="font-family:'Syne',sans-serif;font-size:15px;font-weight:700;color:var(--tx1);">Groq LPU Inference</div>
                        <div style="font-size:11px;font-weight:700;color:#10B981;">Llama 3.1 8B Ultra-Fast</div>
                    </div>
                </div>
                <div style="font-size:12.5px;color:var(--tx2);line-height:1.7;margin-bottom:12px;">
                    Sub-200ms ultra-low latency inference for conversational translation polishing, video intelligence Q&A, and real-time multilingual executive summaries.
                </div>
                <div style="display:flex;flex-wrap:wrap;gap:6px;">
                    <span class="chip chip-g" style="font-size:10px;">&lt;200ms Latency</span>
                    <span class="chip chip-g" style="font-size:10px;">Context Aware</span>
                    <span class="chip chip-g" style="font-size:10px;">Video Q&amp;A</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

    t_r2_c1, t_r2_c2, t_r2_c3 = st.columns(3, gap="medium")
    with t_r2_c1:
        with st.container(border=True):
            st.markdown(
                """
                <div style="display:flex;align-items:center;gap:10px;margin-bottom:10px;">
                    <span style="font-size:26px;">🎬</span>
                    <div>
                        <div style="font-family:'Syne',sans-serif;font-size:15px;font-weight:700;color:var(--tx1);">Dual-Pass FFmpeg 7.0</div>
                        <div style="font-size:11px;font-weight:700;color:#FBBF24;">Atempo Phase Vocoder</div>
                    </div>
                </div>
                <div style="font-size:12.5px;color:var(--tx2);line-height:1.7;margin-bottom:12px;">
                    Lossless atempo audio-video alignment dynamically matches target speech duration to original video mouth pauses with sub-pixel burn-in SRT subtitle overlay.
                </div>
                <div style="display:flex;flex-wrap:wrap;gap:6px;">
                    <span class="chip" style="font-size:10px;color:#FBBF24;border-color:rgba(251,191,36,.3);">Atempo Sync</span>
                    <span class="chip" style="font-size:10px;color:#FBBF24;border-color:rgba(251,191,36,.3);">Burn-in Subtitles</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
    with t_r2_c2:
        with st.container(border=True):
            st.markdown(
                """
                <div style="display:flex;align-items:center;gap:10px;margin-bottom:10px;">
                    <span style="font-size:26px;">🛡️</span>
                    <div>
                        <div style="font-family:'Syne',sans-serif;font-size:15px;font-weight:700;color:var(--tx1);">Google Identity &amp; Auth</div>
                        <div style="font-size:11px;font-weight:700;color:#818CF8;">OAuth 2.0 PKCE &amp; PBKDF2</div>
                    </div>
                </div>
                <div style="font-size:12.5px;color:var(--tx2);line-height:1.7;margin-bottom:12px;">
                    Enterprise Single Sign-On (SSO), PBKDF2 HMAC-SHA256 password hashing, token-bucket anti-abuse rate limiting, and tamper-evident audit logging.
                </div>
                <div style="display:flex;flex-wrap:wrap;gap:6px;">
                    <span class="chip chip-a" style="font-size:10px;">Google SSO</span>
                    <span class="chip chip-a" style="font-size:10px;">PBKDF2 SHA-256</span>
                    <span class="chip chip-a" style="font-size:10px;">RBAC Protected</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
    with t_r2_c3:
        with st.container(border=True):
            st.markdown(
                """
                <div style="display:flex;align-items:center;gap:10px;margin-bottom:10px;">
                    <span style="font-size:26px;">🔷</span>
                    <div>
                        <div style="font-family:'Syne',sans-serif;font-size:15px;font-weight:700;color:var(--tx1);">Multi-Engine Redundancy</div>
                        <div style="font-size:11px;font-weight:700;color:#C084FC;">Edge-TTS &amp; Google Universal</div>
                    </div>
                </div>
                <div style="font-size:12.5px;color:var(--tx2);line-height:1.7;margin-bottom:12px;">
                    Automatic multi-tiered failover pipeline guarantees 99.99% service availability with high-definition neural voice synthesis across all 12 scheduled languages.
                </div>
                <div style="display:flex;flex-wrap:wrap;gap:6px;">
                    <span class="chip" style="font-size:10px;color:#C084FC;border-color:rgba(192,132,252,.3);">99.99% Uptime</span>
                    <span class="chip" style="font-size:10px;color:#C084FC;border-color:rgba(192,132,252,.3);">Auto Failover</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # ── INTERACTIVE STUDIO VOICE PLAYGROUND ──────────────────────────────
    st.markdown('<div class="idiv" style="margin:28px 0 20px;"></div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div style="text-align:center;margin-bottom:18px;">
            <div class="eyebrow">INTERACTIVE VOICE AUDITION STUDIO</div>
            <div class="h-sec">Audition 10+ Authentic Indic Neural Speakers</div>
            <div style="font-size:13px;color:var(--tx2);">Preview the regional cadence, pitch, and native delivery before starting your dubbing job:</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.container(border=True):
        vp_col1, vp_col2 = st.columns([1.2, 1.8], gap="large")
        with vp_col1:
            st.markdown('<div class="label">Select Language to Audition</div>', unsafe_allow_html=True)
            aud_lang = st.selectbox(
                "Audition Language",
                list(LANGUAGES.keys()),
                index=0,
                key="audition_lang_sel",
                label_visibility="collapsed",
            )
            st.markdown('<div class="label" style="margin-top:10px;">Select Native Speaker</div>', unsafe_allow_html=True)
            aud_speakers = [
                ("Priya", "Female · Natural, Expressive & Melodic", "Delhi / UP / MP"),
                ("Aditya", "Male · Clear, Deep Narration", "Delhi / Haryana / MP"),
                ("Ritu", "Female · Polite, Engaging & Professional", "Pan-India"),
                ("Rahul", "Male · Steady, Authoritative & Warm", "Pan-India"),
                ("Pooja", "Female · Melodic, Friendly & Conversational", "Tamil Nadu / Chennai"),
                ("Rohan", "Male · Modern Youthful & Dynamic", "Mumbai / Pune"),
                ("Shreya", "Female · Clear Storytelling & Audiobooks", "Kolkata / Bengal"),
                ("Simran", "Female · Expressive Indic Tone & Passion", "Punjab / Chandigarh"),
                ("Dev", "Male · Energetic, Broadcast & Commercial", "Gujarat"),
                ("Amit", "Male · Calm, Trustworthy & Corporate", "Pan-India"),
            ]
            spk_display_list = [f"{s[0]} ({s[1]})" for s in aud_speakers]
            sel_spk_idx = st.selectbox("Audition Speaker", spk_display_list, index=0, key="audition_spk_sel", label_visibility="collapsed")
            chosen_spk = aud_speakers[spk_display_list.index(sel_spk_idx)]

        with vp_col2:
            l_info = LANGUAGES.get(aud_lang, {})
            native_script = l_info.get("native_name", aud_lang)
            st.markdown(
                f"""
                <div style="background:var(--sur2);border:1px solid var(--bdr2);border-radius:24px;padding:18px 22px;margin-bottom:14px;">
                    <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:8px;">
                        <span style="font-family:'Syne',sans-serif;font-size:20px;font-weight:800;color:var(--p1);">{native_script} &middot; {aud_lang}</span>
                        <span class="chip chip-p" style="font-size:10.5px;">Sarvam Bulbul:v3</span>
                    </div>
                    <div style="font-size:14px;font-weight:700;color:var(--tx1);margin-bottom:4px;">Speaker: {chosen_spk[0]}</div>
                    <div style="font-size:12px;color:var(--tx2);margin-bottom:10px;">Voice Profile: <em>{chosen_spk[1]}</em> &bull; Region: {chosen_spk[2]}</div>
                    <div style="background:var(--glass);border:1px solid var(--bdr);border-radius:16px;padding:12px 16px;font-size:13.5px;color:var(--tx1);line-height:1.7;">
                        💬 <em>&ldquo;नमस्ते! मैं भाषा सेतु AI हूँ। मैं आपकी वीडियो को अत्यंत स्वाभाविक और सटीक भारतीय क्षेत्रीय आवाज़ में डब कर सकता हूँ।&rdquo;</em>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if st.button(f"🎙️ Launch Studio Dubbing with {chosen_spk[0]} in {aud_lang} →", type="primary", use_container_width=True, key="btn_aud_launch"):
                st.session_state["sel_lang"] = aud_lang
                st.session_state["switch_tab"] = 1
                st.rerun()

    # ── LIVE INTERACTIVE DUBBING RESOURCE & SPEED ESTIMATOR ─────────────
    st.markdown('<div class="idiv" style="margin:28px 0 20px;"></div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div style="text-align:center;margin-bottom:18px;">
            <div class="eyebrow">PIPELINE PERFORMANCE CALCULATOR</div>
            <div class="h-sec">Live Dubbing Speed &amp; Resource Estimator</div>
            <div style="font-size:13px;color:var(--tx2);">Calculate the processing duration, sampling frequency, and concurrency budget for your video:</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.container(border=True):
        calc_c1, calc_c2 = st.columns([1.2, 1.8], gap="large")
        with calc_c1:
            st.markdown('<div class="label">Video Duration (Minutes)</div>', unsafe_allow_html=True)
            v_mins = st.slider("Duration", 1, 30, 3, key="home_calc_mins", label_visibility="collapsed")
            st.markdown('<div class="label" style="margin-top:10px;">Number of Target Languages</div>', unsafe_allow_html=True)
            t_count = st.slider("Target Languages", 1, 12, 1, key="home_calc_langs", label_visibility="collapsed")
        with calc_c2:
            est_seconds = int((v_mins * 22) + (t_count * 18))
            est_time_str = f"{est_seconds // 60}m {est_seconds % 60}s" if est_seconds >= 60 else f"{est_seconds}s"
            est_words = v_mins * 135
            est_chars = est_words * 6

            st.markdown(
                f"""
                <div style="display:grid;grid-template-columns:repeat(2,1fr);gap:12px;">
                    <div style="background:var(--sur2);border:1px solid var(--bdr);border-radius:20px;padding:14px 18px;text-align:center;">
                        <div style="font-size:11px;font-weight:700;color:var(--tx3);text-transform:uppercase;">Estimated Dub Time</div>
                        <div style="font-family:'Syne',sans-serif;font-size:24px;font-weight:800;color:var(--p1);margin-top:2px;">~{est_time_str}</div>
                        <div style="font-size:10.5px;color:#10B981;font-weight:600;margin-top:2px;">&bull; Parallel NVMe Mux</div>
                    </div>
                    <div style="background:var(--sur2);border:1px solid var(--bdr);border-radius:20px;padding:14px 18px;text-align:center;">
                        <div style="font-size:11px;font-weight:700;color:var(--tx3);text-transform:uppercase;">Master Audio Quality</div>
                        <div style="font-family:'Syne',sans-serif;font-size:24px;font-weight:800;color:var(--a1);margin-top:2px;">24,000 Hz</div>
                        <div style="font-size:10.5px;color:var(--tx2);font-weight:600;margin-top:2px;">&bull; Studio-Grade AAC</div>
                    </div>
                    <div style="background:var(--sur2);border:1px solid var(--bdr);border-radius:20px;padding:14px 18px;text-align:center;">
                        <div style="font-size:11px;font-weight:700;color:var(--tx3);text-transform:uppercase;">Estimated Words</div>
                        <div style="font-family:'Syne',sans-serif;font-size:24px;font-weight:800;color:var(--tx1);margin-top:2px;">~{est_words:,}</div>
                        <div style="font-size:10.5px;color:var(--tx3);margin-top:2px;">~{est_chars:,} Translated Chars</div>
                    </div>
                    <div style="background:var(--sur2);border:1px solid var(--bdr);border-radius:20px;padding:14px 18px;text-align:center;">
                        <div style="font-size:11px;font-weight:700;color:var(--tx3);text-transform:uppercase;">Pipeline Concurrency</div>
                        <div style="font-family:'Syne',sans-serif;font-size:24px;font-weight:800;color:#10B981;margin-top:2px;">{t_count} Stream{'s' if t_count > 1 else ''}</div>
                        <div style="font-size:10.5px;color:var(--tx3);margin-top:2px;">Zero Pipeline Bottleneck</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # ── BEFORE VS AFTER TRANSFORMATION COMPARISON ────────────────────────
    st.markdown('<div class="idiv" style="margin:28px 0 20px;"></div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div style="text-align:center;margin-bottom:18px;">
            <div class="eyebrow">LOCALIZATION TRANSFORMATION MATRIX</div>
            <div class="h-sec">Before &amp; After Dubbing with Bhasha Setu</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    b_col1, b_col2 = st.columns(2, gap="medium")
    with b_col1:
        with st.container(border=True):
            st.markdown(
                """
                <div style="font-family:'Syne',sans-serif;font-size:15px;font-weight:700;color:var(--tx3);margin-bottom:10px;">
                    ❌ Raw / Monolingual Source Video
                </div>
                <div style="font-size:12.5px;color:var(--tx2);line-height:1.8;">
                    &bull; <strong>Audience Reach:</strong> Restricted only to native English or single-language speakers.<br>
                    &bull; <strong>Subtitles:</strong> Absent or poorly auto-generated without punctuation.<br>
                    &bull; <strong>Audio Sync:</strong> No regional language dialogue exists.<br>
                    &bull; <strong>Interactivity:</strong> Passive viewing with zero multilingual Q&amp;A or AI comprehension.
                </div>
                """,
                unsafe_allow_html=True,
            )
    with b_col2:
        with st.container(border=True):
            st.markdown(
                """
                <div style="font-family:'Syne',sans-serif;font-size:15px;font-weight:700;color:#10B981;margin-bottom:10px;">
                    ✅ Bhasha Setu AI Studio Dubbed Output
                </div>
                <div style="font-size:12.5px;color:var(--tx1);line-height:1.8;">
                    &bull; <strong>Audience Reach:</strong> Scaled to 1.4+ Billion speakers across 12+ Indian regions.<br>
                    &bull; <strong>Subtitles:</strong> Burned-in word-level SRT with crisp font anti-aliasing.<br>
                    &bull; <strong>Audio Sync:</strong> Lossless atempo phase vocoder matches speech duration to mouth pauses.<br>
                    &bull; <strong>Interactivity:</strong> Instant AI Video Summary and interactive Q&amp;A in the dubbed language.
                </div>
                """,
                unsafe_allow_html=True,
            )

    # ── 12+ Indian Regional Languages Showcase Grid ──────────────────────
    st.markdown('<div class="idiv" style="margin:28px 0 16px;"></div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div style="text-align:center;margin-bottom:18px;">
            <div class="eyebrow">INDIVIDUAL REGIONAL VOICES</div>
            <div class="h-sec">12+ Official Indian Regional Languages</div>
            <div style="font-size:13px;color:var(--tx2);">Click any language card below to launch the dubbing studio pre-configured for that language:</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    indic_langs = [
        ("Hindi", "हिन्दी", "#E05C1A", "HI", "Delhi/UP/MP", "Priya & Aditya"),
        ("Punjabi", "ਪੰਜਾਬੀ", "#4F46E5", "PB", "Punjab/Chd", "Simran & Rahul"),
        ("Tamil", "தமிழ்", "#C0392B", "TA", "Chennai/TN", "Pooja & Vijay"),
        ("Telugu", "తెలుగు", "#7B2FBE", "TE", "Hyd/Andhra", "Kavitha & Tarun"),
        ("Kannada", "ಕನ್ನಡ", "#0891B2", "KN", "Bengaluru/KA", "Chaitra & Chetan"),
        ("Malayalam", "മലയാളം", "#059669", "ML", "Kerala", "Midhun & Ananya"),
        ("Bengali", "বাংলা", "#D97706", "BN", "Kolkata/WB", "Roopa & Arnab"),
        ("Marathi", "मराठी", "#BE185D", "MR", "Mumbai/Pune", "Ishita & Soham"),
        ("Gujarati", "ગુજરાતી", "#0D9488", "GU", "Ahmedabad/GJ", "Pooja & Bhavik"),
        ("Odia", "ଓଡ଼ିଆ", "#0284C7", "OD", "Bhubaneswar/OD", "Sarita & Subrat"),
        ("Assamese", "অসমীয়া", "#EA580C", "AS", "Guwahati/AS", "Kangkana & Moni"),
        ("Urdu", "اردو", "#65A30D", "UR", "Pan-India", "Rizwan & Zoya"),
    ]

    ig_rows = [indic_langs[:4], indic_langs[4:8], indic_langs[8:]]
    for r_langs in ig_rows:
        i_cols = st.columns(4, gap="small")
        for col, (l_name, l_native, l_col, l_code, l_reg, l_spk) in zip(i_cols, r_langs):
            with col:
                with st.container(border=True):
                    st.markdown(
                        f"""
                        <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:6px;">
                            <span style="font-family:'Syne',sans-serif;font-size:18px;font-weight:800;color:{l_col};">{l_native}</span>
                            <span class="chip" style="font-size:10px;font-weight:700;background:{l_col}22;color:{l_col};border-color:{l_col}44;">{l_code}</span>
                        </div>
                        <div style="font-size:13.5px;font-weight:700;color:var(--tx1);">{l_name}</div>
                        <div style="font-size:11px;color:var(--tx3);margin-bottom:10px;">Region: {l_reg} &bull; Voices: {l_spk}</div>
                        """,
                        unsafe_allow_html=True,
                    )
                    if st.button(f"Dub in {l_name} →", key=f"grid_btn_{l_name}", use_container_width=True):
                        st.session_state["sel_lang"] = l_name
                        st.session_state["switch_tab"] = 1
                        st.rerun()

    # ── Platform Capabilities Grid ───────────────────────────────────────
    st.markdown('<div class="idiv" style="margin:30px 0 16px;"></div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div style="text-align:center;margin-bottom:20px;">
            <div class="eyebrow">ENTERPRISE HIGHLIGHTS</div>
            <div class="h-sec">Core Platform Innovations</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    fa, fb = st.columns(2, gap="large")
    feats = [
        ("🌍", t("feat_any_lang_title"), t("feat_any_lang_desc")),
        ("🔍", t("feat_auto_detect_title"), t("feat_auto_detect_desc")),
        ("🗣️", t("feat_neural_tts_title"), t("feat_neural_tts_desc")),
        ("⚡", t("feat_audio_sync_title"), t("feat_audio_sync_desc")),
        ("🧠", t("feat_llm_polish_title"), t("feat_llm_polish_desc")),
        ("📝", t("feat_srt_title"), t("feat_srt_desc")),
        ("🔄", t("feat_batch_title"), t("feat_batch_desc")),
        ("🔬", t("feat_hil_title"), t("feat_hil_desc")),
    ]
    for i, (icon, title, desc) in enumerate(feats):
        col = fa if i % 2 == 0 else fb
        with col:
            st.markdown(
                f'<div style="background:var(--glass);border:1px solid var(--bdr);border-radius:24px;'
                f'padding:20px;margin-bottom:14px;transition:all .2s;animation:fadeUp .4s ease both;'
                f'animation-delay:{i*.05:.2f}s;box-shadow:var(--sh);">'
                f'<div style="font-size:24px;margin-bottom:9px">{icon}</div>'
                f'<div style="font-family:\'Syne\',sans-serif;font-size:14.5px;font-weight:700;'
                f'color:var(--tx1);margin-bottom:5px">{title}</div>'
                f'<div style="font-size:12.5px;color:var(--tx2);line-height:1.7">{desc}</div></div>',
                unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  DUB VIDEO
# ══════════════════════════════════════════════════════════════════════════════
with t_dub:
    da, db = st.columns([1.15, 0.85], gap="large")

    with da:
        st.markdown(f'<div class="eyebrow">{t("label_configure")}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="h-sec" style="margin-bottom:16px">{t("title_dub_video")}</div>', unsafe_allow_html=True)

        with st.container(border=True):
            # ── Source language ───────────────────────────────────────
            st.markdown(f'<div class="label">{t("label_source_lang")}</div>', unsafe_allow_html=True)
            src_lang_opts = list(SOURCE_LANGUAGES.keys()) if PIPELINE_READY else list(LANG_CODES.keys())
            sel_src = st.selectbox(
                t("label_source_lang"), src_lang_opts,
                index=src_lang_opts.index(st.session_state.sel_src_lang)
                      if st.session_state.sel_src_lang in src_lang_opts else 0,
                key="src_lang_sel", label_visibility="collapsed",
                help="Language spoken in the video. Choose Auto-detect to let AI identify it.",
            )
            st.session_state.sel_src_lang = sel_src
            if sel_src == "Auto-detect":
                st.markdown(
                    f'<div style="font-size:11.5px;color:var(--a1);padding:5px 0">'
                    f'{t("auto_detect_hint")}</div>',
                    unsafe_allow_html=True)

            st.markdown('<div class="idiv"></div>', unsafe_allow_html=True)

            # ── Target language ────────────────────────────────────────
            st.markdown(f'<div class="label">{t("label_target_lang")}</div>', unsafe_allow_html=True)
            lang_list = list(LANGUAGES.keys())
            sel_lang  = st.selectbox(
                t("label_target_lang"), lang_list,
                index=lang_list.index(st.session_state.sel_lang)
                      if st.session_state.sel_lang in lang_list else 0,
                key="tgt_lang_sel", label_visibility="collapsed",
            )
            st.session_state.sel_lang = sel_lang

            # Visual language direction strip
            cfg = LANGUAGES.get(sel_lang, {})
            st.markdown(
                f'<div class="lang-strip">'
                f'<span class="lang-pill">{sel_src}</span>'
                f'<span class="lang-arrow">→</span>'
                f'{lbadge(sel_lang, 32, 8)}'
                f'<div style="margin-left:4px"><div style="font-size:13px;font-weight:700;'
                f'color:var(--tx1)">{sel_lang}</div>'
                f'<div style="font-size:11px;color:var(--tx3)">{cfg.get("native_name","")}'
                f' · {cfg.get("tts","").upper()}</div></div>'
                f'</div>',
                unsafe_allow_html=True)

        # ── Input source ───────────────────────────────────────────────
        with st.container(border=True):
            st.markdown(f'<div class="label">{t("label_video_source")}</div>', unsafe_allow_html=True)
            inp_mode = st.radio("Input", [t("label_upload_file"), t("label_paste_url")], horizontal=True,
                                key="dub_input_mode", label_visibility="collapsed")

            uploaded_file = None
            video_url_val = ""

            if inp_mode == t("label_upload_file"):
                uploaded_file = st.file_uploader(t("upload_mp4"), type=["mp4","mov","avi","mkv"],
                                                  key="dub_uploader", label_visibility="collapsed")
            else:
                video_url_val = st.text_input("Video URL",
                    placeholder=t("video_url_placeholder"),
                    key="dub_url_input", label_visibility="collapsed",
                    value=st.session_state.url_text)
                st.session_state.url_text = video_url_val

                if video_url_val and PIPELINE_READY and st.button(t("btn_fetch_info"), key="fetch_info_btn"):
                    with st.spinner(t("status_fetching_info")):
                        try:
                            info = get_video_info(video_url_val)
                            st.session_state.url_info = info
                        except Exception as ex:
                            st.error(str(ex))
                if st.session_state.url_info:
                    info = st.session_state.url_info
                    d_dur = info.get("duration") or 0
                    dur   = f"{int(d_dur // 60)}:{int(d_dur % 60):02d}"
                    esc_title = html.escape(str(info.get("title", "")))[:60]
                    esc_platform = html.escape(str(info.get("platform", "")))
                    st.markdown(
                        f'<div style="background:var(--gs);border:1px solid var(--gb);'
                        f'border-radius:9px;padding:10px 13px;margin-top:8px">'
                        f'<div style="font-size:12.5px;font-weight:700;color:var(--g1)">'
                        f'{esc_title}</div>'
                        f'<div style="font-size:11px;color:var(--tx3);margin-top:2px">'
                        f'{esc_platform} · {dur}</div></div>',
                        unsafe_allow_html=True)

        # ── AI Model & Engine Architecture (Sarvam vs AWS vs Edge) ────
        with st.container(border=True):
            st.markdown(
                '<div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:12px">'
                '<div style="display:flex;align-items:center;gap:7px">'
                '<span style="font-size:17px">⚡</span>'
                '<span style="font-family:Syne,sans-serif;font-size:13.5px;font-weight:700;color:var(--tx1)">'
                'AI Model &amp; Synthesis Architecture</span></div>'
                '<span class="chip chip-p" style="font-size:10px">Context-Aware AI</span></div>',
                unsafe_allow_html=True,
            )

            ec1, ec2 = st.columns(2)
            with ec1:
                st.markdown(f'<div class="label">🌐 Translation Model</div>', unsafe_allow_html=True)
                trans_opts = [
                    "Auto (Sarvam Mayura Indic / AWS Cloud)",
                    "Sarvam AI (Mayura:v1 Indic Neural)",
                    "AWS Translate (Cloud Neural)",
                    "Google / Deep Translator (Universal)",
                ]
                sel_trans_disp = st.selectbox("Translation Model", trans_opts, index=0, key="dub_trans_sel", label_visibility="collapsed")
                trans_map = {
                    "Auto (Sarvam Mayura Indic / AWS Cloud)": "auto",
                    "Sarvam AI (Mayura:v1 Indic Neural)": "sarvam",
                    "AWS Translate (Cloud Neural)": "aws",
                    "Google / Deep Translator (Universal)": "google",
                }
                dub_chosen_translate_engine = trans_map[sel_trans_disp]

            with ec2:
                st.markdown(f'<div class="label">🎙️ Neural Voice Engine</div>', unsafe_allow_html=True)
                tts_engine_opts = [
                    "Auto (Sarvam Bulbul Indic / Edge-TTS)",
                    "Sarvam AI (Bulbul:v3 Indic Neural)",
                    "AWS Polly (Neural Voice)",
                    "Edge-TTS (High-Definition Neural)",
                    "Google TTS (Standard)",
                ]
                sel_engine_display = st.selectbox("Voice Engine", tts_engine_opts, index=0, key="dub_tts_engine_sel", label_visibility="collapsed")
                tts_engine_map = {
                    "Auto (Sarvam Bulbul Indic / Edge-TTS)": "auto",
                    "Sarvam AI (Bulbul:v3 Indic Neural)": "sarvam",
                    "AWS Polly (Neural Voice)": "polly",
                    "Edge-TTS (High-Definition Neural)": "edge",
                    "Google TTS (Standard)": "gtts",
                }
                dub_chosen_engine = tts_engine_map[sel_engine_display]

            sarvam_speaker_val = "priya"
            if dub_chosen_engine in ("sarvam", "auto") or dub_chosen_translate_engine in ("sarvam", "auto"):
                st.markdown('<div class="idiv" style="margin:10px 0 8px"></div>', unsafe_allow_html=True)
                sv_c1, sv_c2 = st.columns([1.3, 1.7])
                with sv_c1:
                    st.markdown(f'<div class="label">🌟 Sarvam Voice Speaker</div>', unsafe_allow_html=True)
                    spk_names = [
                        "Priya (Female · Natural & Expressive)",
                        "Aditya (Male · Clear Narration)",
                        "Ritu (Female · Polite & Engaging)",
                        "Rahul (Male · Deep & Steady)",
                        "Pooja (Female · Melodic & Friendly)",
                        "Rohan (Male · Modern Youth)",
                        "Shreya (Female · Clear Storytelling)",
                        "Simran (Female · Expressive Indic Tone)",
                        "Dev (Male · Energetic & Dynamic)",
                        "Amit (Male · Calm & Trustworthy)",
                    ]
                    sel_spk_disp = st.selectbox("Sarvam Voice Speaker", spk_names, index=0, key="dub_sarvam_spk_sel", label_visibility="collapsed")
                    sarvam_speaker_val = sel_spk_disp.split(" ")[0].lower()
                with sv_c2:
                    st.markdown(
                        '<div style="font-size:11.5px;color:var(--tx2);padding-top:22px;line-height:1.6">'
                        '✨ <strong style="color:var(--p1)">Sarvam AI Bulbul &amp; Mayura</strong> are context-optimized for Indian linguistic nuances, regional accents, and authentic pitch.</div>',
                        unsafe_allow_html=True,
                    )

        # ── Audio Controls & Fine-Tuning ──────────────────────────────
        with st.expander("🎛️ Audio Controls & Subtitle Customization", expanded=False):
            c1, c2 = st.columns(2)
            with c1:
                gen_srt    = st.checkbox(t("opt_generate_srt"), key="dub_srt")
                polish     = st.checkbox(t("opt_llm_polish"), key="dub_polish",
                                         help="Groq AI polish improves conversational naturalness")
                hil_enable = st.checkbox(t("opt_hil_review"), key="dub_hil",
                                         help="Review transcript + translation before TTS")
            with c2:
                preview_secs = st.slider(t("opt_preview_secs"), 0, 60, 0, key="dub_prev_secs")
                vol_boost    = st.slider(t("opt_voice_volume"), 1.0, 4.0, 2.0, 0.5, key="dub_vol")
                pitch_pct    = st.slider(t("opt_voice_pitch"), -20, 20, 0, key="dub_pitch")
                bg_music     = st.slider(t("opt_bg_music"), 0.0, 0.5, 0.0, 0.05, key="dub_bgm",
                                         help=t("opt_bg_music_help"))

        # ── Run button ─────────────────────────────────────────────────
        can_run = bool(uploaded_file or video_url_val) and PIPELINE_READY
        if st.button(t("btn_start_dubbing"), key="dub_run", type="primary",
                     use_container_width=True, disabled=not can_run):

            if not PIPELINE_READY:
                st.error(t("err_pipeline_not_ready"))
            elif not uploaded_file and not video_url_val:
                st.warning(t("err_no_file_or_url"))
            else:
                is_allowed, retry_sec = check_rate_limit(f"dub_{current_user['id']}", max_requests=5, window_seconds=300)
                if not is_allowed:
                    st.error(f"🛑 Security Rate Limit: You can submit at most 5 dubbing requests every 5 minutes. Please wait {retry_sec}s.")
                    st.stop()
                log_audit_event("dubbing_started", current_user["id"], current_user["username"], f"{sel_src} -> {sel_lang}")

                st.session_state.running    = True
                st.session_state.result     = None
                st.session_state.pcts       = [0.0]*5
                st.session_state.msgs       = [""]*5
                st.session_state.cur_stage  = 0
                st.session_state.hil_phase  = "idle"

                progress_placeholder = st.empty()

                def _cb(stage, pct, msg):
                    st.session_state.pcts[stage-1] = pct
                    st.session_state.msgs[stage-1] = msg
                    st.session_state.cur_stage = stage
                    progress_placeholder.markdown(
                        pdash_html(st.session_state.pcts, st.session_state.msgs, stage, t("status_dubbing")),
                        unsafe_allow_html=True)

                video_path = ""
                try:
                    if uploaded_file:
                        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
                        tmp.write(uploaded_file.read()); tmp.flush(); tmp.close()
                        video_path = tmp.name

                    effective_secs = preview_secs if preview_secs > 0 else None

                    if hil_enable:
                        # Phase 1: transcribe + translate only
                        r1 = run_transcribe_and_translate(
                            video_path=video_path,
                            video_url=video_url_val if not uploaded_file else "",
                            target_language=sel_lang,
                            progress_cb=_cb,
                            generate_srt=gen_srt,
                            polish_translation=polish,
                            source_language=sel_src,
                            translate_engine=dub_chosen_translate_engine,
                            tts_engine=dub_chosen_engine,
                            sarvam_speaker=sarvam_speaker_val,
                        )
                        st.session_state.hil_data  = r1
                        st.session_state.hil_phase = "review"
                        st.session_state.hil_tmp   = r1.get("video_path", video_path)
                    else:
                        result = run_pipeline(
                            video_path=video_path,
                            video_url=video_url_val if not uploaded_file else "",
                            target_language=sel_lang,
                            progress_cb=_cb,
                            generate_srt=gen_srt,
                            polish_translation=polish,
                            voice_pitch=pitch_pct,
                            vol_boost=vol_boost,
                            bg_music_vol=bg_music,
                            source_language=sel_src,
                            translate_engine=dub_chosen_translate_engine,
                            tts_engine=dub_chosen_engine,
                            sarvam_speaker=sarvam_speaker_val,
                        )

                        if effective_secs:
                            prev = clip_preview(result["output_path"], effective_secs)
                            result["preview_path"] = prev

                        st.session_state.result  = result
                        st.session_state.running = False
                        save_history({
                            "job_id":       result["job_id"],
                            "language":     sel_lang,
                            "source_lang":  sel_src,
                            "output_path":  result["output_path"],
                            "transcript_len": len(result.get("transcript","")),
                            "timestamp":    datetime.now().isoformat(),
                        })
                        progress_placeholder.markdown(
                            pdash_html(st.session_state.pcts, st.session_state.msgs, 5,
                                       t("status_complete"), done=True),
                            unsafe_allow_html=True)
                except Exception as ex:
                    st.session_state.running = False
                    progress_placeholder.markdown(
                        pdash_html(st.session_state.pcts, st.session_state.msgs,
                                   st.session_state.cur_stage, str(ex)[:80], err=True),
                        unsafe_allow_html=True)
                    st.error(f"**{t('err_pipeline_error')}:** {ex}")
                finally:
                    if video_path and os.path.exists(video_path):
                        if not hil_enable or st.session_state.get("hil_phase") != "review":
                            try: os.remove(video_path)
                            except Exception: pass

        elif not PIPELINE_READY:
            st.info(t("err_install_deps"))

    # ── Results panel ──────────────────────────────────────────────────────────
    with db:
        st.markdown(f'<div class="eyebrow">{t("label_output")}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="h-sec" style="margin-bottom:16px">{t("label_result")}</div>', unsafe_allow_html=True)

        # Human-in-the-loop review panel
        if st.session_state.hil_phase == "review" and st.session_state.hil_data:
            d = st.session_state.hil_data
            st.markdown(
                f'<div style="background:var(--ams);border:1px solid var(--amb);border-radius:11px;'
                f'padding:12px 15px;margin-bottom:14px">'
                f'<div style="font-size:13px;font-weight:700;color:var(--am1)">{t("hil_review_active")}</div>'
                f'<div style="font-size:11.5px;color:var(--tx2);margin-top:3px">'
                f'{t("hil_review_desc")}</div></div>',
                unsafe_allow_html=True)

            det = d.get("detected_language_code","")
            if det:
                esc_det = html.escape(str(det))
                st.markdown(f'<div style="font-size:11.5px;color:var(--a1);margin-bottom:8px">'
                            f'{t("hil_detected_lang")}: <strong>{esc_det}</strong></div>',
                            unsafe_allow_html=True)

            with st.expander(t("hil_transcript"), expanded=True):
                edited_tr = st.text_area("Transcript", d["transcript"], height=160, key="hil_tr_edit",
                                         label_visibility="collapsed")
            with st.expander(t("hil_translation"), expanded=True):
                edited_tx = st.text_area("Translation", d["translation"], height=160, key="hil_tx_edit",
                                         label_visibility="collapsed")

            hc1, hc2 = st.columns(2)
            with hc1:
                if st.button(t("btn_approve_dub"), key="hil_approve", type="primary", use_container_width=True):
                    progress_ph = st.empty()
                    def _cb2(stage, pct, msg):
                        st.session_state.pcts[stage-1] = pct
                        st.session_state.msgs[stage-1] = msg
                        progress_ph.markdown(
                            pdash_html(st.session_state.pcts, st.session_state.msgs, stage, t("status_synthesizing")),
                            unsafe_allow_html=True)
                    try:
                        r2 = run_tts_and_mux(
                            video_path=st.session_state.hil_tmp,
                            target_language=d["language"],
                            final_text=edited_tx,
                            job_id=d["job_id"],
                            progress_cb=_cb2,
                            srt_path=d.get("srt_path",""),
                            voice_pitch=st.session_state.get("dub_pitch", 0),
                            vol_boost=st.session_state.get("dub_vol", 2.0),
                            bg_music_vol=st.session_state.get("dub_bgm", 0.0),
                            tts_engine=st.session_state.get("dub_tts_engine_sel", "auto"),
                            sarvam_speaker=st.session_state.get("dub_sarvam_spk_sel", "priya").split(" ")[0].lower(),
                        )
                        st.session_state.result    = {
                            **r2, "transcript": edited_tr,
                            "detected_language_code": d.get("detected_language_code", ""),
                            "source_language": d.get("source_language", ""),
                        }
                        st.session_state.hil_phase = "idle"
                        save_history({
                            "job_id": r2["job_id"], "language": sel_lang,
                            "source_lang": sel_src, "output_path": r2["output_path"],
                            "transcript_len": len(edited_tr),
                            "timestamp": datetime.now().isoformat(),
                        })
                        if st.session_state.hil_tmp and os.path.exists(st.session_state.hil_tmp):
                            try: os.remove(st.session_state.hil_tmp)
                            except Exception: pass
                        st.rerun()
                    except Exception as ex:
                        st.error(str(ex))
            with hc2:
                if st.button(t("btn_cancel"), key="hil_cancel", use_container_width=True):
                    if st.session_state.hil_tmp and os.path.exists(st.session_state.hil_tmp):
                        try: os.remove(st.session_state.hil_tmp)
                        except Exception: pass
                    st.session_state.hil_phase = "idle"
                    st.session_state.hil_data  = None
                    st.rerun()

        # Final result display
        elif st.session_state.result:
            r = st.session_state.result
            esc_jid = html.escape(str(r["job_id"]))
            esc_lang = html.escape(str(sel_lang))
            st.markdown(
                f'<div class="res-ok">'
                f'<div class="res-ico">✅</div>'
                f'<div><div class="res-ttl">{t("result_complete")}</div>'
                f'<div class="res-sub">{t("result_job")} {esc_jid} · {esc_lang}</div></div></div>',
                unsafe_allow_html=True)

            det = r.get("detected_language_code","")
            if det:
                esc_det = html.escape(str(det))
                esc_src = html.escape(str(r.get("source_language","")))
                st.markdown(f'<div style="font-size:11.5px;color:var(--a1);margin-bottom:10px">'
                            f'{t("result_source_detected")}: <code>{esc_det}</code>'
                            f' → <strong>{esc_src}</strong></div>',
                            unsafe_allow_html=True)

            out  = r["output_path"]
            prev = r.get("preview_path", out)
            st.video(prev if prev and os.path.exists(prev) else out)

            if os.path.exists(out):
                with open(out, "rb") as f:
                    st.download_button(t("btn_download_video"), data=f, mime="video/mp4",
                                       file_name=f"bhasha_setu_{sel_lang}_{r['job_id']}.mp4",
                                       key="dl_result", use_container_width=True)

            if r.get("srt_path") and os.path.exists(r["srt_path"]):
                with open(r["srt_path"],"r",encoding="utf-8") as sf:
                    st.download_button(t("btn_download_srt"), data=sf.read(),
                                       mime="text/plain", file_name=f"subtitles_{r['job_id']}.srt",
                                       key="dl_srt", use_container_width=True)

            with st.expander(t("hil_transcript"), expanded=False):
                tr = r.get("transcript","")
                esc_tr = html.escape(str(tr))
                st.markdown(f'<div class="tp">{esc_tr}</div>', unsafe_allow_html=True)
                if summarize_transcript:
                    try:
                        s = summarize_transcript(tr, language=sel_lang)
                        if s: st.info(s)
                    except Exception: pass

            with st.expander(t("hil_translation"), expanded=False):
                esc_trans = html.escape(str(r.get("translation","")))
                st.markdown(f'<div class="tp">{esc_trans}</div>',
                            unsafe_allow_html=True)

            # ── VIDEO INTELLIGENCE: MULTILINGUAL SUMMARY & Q&A ──
            st.markdown('<div class="idiv" style="margin:20px 0 14px"></div>', unsafe_allow_html=True)
            st.markdown(
                '<div style="display:flex;align-items:center;gap:8px;margin-bottom:12px">'
                '<span style="font-size:20px">🧠</span>'
                '<div><div style="font-family:Syne,sans-serif;font-size:14.5px;font-weight:700;color:var(--tx1)">'
                f'Video Intelligence &middot; Summary &amp; Q&amp;A</div>'
                f'<div style="font-size:11.5px;color:var(--tx3)">Ask anything or summarize in <strong>{esc_lang}</strong></div></div></div>',
                unsafe_allow_html=True,
            )

            tr_text = r.get("transcript", "")
            tx_text = r.get("translation", "")

            # Interactive Tabs for Summary and Q&A
            vi_tab_sum, vi_tab_qa = st.tabs([f"📝 AI Video Summary ({sel_lang})", f"💬 Ask About Video ({sel_lang})"])

            with vi_tab_sum:
                if st.button(f"✨ Generate Detailed AI Summary in {sel_lang}", key=f"btn_sum_{r['job_id']}", type="primary", use_container_width=True):
                    with st.spinner(f"Analyzing video content and generating structured summary in {sel_lang}…"):
                        sum_prompt = [
                            {"role": "system", "content": f"You are an expert video summarizer. Write a clear, comprehensive, and engaging summary of the video transcript in {sel_lang} language. Format with: 1) A 2-sentence executive summary, 2) 3-4 bullet points of the key highlights/events, and 3) A key takeaway. Ensure the entire response is in {sel_lang}."},
                            {"role": "user", "content": f"Video Transcript:\n{tr_text[:4000]}\n\nTranslated Dub Content ({sel_lang}):\n{tx_text[:4000]}"}
                        ]
                        summary_res = _call_groq_rest(sum_prompt, max_tokens=700, temperature=0.4)
                        st.session_state[f"video_summary_{r['job_id']}"] = summary_res

                saved_sum = st.session_state.get(f"video_summary_{r['job_id']}")
                if saved_sum:
                    st.markdown(
                        f'<div style="background:var(--sur2);border:1px solid var(--bdr2);border-radius:12px;padding:16px 18px;margin-top:10px;">'
                        f'<div style="font-size:11px;font-weight:700;color:var(--p1);text-transform:uppercase;letter-spacing:0.8px;margin-bottom:8px">📋 Comprehensive Summary ({esc_lang})</div>'
                        f'<div style="font-size:13.5px;color:var(--tx1);line-height:1.8;white-space:pre-wrap;">{html.escape(saved_sum)}</div></div>',
                        unsafe_allow_html=True,
                    )

            with vi_tab_qa:
                st.markdown(f'<div style="font-size:12px;color:var(--tx2);margin-bottom:8px">Ask any question about what is explained or spoken in this video. Answers are formulated in <strong>{esc_lang}</strong>:</div>', unsafe_allow_html=True)
                v_q = st.text_input("Ask about this video", placeholder=f"e.g. What is the central message? (Type in any language)", key=f"vq_in_{r['job_id']}", label_visibility="collapsed")
                if st.button("Ask Video AI →", key=f"btn_vq_{r['job_id']}", use_container_width=True):
                    if not v_q.strip():
                        st.warning("Please type your question first.")
                    else:
                        with st.spinner(f"Formulating answer in {sel_lang}…"):
                            qa_prompt = [
                                {"role": "system", "content": f"You are a specialized Video Intelligence Assistant. Answer the user's question accurately and informatively using ONLY the video transcript and translation provided below. Always formulate your response in {sel_lang} language."},
                                {"role": "user", "content": f"Video Transcript:\n{tr_text[:3500]}\n\nVideo Dubbing Translation ({sel_lang}):\n{tx_text[:3500]}\n\nUser Question: {v_q}"}
                            ]
                            qa_ans = _call_groq_rest(qa_prompt, max_tokens=550, temperature=0.5)
                            st.session_state[f"video_qa_{r['job_id']}"] = {"q": v_q, "a": qa_ans}

                saved_qa = st.session_state.get(f"video_qa_{r['job_id']}")
                if saved_qa:
                    st.markdown(
                        f'<div style="background:var(--sur2);border:1px solid var(--bdr2);border-radius:12px;padding:14px 18px;margin-top:10px;">'
                        f'<div style="font-size:11px;font-weight:700;color:var(--a1);margin-bottom:4px">QUESTION</div>'
                        f'<div style="font-size:13px;font-weight:600;color:var(--tx1);margin-bottom:10px">{html.escape(saved_qa["q"])}</div>'
                        f'<div style="font-size:11px;font-weight:700;color:var(--p1);margin-bottom:4px">AI ANSWER ({esc_lang})</div>'
                        f'<div style="font-size:13.5px;color:var(--tx1);line-height:1.75;white-space:pre-wrap;">{html.escape(saved_qa["a"])}</div></div>',
                        unsafe_allow_html=True,
                    )
        else:
            st.markdown(
                f'<div style="text-align:center;padding:50px 20px;animation:fadeIn .5s ease both">'
                f'<div style="font-size:44px;margin-bottom:13px">🎬</div>'
                f'<div style="font-family:Syne,sans-serif;font-size:15px;font-weight:700;color:var(--tx2)">'
                f'{t("ready_to_dub")}</div>'
                f'<div style="font-size:12.5px;color:var(--tx3);margin-top:6px;line-height:1.7">'
                f'{t("ready_to_dub_desc")}</div></div>',
                unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  BATCH DUB
# ══════════════════════════════════════════════════════════════════════════════
with t_batch:
    ba, bb = st.columns([1.1, 0.9], gap="large")
    with ba:
        st.markdown(f'<div class="eyebrow">{t("title_batch_results")}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="h-sec" style="margin-bottom:16px">{t("title_batch_dub")}</div>',
                    unsafe_allow_html=True)
        with st.container(border=True):
            st.markdown(f'<div class="label">{t("label_source_lang")}</div>', unsafe_allow_html=True)
            b_src_lang = st.selectbox("Batch source lang",
                list(SOURCE_LANGUAGES.keys()) if PIPELINE_READY else list(LANG_CODES.keys()),
                key="b_src_lang_sel", label_visibility="collapsed")
            st.markdown('<div class="idiv"></div>', unsafe_allow_html=True)
            st.markdown(f'<div class="label">{t("label_target_lang")}</div>', unsafe_allow_html=True)
            batch_langs = st.multiselect("Select target languages", list(LANGUAGES.keys()),
                default=st.session_state.batch_langs, key="batch_lang_ms",
                label_visibility="collapsed", placeholder="Select languages…")
            st.session_state.batch_langs = batch_langs
            if batch_langs:
                badges = "".join(lbadge(l, 34, 8) for l in batch_langs)
                st.markdown(f'<div style="display:flex;gap:6px;flex-wrap:wrap;margin-top:6px">{badges}</div>',
                            unsafe_allow_html=True)

        with st.container(border=True):
            st.markdown(f'<div class="label">{t("label_video_source")}</div>', unsafe_allow_html=True)
            b_inp = st.radio("Batch input", [t("label_upload_file"), t("label_paste_url")], horizontal=True,
                             key="b_inp_mode", label_visibility="collapsed")
            b_file = None; b_url = ""
            if b_inp == t("label_upload_file"):
                b_file = st.file_uploader(t("upload_mp4"), type=["mp4","mov","avi","mkv"],
                                          key="batch_uploader", label_visibility="collapsed")
            else:
                b_url = st.text_input("Batch URL", placeholder=t("video_url_placeholder"),
                                      key="b_url", label_visibility="collapsed")

        with st.container(border=True):
            st.markdown(f'<div class="label">⚡ Batch AI Engine Architecture</div>', unsafe_allow_html=True)
            bc_col1, bc_col2 = st.columns(2)
            with bc_col1:
                batch_trans_engine_disp = st.selectbox(
                    "Batch Translation Engine",
                    [
                        "Auto (Sarvam Mayura Indic / AWS)",
                        "Sarvam AI (Mayura:v1 Indic Neural)",
                        "AWS Translate (Cloud Neural)",
                        "Google / Deep Translator (Universal)",
                    ],
                    index=0,
                    key="batch_trans_sel",
                )
                b_trans_map = {
                    "Auto (Sarvam Mayura Indic / AWS)": "auto",
                    "Sarvam AI (Mayura:v1 Indic Neural)": "sarvam",
                    "AWS Translate (Cloud Neural)": "aws",
                    "Google / Deep Translator (Universal)": "google",
                }
                batch_trans_engine = b_trans_map[batch_trans_engine_disp]
            with bc_col2:
                batch_tts_engine_disp = st.selectbox(
                    "Batch Voice Engine",
                    [
                        "Auto (Sarvam Bulbul Indic / Edge-TTS)",
                        "Sarvam AI (Bulbul:v3 Indic Neural)",
                        "Edge-TTS (High-Definition Neural)",
                        "AWS Polly (Neural Voice)",
                        "Google TTS (Standard)",
                    ],
                    index=0,
                    key="batch_tts_sel",
                )
                b_tts_map = {
                    "Auto (Sarvam Bulbul Indic / Edge-TTS)": "auto",
                    "Sarvam AI (Bulbul:v3 Indic Neural)": "sarvam",
                    "Edge-TTS (High-Definition Neural)": "edge",
                    "AWS Polly (Neural Voice)": "polly",
                    "Google TTS (Standard)": "gtts",
                }
                batch_tts_engine = b_tts_map[batch_tts_engine_disp]

        b_polish = st.checkbox(t("opt_llm_polish"), key="b_polish")
        b_srt    = st.checkbox(t("opt_generate_srt"), key="b_srt")

        can_batch = bool((b_file or b_url) and batch_langs and PIPELINE_READY)
        if st.button(t("btn_start_batch"), key="batch_run", type="primary",
                     use_container_width=True, disabled=not can_batch):
            is_allowed, retry_sec = check_rate_limit(f"batch_{current_user['id']}", max_requests=3, window_seconds=300)
            if not is_allowed:
                st.error(f"🛑 Security Rate Limit: Maximum 3 batch runs per 5 minutes. Please wait {retry_sec}s.")
                st.stop()
            log_audit_event("batch_dubbing_started", current_user["id"], current_user["username"], f"Targets: {len(batch_langs)} langs")

            st.session_state.batch_results = []
            b_path = ""
            try:
                if b_file:
                    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
                    tmp.write(b_file.read()); tmp.flush(); tmp.close()
                    b_path = tmp.name

                pbar = st.progress(0, text=t("status_starting_batch"))
                for idx, lang in enumerate(batch_langs):
                    pbar.progress((idx)/len(batch_langs), text=f"{t('dubbing_to')} {lang}…")
                    try:
                        r = run_pipeline(
                            video_path=b_path,
                            video_url=b_url if not b_file else "",
                            target_language=lang,
                            polish_translation=b_polish,
                            generate_srt=b_srt,
                            source_language=b_src_lang,
                            translate_engine=batch_trans_engine,
                            tts_engine=batch_tts_engine,
                            sarvam_speaker="priya",
                        )
                        st.session_state.batch_results.append({"lang": lang, "result": r, "ok": True})
                        save_history({
                            "job_id": r["job_id"], "language": lang, "source_lang": b_src_lang,
                            "output_path": r["output_path"],
                            "transcript_len": len(r.get("transcript","")),
                            "timestamp": datetime.now().isoformat(), "batch": True,
                        })
                    except Exception as ex:
                        st.session_state.batch_results.append({"lang": lang, "error": str(ex), "ok": False})
                    pbar.progress((idx+1)/len(batch_langs), text=f"{t('done_colon')} {lang}")
                pbar.empty()
            finally:
                if b_path and os.path.exists(b_path):
                    try: os.remove(b_path)
                    except Exception: pass

    with bb:
        st.markdown(f'<div class="eyebrow">{t("title_batch_results")}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="h-sec" style="margin-bottom:16px">{t("title_downloads")}</div>', unsafe_allow_html=True)
        if st.session_state.batch_results:
            for item in st.session_state.batch_results:
                lang   = item["lang"]
                esc_lang = html.escape(str(lang))
                color  = LANG_COLOR.get(lang,"#6366F1")
                short  = html.escape(LANG_SHORT.get(lang,"??"))
                if item["ok"]:
                    out = item["result"]["output_path"]
                    rc1, rc2 = st.columns([3,1])
                    rc1.markdown(
                        f'<div style="display:flex;align-items:center;gap:9px;padding:9px 11px;'
                        f'background:var(--sur2);border:1px solid var(--bdr);'
                        f'border-radius:9px;margin-bottom:6px">'
                        f'<div style="width:32px;height:32px;border-radius:8px;background:{color};'
                        f'display:flex;align-items:center;justify-content:center;'
                        f'font-family:Syne,sans-serif;font-size:10px;font-weight:800;color:#fff">{short}</div>'
                        f'<div><div style="font-size:12.5px;font-weight:600;color:var(--tx1)">{esc_lang}</div>'
                        f'<span class="chip chip-g" style="font-size:9.5px">{t("result_done")}</span></div></div>',
                        unsafe_allow_html=True)
                    if os.path.exists(out):
                        with rc2:
                            try:
                                with open(out,"rb") as f:
                                    st.download_button("⬇️", data=f, mime="video/mp4",
                                        file_name=f"{lang}_{item['result']['job_id']}.mp4",
                                        key=f"bdl_{lang}")
                            except Exception:
                                pass
                else:
                    esc_err = html.escape(str(item.get("error", "")))[:60]
                    st.markdown(
                        f'<div style="display:flex;align-items:center;gap:9px;padding:9px 11px;'
                        f'background:var(--rs);border:1px solid var(--rb);border-radius:9px;margin-bottom:6px">'
                        f'<div style="width:32px;height:32px;border-radius:8px;background:{color};'
                        f'display:flex;align-items:center;justify-content:center;'
                        f'font-family:Syne,sans-serif;font-size:10px;font-weight:800;color:#fff">{short}</div>'
                        f'<div><div style="font-size:12.5px;font-weight:600;color:var(--r1)">{esc_lang} — {t("result_failed")}</div>'
                        f'<div style="font-size:10.5px;color:var(--tx3)">{esc_err}</div></div></div>',
                        unsafe_allow_html=True)
        else:
            st.markdown(
                f'<div style="text-align:center;padding:50px 20px">'
                f'<div style="font-size:40px;margin-bottom:12px">📦</div>'
                f'<div style="font-size:14px;color:var(--tx2)">{t("no_batch_results")}</div></div>',
                unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  HUMAN-IN-THE-LOOP (dedicated tab)
# ══════════════════════════════════════════════════════════════════════════════
with t_hil:
    st.markdown(f'<div class="eyebrow">{t("feat_hil_title")}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="h-sec" style="margin-bottom:8px">{t("title_review_dub")}</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="body" style="margin-bottom:20px">'
        f'{t("hil_body_desc")}</div>',
        unsafe_allow_html=True)

    h1, h2 = st.columns([1.1, 0.9], gap="large")
    with h1:
        with st.container(border=True):
            hc1, hc2 = st.columns(2)
            with hc1:
                h_src_lang = st.selectbox(t("label_source_lang"),
                    list(SOURCE_LANGUAGES.keys()) if PIPELINE_READY else ["Auto-detect","English"],
                    key="hil_src_sel")
            with hc2:
                h_tgt_lang = st.selectbox(t("label_target_lang"), list(LANGUAGES.keys()), key="hil_tgt_sel")

            h_file = st.file_uploader(t("upload_mp4"), type=["mp4","mov","avi","mkv"],
                                      key="hil_uploader", label_visibility="collapsed")
            h_url  = st.text_input("or paste URL", placeholder=t("video_url_placeholder"), key="hil_url_input")
            h_polish = st.checkbox(t("opt_llm_polish"), key="hil_polish")
            h_srt    = st.checkbox(t("opt_generate_srt"), key="hil_srt")

            if st.button(t("btn_transcribe_translate"), key="hil_p1_btn", type="primary",
                         use_container_width=True,
                         disabled=not (PIPELINE_READY and (h_file or h_url))):
                is_allowed, retry_sec = check_rate_limit(f"hil_{current_user['id']}", max_requests=5, window_seconds=300)
                if not is_allowed:
                    st.error(f"🛑 Security Rate Limit: Maximum 5 review requests per 5 minutes. Please wait {retry_sec}s.")
                    st.stop()
                log_audit_event("hil_phase1_started", current_user["id"], current_user["username"], f"{h_src_lang} -> {h_tgt_lang}")

                ph = st.empty()
                p1_pcts = [0.0]*5; p1_msgs = [""]*5
                def _hcb(stage, pct, msg):
                    p1_pcts[stage-1]=pct; p1_msgs[stage-1]=msg
                    ph.markdown(pdash_html(p1_pcts, p1_msgs, stage, t("status_transcribing")),
                                unsafe_allow_html=True)
                try:
                    vpath = ""
                    if h_file:
                        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
                        tmp.write(h_file.read()); tmp.flush(); tmp.close()
                        vpath = tmp.name
                    r1 = run_transcribe_and_translate(
                        video_path=vpath, video_url=h_url if not h_file else "",
                        target_language=h_tgt_lang, progress_cb=_hcb,
                        generate_srt=h_srt, polish_translation=h_polish,
                        source_language=h_src_lang,
                    )
                    st.session_state.hil4_data  = r1
                    st.session_state.hil4_phase = "review"
                    st.session_state.hil4_tmp   = r1.get("video_path", vpath)
                    ph.markdown(pdash_html([100,100,100,0,0],[""]*5,3,t("status_phase1_done"),done=False),
                                unsafe_allow_html=True)
                    st.rerun()
                except Exception as ex:
                    if vpath and os.path.exists(vpath):
                        try: os.remove(vpath)
                        except Exception: pass
                    st.error(str(ex))

    with h2:
        if st.session_state.hil4_phase == "review" and st.session_state.hil4_data:
            d  = st.session_state.hil4_data
            det = d.get("detected_language_code","")
            if det:
                esc_det = html.escape(str(det))
                st.markdown(f'<div style="font-size:12px;color:var(--a1);margin-bottom:10px">'
                            f'{t("hil_detected_lang")}: <code>{esc_det}</code></div>', unsafe_allow_html=True)
            with st.container(border=True):
                st.markdown(f'<div class="label">{t("hil_transcript")}</div>', unsafe_allow_html=True)
                e_tr = st.text_area("Transcript edit", d["transcript"], height=150,
                                    key="hil_tr2", label_visibility="collapsed")
                st.markdown('<div class="idiv"></div>', unsafe_allow_html=True)
                st.markdown(f'<div class="label">{t("hil_translation")}</div>', unsafe_allow_html=True)
                e_tx = st.text_area("Translation edit", d["translation"], height=150,
                                    key="hil_tx2", label_visibility="collapsed")

                hbv = st.slider(t("opt_voice_volume"), 1.0, 4.0, 2.0, 0.5, key="hil_vol2")
                hbp = st.slider(t("opt_voice_pitch"), -20, 20, 0, key="hil_pitch2")

                # Voice Engine & Sarvam Speaker selection for HIL Review
                st.markdown(f'<div class="label" style="margin-top:10px">Neural Voice Synthesis Engine</div>', unsafe_allow_html=True)
                hil_engine_opts = [
                    "Auto (Best Available)",
                    "Sarvam AI (Bulbul:v1 Indic Neural)",
                    "Edge-TTS (Neural Regional)",
                    "Amazon Polly (Neural Voice)",
                    "Google TTS (Fallback)",
                ]
                hil_eng_map = {
                    "Auto (Best Available)": "auto",
                    "Sarvam AI (Bulbul:v3 Indic Neural)": "sarvam",
                    "Edge-TTS (Neural Regional)": "edge",
                    "Amazon Polly (Neural Voice)": "polly",
                    "Google TTS (Fallback)": "gtts",
                }
                hil_sel_eng = st.selectbox("Voice Engine (TTS)", hil_engine_opts, index=0, key="hil_engine_sel", label_visibility="collapsed")
                hil_engine_code = hil_eng_map.get(hil_sel_eng, "auto")

                hil_sarvam_spk = "priya"
                if hil_engine_code in ["sarvam", "auto"] and is_sarvam_available():
                    spk_names = list(SARVAM_VOICES.keys()) if SARVAM_VOICES else ["priya", "aditya", "ritu", "rahul", "pooja", "rohan", "shreya", "simran", "dev", "amit"]
                    hil_sel_spk = st.selectbox("Sarvam Voice Speaker", spk_names, index=0, key="hil_sarvam_spk_sel")
                    hil_sarvam_spk = hil_sel_spk.lower().strip()

                btn_col1, btn_col2 = st.columns([2, 1])
                with btn_col1:
                    app_clicked = st.button(t("btn_approve_synthesize"), key="hil_app2", type="primary",
                                            use_container_width=True)
                with btn_col2:
                    if st.button(t("btn_cancel"), key="hil4_cancel", use_container_width=True):
                        if st.session_state.get("hil4_tmp") and os.path.exists(st.session_state["hil4_tmp"]):
                            try: os.remove(st.session_state["hil4_tmp"])
                            except Exception: pass
                        st.session_state.hil4_phase = "idle"
                        st.session_state.hil4_data = None
                        st.session_state.hil4_tmp = None
                        st.rerun()

                if app_clicked:
                    ph2 = st.empty()
                    p2_pcts = [100,100,100,0,0]; p2_msgs = [""]*5
                    def _hcb2(stage, pct, msg):
                        p2_pcts[stage-1]=pct; p2_msgs[stage-1]=msg
                        ph2.markdown(pdash_html(p2_pcts,p2_msgs,stage,t("status_synthesizing")),
                                     unsafe_allow_html=True)
                    try:
                        r2 = run_tts_and_mux(
                            video_path=st.session_state.hil4_tmp,
                            target_language=d["language"],
                            final_text=e_tx, job_id=d["job_id"],
                            progress_cb=_hcb2, srt_path=d.get("srt_path",""),
                            voice_pitch=hbp, vol_boost=hbv,
                            tts_engine=hil_engine_code,
                            sarvam_speaker=hil_sarvam_spk,
                        )
                        st.session_state.result   = {
                            **r2, "transcript": e_tr,
                            "detected_language_code": d.get("detected_language_code", ""),
                            "source_language": d.get("source_language", ""),
                        }
                        st.session_state.hil4_phase = "done"
                        save_history({
                            "job_id": r2["job_id"], "language": d["language"],
                            "source_lang": d.get("source_language",""),
                            "output_path": r2["output_path"],
                            "transcript_len": len(e_tr),
                            "timestamp": datetime.now().isoformat(),
                        })
                        track_event("hil_dubbing_completed", current_user["id"], current_user["username"], {
                            "job_id": r2["job_id"],
                            "target_language": d["language"],
                            "tts_engine": hil_engine_code,
                            "sarvam_speaker": hil_sarvam_spk,
                        })
                        if st.session_state.hil4_tmp and os.path.exists(st.session_state.hil4_tmp):
                            try: os.remove(st.session_state.hil4_tmp)
                            except Exception: pass
                        st.success(f"Done! See result in Dub Video tab. Job {r2['job_id']}")
                        out = r2["output_path"]
                        if os.path.exists(out):
                            with open(out,"rb") as f:
                                st.download_button(t("btn_download"), data=f, mime="video/mp4",
                                                   file_name=f"hil_{r2['job_id']}.mp4", key="hil_dl")
                        st.session_state.hil4_phase = "idle"
                    except Exception as ex:
                        st.error(str(ex))
        else:
            st.markdown(
                f'<div style="text-align:center;padding:50px 20px">'
                f'<div style="font-size:40px;margin-bottom:12px">🔬</div>'
                f'<div style="font-size:14px;color:var(--tx2)">{t("submit_to_begin")}</div></div>',
                unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  TEXT TRANSLATOR
# ══════════════════════════════════════════════════════════════════════════════
with t_tr:
    st.markdown(f'<div class="eyebrow">{t("tab_translate")}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="h-sec" style="margin-bottom:14px">{t("title_text_translator")}</div>',
                unsafe_allow_html=True)

    # Engine Selection & Swap Control Bar
    tb_c1, tb_c2, tb_c3 = st.columns([1.5, 0.8, 1.7])
    with tb_c1:
        tr_engine_disp = st.selectbox(
            "Translation Engine",
            [
                "Auto (Sarvam Mayura Indic / AWS / Google)",
                "Sarvam AI (Mayura:v1 Indic Neural)",
                "Google Translator (Universal)",
                "AWS Translate (Cloud Neural)",
            ],
            index=0,
            key="text_tr_engine_sel",
        )
        tr_engine_map = {
            "Auto (Sarvam Mayura Indic / AWS / Google)": "auto",
            "Sarvam AI (Mayura:v1 Indic Neural)": "sarvam",
            "Google Translator (Universal)": "google",
            "AWS Translate (Cloud Neural)": "aws",
        }
        chosen_tr_engine = tr_engine_map[tr_engine_disp]

    with tb_c2:
        st.markdown('<div style="height:27px"></div>', unsafe_allow_html=True)
        if st.button("⇄ Swap", key="tr_swap_btn", use_container_width=True, help="Swap source and target languages"):
            old_src = st.session_state.tr_src
            old_tgt = st.session_state.tr_tgt
            old_in = st.session_state.tr_input
            old_out = st.session_state.tr_output
            st.session_state.tr_src = old_tgt if old_tgt != "Auto-detect" else "English"
            st.session_state.tr_tgt = old_src if old_src != "Auto-detect" else "Hindi"
            st.session_state.tr_input = old_out
            st.session_state.tr_output = old_in
            st.rerun()

    with tb_c3:
        st.markdown(
            '<div style="font-size:11.5px;color:var(--tx2);padding-top:32px">'
            '✨ <strong style="color:var(--p1)">Sarvam Mayura &amp; AWS</strong> provide context-aware cultural nuance and idiom preservation.</div>',
            unsafe_allow_html=True,
        )

    ta1, ta2 = st.columns(2, gap="large")
    all_langs = list(LANG_CODES.keys())

    with ta1:
        src_tr = st.selectbox(t("translate_from"), all_langs,
            index=all_langs.index(st.session_state.tr_src) if st.session_state.tr_src in all_langs else 1,
            key="tr_src_sel")
        st.session_state.tr_src = src_tr
        tr_in = st.text_area(t("translate_from"), st.session_state.tr_input, height=220,
                              placeholder=t("translate_source_placeholder"), key="tr_in_area",
                              label_visibility="collapsed")
        st.session_state.tr_input = tr_in
        st.markdown(f'<div style="font-size:11px;color:var(--tx3);margin-top:-6px;margin-bottom:8px">Characters: {len(tr_in):,}</div>', unsafe_allow_html=True)

        if st.button(t("btn_translate"), key="tr_go", type="primary", use_container_width=True):
            with st.spinner("Translating text with neural accuracy…"):
                res, err = do_translate(tr_in, src_tr, st.session_state.tr_tgt, engine=chosen_tr_engine)
            if err:
                st.error(f"Translation Error: {err}")
            else:
                st.session_state.tr_output = res or ""
                st.rerun()

    with ta2:
        tgt_tr = st.selectbox(t("translate_to"), all_langs,
            index=all_langs.index(st.session_state.tr_tgt) if st.session_state.tr_tgt in all_langs else 2,
            key="tr_tgt_sel")
        st.session_state.tr_tgt = tgt_tr
        st.text_area(t("translate_to"), st.session_state.tr_output, height=220,
                     placeholder=t("translate_output_placeholder"), key="tr_out_area",
                     label_visibility="collapsed")
        st.markdown(f'<div style="font-size:11px;color:var(--tx3);margin-top:-6px;margin-bottom:8px">Translated Chars: {len(st.session_state.tr_output):,}</div>', unsafe_allow_html=True)

        if st.session_state.tr_output:
            st.download_button(t("btn_download_translation"), data=st.session_state.tr_output,
                               file_name=f"translation_{tgt_tr}.txt", mime="text/plain",
                               key="tr_dl", use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
#  AI CHAT
# ══════════════════════════════════════════════════════════════════════════════
with t_chat:

    import streamlit.components.v1 as _components

    ca, cb = st.columns([1.65, 0.35], gap="large")

    # ── Right sidebar ──────────────────────────────────────────────────────────
    with cb:
        with st.container(border=True):
            st.markdown(f'<div class="label">{t("chat_response_lang")}</div>',
                        unsafe_allow_html=True)
            chat_lang = st.selectbox(
                "Respond in", list(LANG_CODES.keys()),
                index=0, key="chat_lang_sel",
                label_visibility="collapsed")

        with st.container(border=True):
            st.markdown(f'<div class="label">{t("chat_quick_prompts")}</div>',
                        unsafe_allow_html=True)
            _qp_list = [
                ("🌐", t("chat_qp_bhasha")),
                ("🔤", t("chat_qp_translate")),
                ("🤖", t("chat_qp_ai")),
                ("🇮🇳", t("chat_qp_india")),
            ]
            for icon, qp in _qp_list:
                if st.button(f"{icon} {qp}", key=f"qp_{qp[:10]}", use_container_width=True):
                    st.session_state.chat_msgs.append({"role": "user", "content": qp})
                    with st.spinner(""):
                        reply = do_chat(qp, st.session_state.chat_msgs, chat_lang)
                    st.session_state.chat_msgs.append({"role": "assistant", "content": reply})
                    st.rerun()

        with st.container(border=True):
            st.markdown(f'<div class="label">{t("chat_model_info")}</div>',
                        unsafe_allow_html=True)
            st.markdown(
                '<div style="background:var(--gs);border:1px solid var(--gb);'
                'border-radius:8px;padding:10px 12px;">'
                '<div style="font-size:13px;font-weight:700;color:var(--g1);">'
                'Llama 3.1 8B</div>'
                '<div style="font-size:11px;color:var(--tx3);margin-top:3px;">'
                'via Groq · Free tier</div>'
                '<div style="font-size:11px;color:var(--tx3);">~200ms latency</div>'
                '</div>',
                unsafe_allow_html=True)
            msg_count = len(st.session_state.chat_msgs)
            if msg_count:
                st.markdown(
                    f'<div style="font-size:11px;color:var(--tx3);'
                    f'text-align:center;margin:8px 0;">'
                    f'<span style="color:var(--p1);font-weight:700;">{msg_count}</span>'
                    f' messages</div>',
                    unsafe_allow_html=True)
                if st.button("🗑️ Clear chat", key="clear_chat",
                             use_container_width=True):
                    st.session_state.chat_msgs = []
                    st.rerun()

    # ── Left: chat window via st.components.v1.html ────────────────────────────
    with ca:

        # Theme-aware chat colors
        _chat_text = "#EEF1FF" if DM else "#080B1A"
        _chat_meta = "#6b7a99" if DM else "#60729E"
        _chat_ai_bg = "rgba(255,255,255,.07)" if DM else "rgba(0,0,0,.04)"
        _chat_ai_bdr = "rgba(255,255,255,.11)" if DM else "rgba(0,0,0,.08)"
        _chat_chip_bg = "rgba(255,255,255,.06)" if DM else "rgba(0,0,0,.04)"
        _chat_chip_bdr = "rgba(255,255,255,.10)" if DM else "rgba(0,0,0,.07)"
        _chat_chip_color = "#9BA3C0" if DM else "#60729E"

        # Build messages HTML
        if not st.session_state.chat_msgs:
            msgs_html = f"""
            <div style="display:flex;flex-direction:column;align-items:center;
              justify-content:center;height:100%;text-align:center;padding:40px 20px;">
              <div style="width:64px;height:64px;border-radius:18px;
                background:linear-gradient(135deg,#F4722B,#E53E3E);
                display:flex;align-items:center;justify-content:center;
                font-size:28px;margin-bottom:14px;
                box-shadow:0 4px 20px rgba(244,114,43,.35);">🪷</div>
              <div style="font-family:sans-serif;font-size:17px;font-weight:700;
                color:{_chat_text};margin-bottom:8px;">{t("chat_greeting")}</div>
              <div style="font-family:sans-serif;font-size:13px;color:{_chat_meta};
                line-height:1.75;max-width:280px;">
                {t("chat_greeting_desc")}
              </div>
              <div style="display:flex;flex-wrap:wrap;gap:8px;
                justify-content:center;margin-top:18px;">
                <span style="background:rgba(244,114,43,.12);
                  border:1px solid rgba(244,114,43,.25);color:#F4722B;
                  font-size:12px;font-weight:600;padding:6px 14px;
                  border-radius:100px;font-family:sans-serif;">{t("chat_translate_text")}</span>
                <span style="background:rgba(34,211,153,.09);
                  border:1px solid rgba(34,211,153,.22);color:#34D399;
                  font-size:12px;font-weight:600;padding:6px 14px;
                  border-radius:100px;font-family:sans-serif;">{t("chat_learn_language")}</span>
                <span style="background:{_chat_chip_bg};
                  border:1px solid {_chat_chip_bdr};color:{_chat_chip_color};
                  font-size:12px;font-weight:600;padding:6px 14px;
                  border-radius:100px;font-family:sans-serif;">{t("chat_indian_culture")}</span>
              </div>
            </div>
            """
        else:
            msgs_html = ""
            for m in st.session_state.chat_msgs:
                ts  = datetime.now().strftime("%I:%M %p")
                txt = html.escape(str(m["content"]))
                if m["role"] == "user":
                    msgs_html += f"""
                    <div style="display:flex;justify-content:flex-end;
                      align-items:flex-end;gap:9px;margin-bottom:16px;">
                      <div style="display:flex;flex-direction:column;
                        align-items:flex-end;max-width:72%;">
                        <div style="background:linear-gradient(135deg,#F4722B,#E53E3E);
                          color:#fff;border-radius:16px 16px 3px 16px;
                          padding:11px 16px;font-family:sans-serif;
                          font-size:14px;line-height:1.7;
                          box-shadow:0 4px 16px rgba(244,114,43,.32);
                          word-break:break-word;">{txt}</div>
                        <div style="font-family:sans-serif;font-size:10px;
                          color:{_chat_meta};margin-top:5px;padding:0 4px;">
                          {t("chat_you")} &middot; {ts}</div>
                      </div>
                      <div style="width:32px;height:32px;border-radius:9px;
                        flex-shrink:0;background:rgba(244,114,43,.12);
                        border:1px solid rgba(244,114,43,.25);
                        display:flex;align-items:center;
                        justify-content:center;font-size:14px;">👤</div>
                    </div>
                    """
                else:
                    msgs_html += f"""
                    <div style="display:flex;justify-content:flex-start;
                      align-items:flex-end;gap:9px;margin-bottom:16px;">
                      <div style="width:32px;height:32px;border-radius:9px;
                        flex-shrink:0;
                        background:linear-gradient(135deg,#F4722B,#E53E3E);
                        box-shadow:0 2px 10px rgba(244,114,43,.3);
                        display:flex;align-items:center;
                        justify-content:center;font-size:14px;">🪷</div>
                      <div style="display:flex;flex-direction:column;
                        align-items:flex-start;max-width:72%;">
                        <div style="background:{_chat_ai_bg};
                          border:1px solid {_chat_ai_bdr};
                          color:{_chat_text};border-radius:16px 16px 16px 3px;
                          padding:11px 16px;font-family:sans-serif;
                          font-size:14px;line-height:1.7;
                          word-break:break-word;">{txt}</div>
                        <div style="font-family:sans-serif;font-size:10px;
                          color:{_chat_meta};margin-top:5px;padding:0 4px;">
                          {t("chat_ai_name")} &middot; {ts}</div>
                      </div>
                    </div>
                    """

        msg_count = len(st.session_state.chat_msgs)

        _cw_border = "rgba(255,255,255,.09)" if DM else "rgba(0,0,0,.09)"
        _cw_bg     = "rgba(255,255,255,.025)" if DM else "rgba(0,0,0,.015)"
        _tb_bg     = "rgba(255,255,255,.05)" if DM else "rgba(0,0,0,.03)"
        _tb_border = "rgba(255,255,255,.08)" if DM else "rgba(0,0,0,.07)"
        _badge_bg  = "rgba(255,255,255,.04)" if DM else "rgba(0,0,0,.04)"
        _badge_bdr = "rgba(255,255,255,.08)" if DM else "rgba(0,0,0,.07)"
        _ft_bg     = "rgba(255,255,255,.04)" if DM else "rgba(0,0,0,.025)"
        _ft_bdr    = "rgba(255,255,255,.07)" if DM else "rgba(0,0,0,.06)"

        # Full chat HTML rendered as a component (bypasses Streamlit markdown sanitizer)
        chat_html = f"""
        <!DOCTYPE html>
        <html>
        <head>
        <meta charset="utf-8">
        <style>
          * {{ box-sizing:border-box; margin:0; padding:0; }}
          body {{
            background: transparent;
            font-family: 'Segoe UI', sans-serif;
            overflow: hidden;
          }}
          ::-webkit-scrollbar {{ width: 4px; }}
          ::-webkit-scrollbar-track {{ background: transparent; }}
          ::-webkit-scrollbar-thumb {{
            background: linear-gradient(#F4722B, #E53E3E);
            border-radius: 2px;
          }}
          .chat-wrap {{
            display: flex;
            flex-direction: column;
            height: 540px;
            border: 1px solid {_cw_border};
            border-radius: 14px;
            overflow: hidden;
            background: {_cw_bg};
          }}
          .chat-topbar {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 12px 18px;
            background: {_tb_bg};
            border-bottom: 1px solid {_tb_border};
            flex-shrink: 0;
          }}
          .chat-topbar-left {{ display:flex; align-items:center; gap:11px; }}
          .chat-ava {{
            width: 38px; height: 38px; border-radius: 11px;
            background: linear-gradient(135deg,#F4722B,#E53E3E);
            display: flex; align-items: center; justify-content: center;
            font-size: 17px;
            box-shadow: 0 3px 12px rgba(244,114,43,.35);
            flex-shrink: 0;
          }}
          .chat-name {{
            font-size: 14px; font-weight: 700; color: {_chat_text};
          }}
          .chat-status {{
            font-size: 11px; color: #34D399;
            display: flex; align-items: center; gap: 4px; margin-top: 2px;
          }}
          .status-dot {{
            width: 6px; height: 6px; border-radius: 50%;
            background: #34D399; display: inline-block;
          }}
          .chat-badge {{
            font-size: 10.5px; color: {_chat_meta};
            background: {_badge_bg};
            border: 1px solid {_badge_bdr};
            padding: 3px 10px; border-radius: 100px;
          }}
          .chat-messages {{
            flex: 1;
            overflow-y: auto;
            padding: 18px 18px 8px 18px;
            scroll-behavior: smooth;
          }}
          .chat-footer {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 9px 18px;
            background: {_ft_bg};
            border-top: 1px solid {_ft_bdr};
            flex-shrink: 0;
          }}
          .footer-txt {{ font-size: 11px; color: {_chat_meta}; }}
          .footer-txt span {{ color: #F4722B; font-weight: 700; }}
        </style>
        </head>
        <body>
        <div class="chat-wrap">
          <div class="chat-topbar">
            <div class="chat-topbar-left">
              <div class="chat-ava">🪷</div>
              <div>
                <div class="chat-name">{t("chat_ai_name")}</div>
                <div class="chat-status">
                  <span class="status-dot"></span> {t("chat_online")}
                </div>
              </div>
            </div>
            <div class="chat-badge">Groq &middot; Llama 3.1</div>
          </div>
          <div class="chat-messages" id="msgs">
            {msgs_html}
          </div>
          <div class="chat-footer">
            <div class="footer-txt">
              <span>{msg_count}</span> {t("chat_messages")} &middot;
              {t("chat_response_lang")}: <span>{chat_lang}</span>
            </div>
            <div class="footer-txt">Groq API &middot; Free</div>
          </div>
        </div>
        <script>
          var el = document.getElementById('msgs');
          if (el) el.scrollTop = el.scrollHeight;
        </script>
        </body>
        </html>
        """

        _components.html(chat_html, height=548, scrolling=False)

        # Chat input below the component
        if prompt := st.chat_input(t("chat_input_placeholder")):
            st.session_state.chat_msgs.append({"role": "user", "content": prompt})
            with st.spinner(""):
                reply = do_chat(prompt, st.session_state.chat_msgs, chat_lang)
            st.session_state.chat_msgs.append({"role": "assistant", "content": reply})
            st.rerun()

with t_hist:
    history = load_history()
    if not history:
        st.markdown(
            f'<div style="text-align:center;padding:60px 20px">'
            f'<div style="font-size:44px;margin-bottom:13px">📭</div>'
            f'<div style="font-size:16px;font-weight:600;color:var(--tx2)">{t("history_no_jobs")}</div>'
            f'<div style="font-size:13px;color:var(--tx3);margin-top:5px">'
            f'{t("history_desc")}</div></div>',
            unsafe_allow_html=True)
    else:
        lu   = {}
        for h in history: l=h.get("language",""); lu[l]=lu.get(l,0)+1
        tc_  = sum(h.get("transcript_len",0) for h in history)
        bc__ = sum(1 for h in history if h.get("batch"))
        chars_disp = f"{tc_//1000}K" if tc_ >= 1000 else str(tc_)
        mc   = st.columns(4, gap="medium")
        for i,(col,(val,lbl)) in enumerate(zip(mc,[
                (len(history), t("history_total_jobs")),(len(lu), t("history_languages")),
                (chars_disp, t("history_chars")),(bc__, t("history_batch_jobs"))])):
            col.markdown(
                f'<div class="met" style="animation:scaleIn .3s ease both;animation-delay:{i*.06:.2f}s">'
                f'<div class="met-n">{val}</div><div class="met-l">{lbl}</div></div>',
                unsafe_allow_html=True)

        st.markdown(f'<div class="idiv"></div><div class="h-card">{t("history_job_log")}</div>', unsafe_allow_html=True)
        for i, h in enumerate(history):
            lang   = h.get("language","—")
            src_l  = h.get("source_lang","—")
            ts_    = h.get("timestamp","")[:16].replace("T"," ")
            jid    = h.get("job_id","—")
            esc_lang  = html.escape(str(lang))
            esc_src_l = html.escape(str(src_l))
            esc_ts_   = html.escape(str(ts_))
            esc_jid   = html.escape(str(jid))
            color  = LANG_COLOR.get(lang,"#6366F1")
            short  = html.escape(LANG_SHORT.get(lang,"?"))
            btag   = (f' <span class="chip chip-a" style="font-size:9px;padding:2px 6px">{t("batch_tag")}</span>'
                      if h.get("batch") else "")
            cc_, cd_ = st.columns([5,1])
            cc_.markdown(
                f'<div class="hcard" style="display:flex;align-items:center;justify-content:space-between">'
                f'<div style="display:flex;align-items:center;gap:10px">'
                f'<div style="width:36px;height:36px;border-radius:9px;background:{color};flex-shrink:0;'
                f'display:flex;align-items:center;justify-content:center;'
                f'font-family:Syne,sans-serif;font-size:11px;font-weight:800;color:#fff">{short}</div>'
                f'<div><div style="font-family:Syne,sans-serif;font-size:13px;font-weight:700;'
                f'color:var(--tx1)">{esc_src_l} → {esc_lang}{btag}</div>'
                f'<div style="font-size:11px;color:var(--tx3)">Job <code>{esc_jid}</code> · {esc_ts_}</div>'
                f'</div></div>'
                f'<div style="font-size:10.5px;color:var(--tx3)">{h.get("transcript_len",0)} {t("history_chars_suffix")}</div>'
                f'</div>',
                unsafe_allow_html=True)
            out = h.get("output_path","")
            if out and os.path.exists(out):
                with cd_:
                    try:
                        with open(out,"rb") as f:
                            st.download_button("⬇️", data=f, mime="video/mp4",
                                file_name=f"{lang}_{jid}.mp4", key=f"hdl_{i}_{jid}")
                    except Exception:
                        pass

        if st.button(t("btn_clear_history"), key="clr_hist"):
            user = get_current_user()
            if user:
                clear_user_history(user["id"], is_admin=(user.get("role") == "admin"))
            st.rerun()

    # ── Production Telemetry & Analytics Dashboard (Checklist: Analytics setup) ──
    st.markdown('<div class="idiv" style="margin:28px 0 18px"></div>', unsafe_allow_html=True)
    st.markdown('##### 📊 Production Telemetry & Performance Analytics')
    analytics_data = get_analytics_summary()
    ac1, ac2, ac3 = st.columns(3)
    with ac1:
        st.metric("Total Dubbing Jobs", analytics_data.get("total_dubs", 0))
    with ac2:
        st.metric("Registered Users", analytics_data.get("total_users", 0))
    with ac3:
        st.metric("Logged Telemetry Events", analytics_data.get("total_events", 0))

    if analytics_data.get("top_languages"):
        st.markdown('<div style="font-size:12px;font-weight:700;color:var(--tx2);margin:12px 0 6px;">Top Dubbed Indic Languages</div>', unsafe_allow_html=True)
        for item in analytics_data["top_languages"]:
            l_name = html.escape(str(item["language"]))
            l_cnt = item["count"]
            st.markdown(
                f'<div style="display:flex;align-items:center;justify-content:space-between;background:var(--sur2);padding:7px 14px;border-radius:8px;margin-bottom:5px;font-size:12.5px;">'
                f'<span style="font-weight:600;color:var(--tx1);">{l_name}</span><span class="chip chip-p">{l_cnt} dubs</span></div>',
                unsafe_allow_html=True,
            )


# ══════════════════════════════════════════════════════════════════════════════
#  NEO4J KNOWLEDGE GRAPH & MEDIA PROVENANCE STUDIO
# ══════════════════════════════════════════════════════════════════════════════
with t_graph:
    from pipeline.graph_db import (
        get_graph_statistics,
        get_visual_graph_elements,
        query_graph_glossary,
        Neo4jGraphService,
        INDIC_KNOWLEDGE_GLOSSARY,
        init_graph_schema,
        _load_local_graph,
        _get_neo4j_config,
    )

    init_graph_schema()
    g_stats = get_graph_statistics()

    # 1. Header Banner & Status
    is_live = g_stats.get("is_live_neo4j", False)
    status_color = "#10B981" if is_live else "#F59E0B"
    status_label = "Active Connected (Bolt / AuraDB)" if is_live else "Local Graph Engine (Sandbox Mode)"
    status_desc = "Cluster operational" if is_live else "Seamless local cache active. Configure remote AuraDB credentials below."

    st.markdown(
        f"""
        <div style="background:var(--glass);border:1px solid var(--bdr);border-radius:24px;padding:24px 28px;margin-bottom:20px;box-shadow:var(--sh);">
            <div style="display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:12px;margin-bottom:12px;">
                <div style="display:flex;align-items:center;gap:12px;">
                    <div style="font-size:32px;filter:drop-shadow(0 2px 8px rgba(244,114,43,0.3));">🕸️</div>
                    <div>
                        <div style="font-family:'Syne',sans-serif;font-size:20px;font-weight:800;color:var(--tx1);">
                            Neo4j Multilingual Knowledge Graph &amp; Provenance Studio
                        </div>
                        <div style="font-size:12px;color:var(--tx2);">
                            Cross-lingual semantic relationships, cultural vernacular preservation, and media pipeline DAG tracing.
                        </div>
                    </div>
                </div>
                <div style="display:inline-flex;align-items:center;gap:8px;padding:6px 16px;border-radius:9999px;background:var(--sur2);border:1px solid var(--bdr2);font-size:11.5px;font-weight:700;color:var(--tx1);">
                    <span style="width:8px;height:8px;border-radius:50%;background:{status_color};display:inline-block;box-shadow:0 0 8px {status_color};"></span>
                    <span>{status_label}</span>
                </div>
            </div>
            <div style="font-size:11.5px;color:var(--tx3);line-height:1.6;">
                {status_desc}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 2. Metrics Row
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Total Graph Nodes", g_stats.get("total_nodes", 0))
    with m2:
        st.metric("Graph Relationships", g_stats.get("total_relationships", 0))
    with m3:
        st.metric("Indic Cultural Concepts", g_stats.get("concept_count", 0))
    with m4:
        st.metric("Dubbing Video Nodes", g_stats.get("video_count", 0))

    st.markdown('<div style="height:12px;"></div>', unsafe_allow_html=True)

    # 3. Interactive Force-Directed Canvas / Visualization
    with st.container(border=True):
        st.markdown(
            """
            <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:14px;">
                <div style="font-family:'Syne',sans-serif;font-size:15px;font-weight:700;color:var(--tx1);">
                    🌐 Interactive Graph Explorer (Force-Directed Graph)
                </div>
                <div style="font-size:11.5px;color:var(--tx3);">
                    Drag nodes to inspect relationships &bull; Colors represent node classes
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        visual_data = get_visual_graph_elements(limit=50)
        nodes_json = json.dumps(visual_data["nodes"])
        edges_json = json.dumps(visual_data["edges"])

        # Embed high-performance standalone interactive HTML5 canvas simulation
        graph_canvas_html = f"""
        <!DOCTYPE html>
        <html>
        <head>
          <meta charset="utf-8">
          <style>
            body {{ margin: 0; padding: 0; background: transparent; overflow: hidden; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }}
            #graph-container {{ width: 100%; height: 420px; position: relative; background: rgba(14, 20, 35, 0.45); border-radius: 18px; border: 1px solid rgba(255,255,255,0.08); overflow: hidden; }}
            canvas {{ width: 100%; height: 100%; display: block; }}
            .legend {{ position: absolute; bottom: 12px; left: 16px; display: flex; gap: 10px; flex-wrap: wrap; pointer-events: none; }}
            .legend-item {{ display: flex; align-items: center; gap: 5px; font-size: 11px; color: #CBD5E1; background: rgba(15,23,42,0.8); padding: 3px 8px; border-radius: 6px; border: 1px solid rgba(255,255,255,0.1); }}
            .dot {{ width: 8px; height: 8px; border-radius: 50%; }}
            #tooltip {{ position: absolute; display: none; background: #0F172A; color: #FFF; padding: 8px 12px; border-radius: 8px; font-size: 11.5px; border: 1px solid rgba(255,255,255,0.2); box-shadow: 0 4px 16px rgba(0,0,0,0.5); pointer-events: none; max-width: 260px; z-index: 10; }}
          </style>
        </head>
        <body>
          <div id="graph-container">
            <canvas id="cv"></canvas>
            <div id="tooltip"></div>
            <div class="legend">
              <div class="legend-item"><span class="dot" style="background:#FF6B35"></span>Video</div>
              <div class="legend-item"><span class="dot" style="background:#3B82F6"></span>Transcript</div>
              <div class="legend-item"><span class="dot" style="background:#10B981"></span>Translation</div>
              <div class="legend-item"><span class="dot" style="background:#8B5CF6"></span>VoicePersona</div>
              <div class="legend-item"><span class="dot" style="background:#EC4899"></span>Concept</div>
              <div class="legend-item"><span class="dot" style="background:#06B6D4"></span>Domain</div>
            </div>
          </div>
          <script>
            const rawNodes = {nodes_json};
            const rawEdges = {edges_json};

            const cv = document.getElementById('cv');
            const ctx = cv.getContext('2d');
            const tooltip = document.getElementById('tooltip');
            let w, h;

            function resize() {{
              w = cv.width = cv.parentElement.clientWidth;
              h = cv.height = cv.parentElement.clientHeight;
            }}
            window.addEventListener('resize', resize);
            resize();

            // Setup Nodes with 2D positions
            const nodeMap = {{}};
            const nodes = rawNodes.map((n, i) => {{
              const angle = (i / rawNodes.length) * 2 * Math.PI;
              const radius = Math.min(w, h) * 0.32 + (Math.random() - 0.5) * 60;
              const node = {{
                ...n,
                x: w / 2 + Math.cos(angle) * radius,
                y: h / 2 + Math.sin(angle) * radius,
                vx: 0,
                vy: 0,
                radius: n.type === 'Concept' ? 14 : (n.type === 'Domain' ? 16 : 12),
              }};
              nodeMap[n.id] = node;
              return node;
            }});

            const edges = rawEdges.map(e => ({{
              source: nodeMap[e.source] || nodeMap[Object.keys(nodeMap)[0]],
              target: nodeMap[e.target] || nodeMap[Object.keys(nodeMap)[0]],
              label: e.label,
            }})).filter(e => e.source && e.target);

            // Simulation loop
            function step() {{
              // Repulsion
              for (let i = 0; i < nodes.length; i++) {{
                for (let j = i + 1; j < nodes.length; j++) {{
                  const dx = nodes[j].x - nodes[i].x;
                  const dy = nodes[j].y - nodes[i].y;
                  const dist = Math.sqrt(dx * dx + dy * dy) || 1;
                  if (dist < 180) {{
                    const force = (180 - dist) / dist * 0.45;
                    nodes[i].vx -= dx * force * 0.05;
                    nodes[i].vy -= dy * force * 0.05;
                    nodes[j].vx += dx * force * 0.05;
                    nodes[j].vy += dy * force * 0.05;
                  }}
                }}
              }}

              // Spring attraction
              edges.forEach(e => {{
                const dx = e.target.x - e.source.x;
                const dy = e.target.y - e.source.y;
                const dist = Math.sqrt(dx * dx + dy * dy) || 1;
                const force = (dist - 110) * 0.035;
                e.source.vx += dx / dist * force;
                e.source.vy += dy / dist * force;
                e.target.vx -= dx / dist * force;
                e.target.vy -= dy / dist * force;
              }});

              // Center gravity & update
              nodes.forEach(n => {{
                if (n !== dragNode) {{
                  n.vx += (w / 2 - n.x) * 0.005;
                  n.vy += (h / 2 - n.y) * 0.005;
                  n.vx *= 0.82;
                  n.vy *= 0.82;
                  n.x += n.vx;
                  n.y += n.vy;
                  n.x = Math.max(25, Math.min(w - 25, n.x));
                  n.y = Math.max(25, Math.min(h - 25, n.y));
                }}
              }});

              draw();
              requestAnimationFrame(step);
            }}

            function draw() {{
              ctx.clearRect(0, 0, w, h);

              // Draw edges
              edges.forEach(e => {{
                ctx.beginPath();
                ctx.moveTo(e.source.x, e.source.y);
                ctx.lineTo(e.target.x, e.target.y);
                ctx.strokeStyle = 'rgba(255, 255, 255, 0.15)';
                ctx.lineWidth = 1.2;
                ctx.stroke();

                // Draw edge arrow
                const midX = (e.source.x + e.target.x) / 2;
                const midY = (e.source.y + e.target.y) / 2;
                ctx.fillStyle = 'rgba(148, 163, 184, 0.6)';
                ctx.font = '9px sans-serif';
                ctx.fillText(e.label, midX + 4, midY - 3);
              }});

              // Draw nodes
              nodes.forEach(n => {{
                ctx.beginPath();
                ctx.arc(n.x, n.y, n.radius, 0, 2 * Math.PI);
                ctx.fillStyle = n.color;
                ctx.fill();
                ctx.lineWidth = 2.5;
                ctx.strokeStyle = n.borderColor;
                ctx.stroke();

                // Labels
                ctx.fillStyle = '#FFFFFF';
                ctx.font = 'bold 10px sans-serif';
                ctx.textAlign = 'center';
                ctx.fillText(n.label, n.x, n.y + n.radius + 13);
              }});
            }}

            // Drag handling
            let dragNode = null;
            cv.addEventListener('mousedown', e => {{
              const rect = cv.getBoundingClientRect();
              const mx = e.clientX - rect.left;
              const my = e.clientY - rect.top;
              dragNode = nodes.find(n => Math.hypot(n.x - mx, n.y - my) <= n.radius + 5);
            }});
            window.addEventListener('mousemove', e => {{
              const rect = cv.getBoundingClientRect();
              const mx = e.clientX - rect.left;
              const my = e.clientY - rect.top;
              if (dragNode) {{
                dragNode.x = mx;
                dragNode.y = my;
              }}
              // Tooltip on hover
              const hover = nodes.find(n => Math.hypot(n.x - mx, n.y - my) <= n.radius + 5);
              if (hover) {{
                tooltip.style.display = 'block';
                tooltip.style.left = (mx + 15) + 'px';
                tooltip.style.top = (my + 10) + 'px';
                tooltip.innerHTML = `<strong>${{hover.label}}</strong><br><span style="color:#94A3B8;">Type: ${{hover.type}}</span><br>${{hover.details}}`;
              }} else if (!dragNode) {{
                tooltip.style.display = 'none';
              }}
            }});
            window.addEventListener('mouseup', () => {{ dragNode = null; }});

            step();
          </script>
        </body>
        </html>
        """
        import streamlit.components.v1 as _components
        _components.html(graph_canvas_html, height=440)

    st.markdown('<div style="height:10px;"></div>', unsafe_allow_html=True)

    # 4. Tabbed Sub-Panels (Glossary Explorer, Provenance DAG, Connection Settings)
    gt1, gt2, gt3 = st.tabs([
        "💡 Cultural Vernacular & GraphRAG Glossary",
        "🔗 Media Provenance & Lineage DAG",
        "⚙️ Neo4j Cluster Configuration",
    ])

    with gt1:
        st.markdown(
            """
            <div style="font-size:13px;color:var(--tx2);margin-bottom:12px;">
                Verified Indian administrative schemes, agricultural concepts, and cultural idioms mapped across regional Indic languages.
                These constraints are queried dynamically during neural translation to prevent awkward literal phrasing.
            </div>
            """,
            unsafe_allow_html=True,
        )
        search_kw = st.text_input("🔍 Search Concepts or Regional Idioms", placeholder="e.g. PM-KISAN, UPI, apple of eye, kisan, health", key="graph_search_kw")

        items_to_show = INDIC_KNOWLEDGE_GLOSSARY
        if search_kw:
            skw = search_kw.lower().strip()
            items_to_show = [
                it for it in INDIC_KNOWLEDGE_GLOSSARY
                if skw in it["canonical_name"].lower()
                or skw in it["domain"].lower()
                or skw in it["description"].lower()
                or any(skw in tv.lower() for tv in it["terms"].values())
            ]

        for item in items_to_show:
            with st.expander(f"📌 {item['canonical_name']} · {item['domain']}", expanded=bool(search_kw)):
                st.markdown(f"**Description:** {item['description']}")
                st.markdown("**Cross-Lingual Verified Equivalents:**")
                tcols = st.columns(4)
                terms_list = list(item["terms"].items())
                for i, (l_code, l_val) in enumerate(terms_list):
                    with tcols[i % 4]:
                        l_name = {"en": "English", "hi": "Hindi", "ta": "Tamil", "te": "Telugu", "kn": "Kannada", "mr": "Marathi", "bn": "Bengali", "pa": "Punjabi"}.get(l_code, l_code.upper())
                        st.markdown(
                            f'<div style="background:var(--sur2);border:1px solid var(--bdr);border-radius:12px;padding:8px 12px;margin-bottom:8px;">'
                            f'<div style="font-size:10px;font-weight:700;color:var(--p1);">{l_name}</div>'
                            f'<div style="font-size:12.5px;font-weight:600;color:var(--tx1);">{l_val}</div>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )

    with gt2:
        st.markdown(
            """
            <div style="font-size:13px;color:var(--tx2);margin-bottom:14px;">
                Cryptographic DAG tracking the complete lineage of every video processed by Bhasha Setu:
                <code>Raw Video &rarr; Speech Transcript &rarr; Neural Translation &rarr; Synthesized Persona &rarr; Dubbed Master MP4</code>.
            </div>
            """,
            unsafe_allow_html=True,
        )
        local_g = _load_local_graph()
        video_nodes = [n for n in local_g.get("nodes", {}).values() if n.get("label") == "Video"]
        if not video_nodes:
            st.info("ℹ️ No dubbing jobs recorded in the provenance graph yet. Run a dubbing job in the 'Dub Video' tab to populate this DAG.")
        else:
            for vn in reversed(video_nodes[-10:]):
                v_id = vn.get("video_id", "unknown")
                v_title = vn.get("title", f"Video {v_id}")
                v_status = vn.get("status", "COMPLETED")
                v_out = vn.get("output_file", "")
                st.markdown(
                    f'<div style="background:var(--glass);border:1px solid var(--bdr);border-radius:16px;padding:14px 18px;margin-bottom:10px;">'
                    f'<div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:6px;">'
                    f'<span style="font-weight:700;font-size:13.5px;color:var(--tx1);">🎥 {v_title}</span>'
                    f'<span class="chip chip-p">{v_status}</span>'
                    f'</div>'
                    f'<div style="font-size:11.5px;color:var(--tx2);">'
                    f'Lineage: <code>{vn.get("source_lang", "Auto")} &rarr; {vn.get("target_lang", "Indic")}</code> &bull; Job ID: <code>{v_id}</code>'
                    f'{" &bull; Output: <code>" + v_out + "</code>" if v_out else ""}'
                    f'</div></div>',
                    unsafe_allow_html=True,
                )

    with gt3:
        st.markdown(
            """
            <div style="font-size:13px;color:var(--tx2);margin-bottom:14px;">
                Connect Bhasha Setu to a <strong>Neo4j AuraDB Cloud Instance</strong> (Free Tier) or a local <strong>Docker Neo4j container</strong>.
            </div>
            """,
            unsafe_allow_html=True,
        )
        cur_uri, cur_user, cur_pwd, cur_db, is_cfg = _get_neo4j_config()
        n_col1, n_col2 = st.columns(2)
        with n_col1:
            neo_uri_input = st.text_input("Neo4j Bolt URI", value=cur_uri or "neo4j+s://your-instance.databases.neo4j.io", key="cfg_neo_uri")
            neo_user_input = st.text_input("Neo4j Username", value=cur_user or "neo4j", key="cfg_neo_user")
        with n_col2:
            neo_pwd_input = st.text_input("Neo4j Password", value=cur_pwd or "", type="password", key="cfg_neo_pwd")
            neo_db_input = st.text_input("Database Name", value=cur_db or "neo4j", key="cfg_neo_db")

        if st.button("🔌 Test & Verify Neo4j Connection", type="primary", use_container_width=True, key="btn_test_neo4j"):
            os.environ["NEO4J_URI"] = neo_uri_input.strip()
            os.environ["NEO4J_USERNAME"] = neo_user_input.strip()
            os.environ["NEO4J_PASSWORD"] = neo_pwd_input.strip()
            os.environ["NEO4J_DATABASE"] = neo_db_input.strip()
            Neo4jGraphService._driver = None
            ok, msg, meta = Neo4jGraphService.test_connection()
            if ok:
                st.success(f"✅ {msg}")
                st.json(meta)
            else:
                st.warning(f"⚠️ {msg}")
                st.info("Tip: Create a free Neo4j AuraDB instance at https://neo4j.com/cloud/platform/aura-graph-database/ and paste the URI and password here.")


# ══════════════════════════════════════════════════════════════════════════════
#  ROADMAP
# ══════════════════════════════════════════════════════════════════════════════
with t_road:
    ra, rb = st.columns([1.4, 0.6], gap="large")
    with ra:
        st.markdown(f'<div class="eyebrow">{t("roadmap_whats_coming")}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="h-sec" style="margin-bottom:16px">{t("title_feature_roadmap")}</div>',
                    unsafe_allow_html=True)
        features = [
            ("01","Any-to-Any Language Dubbing","Auto-detect source language via AWS Transcribe IdentifyLanguage=True + NLLB-200 translation across 200 pairs.","live"),
            ("02","Batch Multi-Language","One upload → all 12 dubbed videos in a single automated run.","live"),
            ("03","Human-in-the-Loop Review","Review and edit transcript + translation before speech synthesis.","live"),
            ("04","LLM Translation Polish","AWS Bedrock Claude refines translation for conversational quality.","live"),
            ("05","SRT Subtitle Generation","Auto-generate word-level timed subtitles alongside dubbed video.","live"),
            ("06","Quick Preview Mode","Dub first N seconds to validate before full processing.","live"),
            ("07","Sentence-Level Timestamp Dubbing","Per-sentence TTS with FFmpeg silence gaps for precise lip-sync.","soon"),
            ("08","Speaker Diarization","Detect multiple speakers and assign distinct voices per speaker.","soon"),
            ("09","Emotion-Aware SSML","Detect punctuation and inject SSML prosody tags for expressive TTS.","soon"),
            ("10","Voice Cloning","Clone original speaker's voice across languages using XTTS-v2.","soon"),
            ("11","PDF / DOCX Translation","Full support for document translation with layout preservation.","plan"),
            ("12","Mobile App","Android/iOS on-device translation and dubbing.","plan"),
        ]
        for i,(num,title,desc,status) in enumerate(features):
            pc = {"live":"p-live","soon":"p-soon","plan":"p-plan"}[status]
            pt = {"live": t("roadmap_live"), "soon": t("roadmap_soon"), "plan": t("roadmap_planned")}[status]
            st.markdown(
                f'<div class="rm" style="animation:slideIn .35s ease both;animation-delay:{i*.04:.2f}s">'
                f'<div class="rm-tag">{t("roadmap_feature")} {num}</div>'
                f'<div class="rm-ttl">{title}</div><div class="rm-desc">{desc}</div>'
                f'<span class="pill {pc}">{pt}</span></div>',
                unsafe_allow_html=True)

    with rb:
        st.markdown(
            f'<div style="background:var(--glass);border:1px solid var(--bdr);'
            f'border-radius:var(--rad);padding:24px 20px;text-align:center;margin-bottom:14px">'
            f'<div style="font-size:46px;margin-bottom:10px;animation:float 4s ease-in-out infinite">🧑‍🎓</div>'
            f'<div style="font-family:Syne,sans-serif;font-size:16px;font-weight:800;'
            f'background:var(--pg2);background-size:200%;-webkit-background-clip:text;'
            f'-webkit-text-fill-color:transparent;background-clip:text;margin-bottom:4px">Abhimanyu</div>'
            f'<div style="font-size:13px;color:var(--tx2);margin-bottom:3px">{t("about_author")}</div>'
            f'<div class="idiv"></div>'
            f'<div style="font-size:13px;color:var(--tx2);line-height:1.75">'
            f'{t("about_mission")}</div></div>',
            unsafe_allow_html=True)

        st.markdown(
            f'<div style="background:var(--glass);border:1px solid var(--bdr);'
            f'border-radius:var(--rad);padding:18px 16px;margin-bottom:12px">'
            f'<div class="label" style="margin-bottom:12px">{t("roadmap_status")}</div>'
            f'<div class="bar-row"><div class="bar-lbl">{t("roadmap_live")}</div>'
            f'<div class="bar-track"><div class="bar-fill" style="width:50%"></div></div>'
            f'<div class="bar-val">6</div></div>'
            f'<div class="bar-row"><div class="bar-lbl">{t("roadmap_soon")}</div>'
            f'<div class="bar-track"><div class="bar-fill" style="width:33%;background:var(--ag)"></div></div>'
            f'<div class="bar-val">4</div></div>'
            f'<div class="bar-row"><div class="bar-lbl">{t("roadmap_planned")}</div>'
            f'<div class="bar-track"><div class="bar-fill" style="width:17%;background:linear-gradient(90deg,var(--tx4),var(--tx3))"></div></div>'
            f'<div class="bar-val">2</div></div></div>',
            unsafe_allow_html=True)

        with st.container(border=True):
            st.markdown(f'<div class="label">{t("roadmap_tech_stack")}</div>', unsafe_allow_html=True)
            techs = "".join(
                f'<span class="chip" style="margin:2px 1px">{icon} {name}</span>'
                for icon, name in [
                    ("🌟","Sarvam AI"),("⚡","Supabase"),("☁️","S3"),("🎙️","Transcribe"),
                    ("🌐","Translate"),("🤖","Polly"),("🔷","edge-tts"),("🔵","gTTS"),
                    ("🎬","FFmpeg"),("🐍","Python"),("⚡","Streamlit"),("🦙","Groq"),
                ])
            st.markdown(f'<div style="line-height:2.2">{techs}</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  ENTERPRISE FOOTER & COMPLIANCE
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="sdiv"></div>', unsafe_allow_html=True)
st.markdown(
    """
    <div style="text-align:center;padding:10px 10px 24px;color:var(--tx3);font-size:12px;line-height:1.8">
      <div style="display:flex;align-items:center;justify-content:center;gap:8px;margin-bottom:6px">
        <span style="font-size:16px">🪷</span>
        <span style="font-family:'Syne',sans-serif;font-weight:700;color:var(--tx1)">Bhasha Setu &middot; AI Bharat</span>
        <span>&middot; Version 5.2 Enterprise Edition</span>
      </div>
      <div>
        <span>Protected by Google Identity Services &amp; PBKDF2 Encryption</span> &middot;
        <span>Sarvam AI Bulbul:v3 &amp; Mayura:v1 Indic Neural Engine</span> &middot;
        <span>Thread-Safe SQLite WAL / Supabase</span>
      </div>
      <div style="margin-top:6px">
        <span>&copy; 2026 J.C. Bose UST YMCA, Faridabad &middot; All Rights Reserved</span>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.expander("📜 Privacy Policy & Terms of Service (Official Documents)", expanded=False):
    doc_tab1, doc_tab2 = st.tabs(["🛡️ Privacy Policy", "⚖️ Terms of Service"])
    with doc_tab1:
        try:
            with open("PRIVACY_POLICY.md", "r", encoding="utf-8") as f:
                st.markdown(f.read())
        except Exception:
            st.markdown("Privacy Policy document is available at repository root.")
    with doc_tab2:
        try:
            with open("TERMS_OF_SERVICE.md", "r", encoding="utf-8") as f:
                st.markdown(f.read())
        except Exception:
            st.markdown("Terms of Service document is available at repository root.")
