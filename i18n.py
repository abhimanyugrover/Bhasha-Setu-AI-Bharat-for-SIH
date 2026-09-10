"""
Bhasha Setu — Internationalization (i18n) Module
Loads translation JSON files and provides t() function for UI string localization.
"""
import json
import os
from functools import lru_cache

try:
    import streamlit as st
except ImportError:
    st = None

_TRANSLATIONS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "translations")

UI_LANGUAGES = {
    "English": "en",
    "हिन्दी (Hindi)": "hi",
    "বাংলা (Bengali)": "bn",
    "తెలుగు (Telugu)": "te",
    "मराठी (Marathi)": "mr",
    "தமிழ் (Tamil)": "ta",
    "ગુજરાતી (Gujarati)": "gu",
    "ಕನ್ನಡ (Kannada)": "kn",
    "മലയാളം (Malayalam)": "ml",
    "ਪੰਜਾਬੀ (Punjabi)": "pa",
    "ଓଡ଼ିଆ (Odia)": "or",
    "অসমীয়া (Assamese)": "as",
    "اردو (Urdu)": "ur",
}

@lru_cache(maxsize=None)
def _load_translations(lang_code: str) -> dict:
    """Load and cache a translation file."""
    path = os.path.join(_TRANSLATIONS_DIR, f"{lang_code}.json")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return {}

def get_ui_lang() -> str:
    """Get current UI language code from session state."""
    if st is not None:
        try:
            return st.session_state.get("ui_lang", "en")
        except Exception:
            pass
    return "en"

def t(key: str, **kwargs) -> str:
    """Get localized string. Falls back to English, then to key itself.
    
    Usage:
        t("hero_title")  # Simple lookup
        t("upload_progress", pct="45%")  # With format arguments
    """
    lang = get_ui_lang()
    translations = _load_translations(lang)
    text = translations.get(key)
    if text is None:
        # Fallback to English
        en = _load_translations("en")
        text = en.get(key, key)
    if kwargs:
        try:
            text = text.format(**kwargs)
        except (KeyError, IndexError):
            pass
    return text
