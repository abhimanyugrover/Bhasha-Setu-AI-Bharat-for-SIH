"""
Bhasha Setu — LLM Transcript Summarizer
Generates a concise summary of the transcript shown in the results panel.

Uses Groq (llama-3.1-8b-instant) — free, no card needed.
Reuses the same GROQ_API_KEY already configured for the AI chat feature.

Fallback chain:
  1. Groq (llama-3.1-8b-instant) — free tier, very fast
  2. Google Gemini (gemini-1.5-flash) — free tier (optional)
  3. Returns None silently if no backend is configured

Configuration — add ANY ONE of these to .streamlit/secrets.toml:
  GROQ_API_KEY   = "gsk_..."
  GEMINI_API_KEY = "AIza..."
"""

import os
import logging
import requests
from typing import Optional

log = logging.getLogger(__name__)

_PROMPT = (
    "You are a helpful assistant. Read the following video transcript and write a "
    "1-2 sentence plain-{language} summary of what the video is about. "
    "Be concise. Output only the summary, nothing else.\n\n"
    "Transcript:\n{transcript}"
)

_MAX_TRANSCRIPT_CHARS = 8000


def _get_secret(key: str) -> Optional[str]:
    """Try st.secrets first (Streamlit Cloud), then os.environ."""
    try:
        import streamlit as st
        val = st.secrets.get(key)
        if val:
            return str(val)
    except Exception:
        pass
    return os.environ.get(key)


# ── Backend: Groq ─────────────────────────────────────────────────────────────

def _summarize_groq(transcript: str, language: str) -> Optional[str]:
    api_key = _get_secret("GROQ_API_KEY")
    if not api_key:
        return None
    try:
        prompt_content = _PROMPT.replace("{language}", str(language)).replace("{transcript}", str(transcript))
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        messages = [{"role": "user", "content": prompt_content}]
        for m_name in ["qwen/qwen3.8-27b", "openai/gpt-oss-20b", "llama-3.1-8b-instant", "llama3-8b-8192"]:
            try:
                payload = {
                    "model": m_name,
                    "messages": messages,
                    "max_tokens": 256,
                    "temperature": 0.3,
                }
                resp = requests.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    headers=headers, json=payload, timeout=30,
                )
                resp.raise_for_status()
                data = resp.json()
                return data["choices"][0]["message"]["content"].strip()
            except Exception:
                continue
        return None
    except Exception as e:
        log.warning(f"Groq summarizer failed: {e}")
        return None


# ── Backend: Gemini (optional fallback) ───────────────────────────────────────

def _summarize_gemini(transcript: str, language: str) -> Optional[str]:
    api_key = _get_secret("GEMINI_API_KEY")
    if not api_key:
        return None
    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-1.5-flash")
        prompt_content = _PROMPT.replace("{language}", str(language)).replace("{transcript}", str(transcript))
        resp  = model.generate_content(
            prompt_content,
            generation_config={"max_output_tokens": 256, "temperature": 0.3},
        )
        return resp.text.strip()
    except ImportError:
        log.debug("google-generativeai not installed — skipping Gemini backend")
        return None
    except Exception as e:
        log.warning(f"Gemini summarizer failed: {e}")
        return None


# ── Public API ────────────────────────────────────────────────────────────────

def summarize_transcript(transcript: str, language: str = "English") -> Optional[str]:
    """
    Summarize a transcript using Groq (primary) or Gemini (fallback).

    Returns a 1-2 sentence summary string, or None if no backend is available.
    The caller (app.py) wraps this in a try/except so failure is always safe.
    """
    if not transcript or not transcript.strip():
        return None

    text = transcript[:_MAX_TRANSCRIPT_CHARS].strip()
    if len(transcript) > _MAX_TRANSCRIPT_CHARS:
        text += "..."

    for fn in [_summarize_groq, _summarize_gemini]:
        result = fn(text, language)
        if result:
            log.info(f"Transcript summarized via {fn.__name__} ({len(result)} chars)")
            return f"AI Summary: {result}"

    return None
