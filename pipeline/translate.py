"""
Bhasha-Setu — Hybrid Translate Service
AWS Translate (primary) + Google Translator (fallback for unsupported languages).
LLM polishing via Groq (llama-3.1-8b-instant) for conversational naturalness.
Handles chunking for the 9,000-byte AWS limit.

KEY FIX: SourceLanguageCode is now dynamic (was hardcoded "en").
LLM POLISH: Uses Groq instead of AWS Bedrock — free, no card needed,
            reuses the same GROQ_API_KEY already used in the AI chat.
"""

import json
import logging
import os
import requests
from typing import Callable, Optional

import boto3
from botocore.exceptions import ClientError
from deep_translator import GoogleTranslator

from pipeline.config import AWS_REGION, TRANSLATE_CHUNK_BYTES
from pipeline.sarvam import is_sarvam_available, translate_sarvam, SARVAM_LANGUAGES

log = logging.getLogger(__name__)

# Languages AWS Translate / ap-south-1 doesn't reliably support as TARGET
_GOOGLE_ONLY_TARGET_LANGS = {"or", "as"}

ISO_TO_LANG_NAME = {
    "hi": "Hindi", "ta": "Tamil", "te": "Telugu", "kn": "Kannada",
    "ml": "Malayalam", "bn": "Bengali", "mr": "Marathi", "gu": "Gujarati",
    "pa": "Punjabi", "or": "Odia", "en": "English", "as": "Assamese", "ur": "Urdu",
}


def _get_secret(key: str) -> str:
    """Try st.secrets first (Streamlit Cloud), then os.environ."""
    try:
        import streamlit as st
        val = st.secrets.get(key)
        if val:
            return str(val)
    except Exception:
        pass
    return os.environ.get(key, "")


def _split_into_chunks(text: str, max_bytes: int = TRANSLATE_CHUNK_BYTES) -> list:
    """
    Split text on sentence boundaries, staying within max_bytes per chunk.
    Falls back to word-level splitting for very long sentences.
    """
    if not text:
        return []
    sentences = []
    buf = ""
    for char in text:
        buf += char
        if char in ".?!।॥؟" and buf.strip():
            sentences.append(buf)
            buf = ""
    if buf.strip():
        sentences.append(buf)

    chunks, current = [], ""
    for sentence in sentences:
        candidate = current + sentence
        if len(candidate.encode("utf-8")) <= max_bytes:
            current = candidate
        else:
            if current:
                chunks.append(current.strip())
            if len(sentence.encode("utf-8")) > max_bytes:
                words, sub = sentence.split(), ""
                for word in words:
                    trial = sub + word + " "
                    if len(trial.encode("utf-8")) <= max_bytes:
                        sub = trial
                    else:
                        if sub:
                            chunks.append(sub.strip())
                        sub = word + " "
                current = sub
            else:
                current = sentence

    if current.strip():
        chunks.append(current.strip())
    return chunks


def translate_text(
    text: str,
    target_lang_code: str,
    source_lang_code: str = "en",
    progress_cb: Optional[Callable[[int, int], None]] = None,
    translate_engine: str = "auto",
) -> str:
    """
    Translate text from source language to target language.

    Args:
        text:             Source text to translate.
        target_lang_code: ISO code for target (e.g. 'hi', 'ta', 'fr').
        source_lang_code: ISO code for source (e.g. 'en', 'hi', 'fr').
                          Defaults to 'en' for backward compatibility.
        progress_cb:      Optional callable(chunk_index, total_chunks).
        translate_engine: 'auto', 'sarvam', 'aws', or 'google'.

    Returns:
        Translated text as a single string.
    """
    if not text or not text.strip():
        raise ValueError("Input text for translation is empty.")

    # Normalize codes — strip region suffix if present (e.g. "en-US" -> "en")
    src = source_lang_code.split("-")[0].lower()
    tgt = target_lang_code.split("-")[0].lower()

    if src == tgt:
        log.info("Source and target language are the same — skipping translation.")
        return text

    tgt_name = ISO_TO_LANG_NAME.get(tgt, "")
    src_name = ISO_TO_LANG_NAME.get(src, "English")

    # 1. SARVAM AI TRANSLATION (Mayura:v1) - State of the art Indic model
    use_sarvam = (translate_engine == "sarvam") or (
        translate_engine == "auto" and is_sarvam_available() and tgt_name in SARVAM_LANGUAGES
    )
    if use_sarvam and is_sarvam_available():
        try:
            log.info(f"Translating via Sarvam AI Mayura:v1: '{src}' -> '{tgt_name}'")
            if progress_cb: progress_cb(0, 1)
            result = translate_sarvam(text, source_lang_name=src_name, target_lang_name=tgt_name)
            if result and result.strip():
                if progress_cb: progress_cb(1, 1)
                return result
            log.warning("Sarvam translation returned empty/None; falling back to AWS/Google.")
        except Exception as e:
            log.warning(f"Sarvam translation failed ({e}), falling back to AWS/Google.")

    # 2. GOOGLE TRANSLATOR (when requested or for targets not supported by AWS)
    if translate_engine == "google" or tgt in _GOOGLE_ONLY_TARGET_LANGS:
        log.info(f"Translating via Google Translator: '{src}' -> '{tgt}'")
        if progress_cb: progress_cb(0, 1)
        chunks_text = _split_into_chunks(text, 4500)
        translated_chunks = []
        for chunk_text in chunks_text:
            translated_chunks.append(GoogleTranslator(source=src, target=tgt).translate(chunk_text))
        result = " ".join(translated_chunks)
        if progress_cb: progress_cb(1, 1)
        return result or text

    # 3. AWS TRANSLATE (Cloud) with fallback to Google Translator
    try:
        client = boto3.client("translate", region_name=AWS_REGION)
        chunks = _split_into_chunks(text)
        total  = len(chunks)
        log.info(f"Translating '{src}' -> '{tgt}' via AWS Translate | {total} chunk(s)")

        translated_parts = []
        for i, chunk in enumerate(chunks):
            if progress_cb:
                progress_cb(i, total)
            resp = client.translate_text(
                Text=chunk,
                SourceLanguageCode=src,
                TargetLanguageCode=tgt,
            )
            translated_parts.append(resp["TranslatedText"])

        if progress_cb:
            progress_cb(total, total)
        return " ".join(translated_parts)

    except Exception as e:
        log.warning(f"AWS Translate failed ({e}), falling back to Google Translator.")
        try:
            chunks_text = _split_into_chunks(text, 4500)
            translated_chunks = []
            for chunk_text in chunks_text:
                translated_chunks.append(GoogleTranslator(source=src, target=tgt).translate(chunk_text))
            result = " ".join(translated_chunks)
            if progress_cb: progress_cb(1, 1)
            return result or text
        except Exception as g_err:
            log.error(f"Google Translator fallback also failed: {g_err}")
            return text


# ─────────────────────────────────────────────────────────────────────────────
#  LLM TRANSLATION POLISH — Groq (free, no card needed)
# ─────────────────────────────────────────────────────────────────────────────

def polish_translation_with_llm(
    translated_text: str,
    target_language: str,
    original_text: str = "",
    source_language: str = "the source language",
) -> str:
    """
    Polish a machine-translated text using Groq (llama-3.1-8b-instant).
    Reuses the same GROQ_API_KEY already configured for the AI chat feature.

    Falls back silently to the original translation if Groq is unavailable.

    Args:
        translated_text:  Raw machine-translated text.
        target_language:  Human-readable target language name (e.g. "Hindi").
        original_text:    Original source text for context (optional).
        source_language:  Human-readable source language name (optional).

    Returns:
        Polished translated text, or the original if polishing fails.
    """
    if not translated_text:
        return translated_text or ""

    api_key = _get_secret("GROQ_API_KEY")

    if not api_key:
        log.warning(
            "LLM polish skipped: GROQ_API_KEY not set. "
            "Add it to .streamlit/secrets.toml to enable polish."
        )
        return translated_text

    context_block = ""
    if original_text:
        snippet = original_text[:400].replace("\n", " ")
        context_block = f"\n\nOriginal {source_language} (for context):\n{snippet}"

    prompt = (
        f"You are a professional {target_language} translator specializing in "
        f"spoken-word content for video dubbing.\n\n"
        f"The following text was machine-translated from {source_language} to "
        f"{target_language}. Lightly refine it so it sounds natural when spoken "
        f"aloud — fix awkward phrasing, unnatural idioms, and overly literal "
        f"translations. Keep the meaning identical. "
        f"Output ONLY the refined {target_language} text, nothing else.\n"
        f"{context_block}\n\n"
        f"Machine-translated text:\n{translated_text}"
    )

    # Keep prompt short to stay within free tier limits
    if len(prompt) > 12000:
        log.warning("LLM polish: prompt too long, skipping.")
        return translated_text

    try:
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        for m_name in ["qwen/qwen3.8-27b", "openai/gpt-oss-20b", "llama-3.1-8b-instant"]:
            try:
                resp = requests.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    headers=headers,
                    json={
                        "model": m_name,
                        "messages": [{"role": "user", "content": prompt}],
                        "max_tokens": 4096,
                        "temperature": 0.3,
                    },
                    timeout=30,
                )
                if resp.status_code == 200:
                    polished = resp.json()["choices"][0]["message"]["content"].strip()
                    log.info(f"LLM polish (Groq {m_name}): {len(translated_text)} -> {len(polished)} chars")
                    return polished
            except Exception:
                continue
        log.warning("LLM polish: No available Groq model responded.")
        return translated_text
    except Exception as e:
        log.warning(f"LLM polish failed (using original translation): {e}")
        return translated_text
