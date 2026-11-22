"""
Bhasha Setu — Video URL Downloader
Downloads videos from YouTube and 1000+ other sites using yt-dlp.

Supports: YouTube, Vimeo, Instagram, Twitter/X, Facebook, Dailymotion,
          TED, Bilibili, Reddit, and most public video platforms.

Quality cap: 720p MP4 — keeps file sizes manageable for the dubbing pipeline.

Public API:
    get_video_info(url)        → metadata dict (no download, instant)
    download_to_temp(url, ...) → local temp path (pipeline handles S3 upload)
"""

import os
import tempfile
import logging
import ipaddress
import socket
import urllib.parse
from typing import Callable, Optional

log = logging.getLogger(__name__)

MAX_HEIGHT = 720

def validate_video_url(url: str) -> str:
    """
    Validate that the URL is a safe, publicly reachable HTTP/HTTPS URL.
    Blocks private networks, loopback, link-local, cloud metadata, and dangerous schemes (SSRF protection).
    """
    if not url or not isinstance(url, str):
        raise ValueError("URL must be a non-empty string.")

    url = url.strip()
    try:
        parsed = urllib.parse.urlparse(url)
    except Exception as e:
        raise ValueError(f"Malformed URL: {e}")

    if parsed.scheme.lower() not in ("http", "https"):
        raise ValueError(f"Disallowed URL scheme '{parsed.scheme}'. Only HTTP and HTTPS are permitted.")

    hostname = parsed.hostname
    if not hostname:
        raise ValueError("URL must have a valid hostname.")

    disallowed_hosts = {
        "localhost", "127.0.0.1", "::1", "0.0.0.0",
        "169.254.169.254", "metadata.google.internal", "instance-data"
    }
    if hostname.lower() in disallowed_hosts:
        raise ValueError(f"Access to '{hostname}' is not permitted.")

    try:
        addr_info = socket.getaddrinfo(hostname, None)
    except socket.gaierror as e:
        raise ValueError(f"Could not resolve hostname '{hostname}': {e}")

    for entry in addr_info:
        ip_str = entry[4][0]
        try:
            ip = ipaddress.ip_address(ip_str)
            if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_multicast or ip.is_reserved or ip.is_unspecified:
                raise ValueError(f"URL resolves to disallowed private/internal IP address: {ip_str}")
        except ValueError as e:
            if "disallowed private/internal IP" in str(e):
                raise
            pass

    return url

_PLATFORM_MAP = {
    "youtube":     "YouTube",
    "vimeo":       "Vimeo",
    "instagram":   "Instagram",
    "twitter":     "Twitter / X",
    "facebook":    "Facebook",
    "dailymotion": "Dailymotion",
    "ted":         "TED",
    "bilibili":    "Bilibili",
    "reddit":      "Reddit",
}

_FORMAT = (
    f"bestvideo[height<={MAX_HEIGHT}][ext=mp4]+bestaudio[ext=m4a]"
    f"/bestvideo[height<={MAX_HEIGHT}]+bestaudio"
    f"/best[height<={MAX_HEIGHT}][ext=mp4]"
    f"/best[height<={MAX_HEIGHT}]"
    f"/best"
)

# Realistic Chrome user-agent — avoids most platform bot-detection
_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/125.0.0.0 Safari/537.36"
)


def _platform(info: dict) -> str:
    extractor = info.get("extractor_key", "").lower()
    return next(
        (v for k, v in _PLATFORM_MAP.items() if k in extractor),
        extractor.capitalize() or "Unknown"
    )


def _friendly_error(e) -> str:
    msg = str(e)
    if "Private video"    in msg: return "This video is private and cannot be accessed."
    if "age"              in msg.lower(): return "This video is age-restricted."
    if "unavailable"      in msg.lower(): return "This video is unavailable or has been removed."
    if "removed"          in msg.lower(): return "This video has been removed."
    if "copyright"        in msg.lower(): return "Video unavailable due to copyright restrictions."
    if "confirm your age" in msg.lower(): return "This video requires sign-in to watch."
    if "403"              in msg: return "YouTube blocked the request (403). Try updating yt-dlp: pip install --upgrade yt-dlp"
    return f"Could not access video: {msg[:200]}"


def _make_opts(extra: dict = None) -> dict:
    """Base yt-dlp options shared by all calls with security defaults."""
    if extra is None: extra = {}
    return {
        "quiet":               True,
        "no_warnings":         True,
        "noplaylist":          True,
        "http_headers":        {"User-Agent": _UA},
        "no_check_certificate": False,
        **extra,
    }


# ── Public: metadata fetch ────────────────────────────────────────────────────

def get_video_info(url: str) -> dict:
    """
    Fetch video metadata WITHOUT downloading. Fast (~1-2 seconds).

    Returns dict: title, duration (seconds|None), thumbnail (url|None),
                  uploader (str|None), platform (str), url (str)

    Raises ValueError for private/unavailable/bad URLs or SSRF attempts.
    Raises RuntimeError if yt-dlp is not installed.
    """
    url = validate_video_url(url)
    try:
        import yt_dlp
    except ImportError:
        raise RuntimeError("yt-dlp not installed. Run: pip install yt-dlp")

    opts = _make_opts({"skip_download": True})
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(url, download=False)
            if not info:
                raise ValueError("Could not fetch video information.")
            return {
                "title":     info.get("title", "Untitled Video"),
                "duration":  info.get("duration"),
                "thumbnail": info.get("thumbnail"),
                "uploader":  info.get("uploader") or info.get("channel"),
                "platform":  _platform(info),
                "url":       url,
            }
    except yt_dlp.utils.DownloadError as e:
        raise ValueError(_friendly_error(e))


# ── Public: download to temp ──────────────────────────────────────────────────

def download_to_temp(
    url:         str,
    progress_cb: Optional[Callable[[float, str], None]] = None,
) -> str:
    """
    Download a video from URL to a system temp file. Returns the local path.

    The CALLER must delete the file after use.
    The pipeline's Stage 1 calls this then immediately uploads to S3.

    Args:
        url:         Public video URL.
        progress_cb: Optional callable(percent: float, message: str).

    Returns:
        Absolute path to downloaded MP4 temp file.

    Raises:
        ValueError  : private / unavailable / bad URL
        RuntimeError: yt-dlp not installed, or unexpected failure
    """
    url = validate_video_url(url)
    try:
        import yt_dlp
    except ImportError:
        raise RuntimeError("yt-dlp not installed. Run: pip install yt-dlp")

    tmp_dir  = tempfile.mkdtemp(prefix="bhasha_setu_")
    try:
        out_tmpl = os.path.join(tmp_dir, "%(id)s.%(ext)s")
        _last    = [-1.0]

        def _hook(d: dict):
            if not progress_cb:
                return
            status = d.get("status", "")
            if status == "downloading":
                total = d.get("total_bytes") or d.get("total_bytes_estimate") or 0
                done  = d.get("downloaded_bytes", 0)
                pct   = min(95.0, (done / total * 100) if total else _last[0] + 0.5)
                if pct - _last[0] >= 1.0:
                    _last[0] = pct
                    speed = (d.get("speed") or 0) / (1024 * 1024)
                    eta   = d.get("eta") or 0
                    msg   = f"Fetching video… {pct:.0f}%"
                    if speed > 0.01: msg += f" · {speed:.1f} MB/s"
                    if eta   > 0:    msg += f" · ~{eta}s left"
                    progress_cb(pct, msg)
            elif status == "finished":
                progress_cb(97.0, "Merging audio & video streams…")

        opts = _make_opts({
            "format":              _FORMAT,
            "outtmpl":             out_tmpl,
            "progress_hooks":      [_hook],
            "merge_output_format": "mp4",
            "retries":             5,
            "fragment_retries":    5,
        })

        try:
            if progress_cb:
                progress_cb(0.0, "Connecting to video source…")

            with yt_dlp.YoutubeDL(opts) as ydl:
                info = ydl.extract_info(url, download=True)

            if not info:
                raise ValueError("Download returned no information.")

            # Find the output file in tmp_dir
            mp4s = sorted(
                [os.path.join(tmp_dir, f) for f in os.listdir(tmp_dir)
                 if f.endswith(".mp4")],
                key=os.path.getmtime,
            )
            if not mp4s:
                # Check for any video file if mp4 merge didn't happen
                all_vids = sorted(
                    [os.path.join(tmp_dir, f) for f in os.listdir(tmp_dir)
                     if os.path.splitext(f)[1] in (".mp4", ".mkv", ".webm")],
                    key=os.path.getmtime,
                )
                if not all_vids:
                    raise RuntimeError("Download finished but no output file found.")
                actual = all_vids[-1]
            else:
                actual = mp4s[-1]

            size_mb = os.path.getsize(actual) / (1024 * 1024)
            if progress_cb:
                progress_cb(100.0, f"Video ready — {size_mb:.1f} MB")
            log.info(f"Temp download: {actual} ({size_mb:.1f} MB)")
            return actual

        except yt_dlp.utils.DownloadError as e:
            raise ValueError(_friendly_error(e))
        except Exception as e:
            raise RuntimeError(f"Unexpected download error: {e}") from e
    except Exception:
        import shutil
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise
