"""
YouTube Transcript Service for a4ai Community Quiz
====================================================
Fetches transcript + metadata from YouTube URLs.
Compatible with both old (<1.0) and new (>=1.0) versions of youtube-transcript-api.
"""

import re
import logging
from typing import Optional
from urllib.parse import urlparse, parse_qs

import requests

# Try to import youtube_transcript_api (handles both old and new versions)
YT_API_AVAILABLE = False
YT_API_NEW = False  # True if v1.0+ (new instance-based API)

try:
    try:
    from youtube_transcript_api import YouTubeTranscriptApi
    YOUTUBE_AVAILABLE = True
except ImportError:
    YouTubeTranscriptApi = None
    YOUTUBE_AVAILABLE = False
    from youtube_transcript_api._errors import (
        TranscriptsDisabled,
        NoTranscriptFound,
        VideoUnavailable,
    )
    # Detect API version: new version has 'list' method, old has 'list_transcripts'
    if hasattr(YouTubeTranscriptApi, "list_transcripts"):
        YT_API_NEW = False  # old static-method API
    else:
        YT_API_NEW = True  # new instance-based API
    YT_API_AVAILABLE = True
except ImportError:
    logging.warning("youtube_transcript_api not installed. Run: pip install youtube-transcript-api")

logger = logging.getLogger(__name__)


class TranscriptFetchError(Exception):
    def __init__(self, message: str, code: str = "fetch_failed"):
        super().__init__(message)
        self.code = code
        self.message = message


def extract_video_id(url: str) -> Optional[str]:
    """Extracts YouTube video ID from any common URL format."""
    if not url:
        return None

    url = url.strip()

    if "youtu.be/" in url:
        match = re.search(r"youtu\.be/([a-zA-Z0-9_-]{11})", url)
        return match.group(1) if match else None

    if "/shorts/" in url:
        match = re.search(r"/shorts/([a-zA-Z0-9_-]{11})", url)
        return match.group(1) if match else None

    if "/embed/" in url:
        match = re.search(r"/embed/([a-zA-Z0-9_-]{11})", url)
        return match.group(1) if match else None

    parsed = urlparse(url)
    if parsed.hostname and "youtube.com" in parsed.hostname:
        qs = parse_qs(parsed.query)
        if "v" in qs and len(qs["v"][0]) == 11:
            return qs["v"][0]

    match = re.search(r"([a-zA-Z0-9_-]{11})", url)
    return match.group(1) if match else None


def fetch_video_metadata(video_id: str) -> dict:
    """Fetches video metadata using YouTube's oEmbed (no API key needed)."""
    oembed_url = f"https://www.youtube.com/oembed?url=https://www.youtube.com/watch?v={video_id}&format=json"

    try:
        resp = requests.get(oembed_url, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        return {
            "title": data.get("title", "Untitled"),
            "channel": data.get("author_name", "Unknown"),
            "channel_url": data.get("author_url", ""),
            "thumbnail": data.get("thumbnail_url", f"https://i.ytimg.com/vi/{video_id}/hqdefault.jpg"),
        }
    except Exception as e:
        logger.warning(f"oEmbed metadata fetch failed: {e}")
        return {
            "title": "Untitled Video",
            "channel": "Unknown",
            "channel_url": "",
            "thumbnail": f"https://i.ytimg.com/vi/{video_id}/hqdefault.jpg",
        }


def _normalize_segments(segments) -> list:
    """Convert FetchedTranscriptSnippet objects (new API) to dicts (old API format)."""
    normalized = []
    for seg in segments:
        if isinstance(seg, dict):
            normalized.append(seg)
        else:
            # New API: object with .text, .start, .duration attributes
            normalized.append({
                "text": getattr(seg, "text", ""),
                "start": getattr(seg, "start", 0.0),
                "duration": getattr(seg, "duration", 0.0),
            })
    return normalized


def fetch_transcript(video_id: str) -> dict:
    """Fetches transcript for a YouTube video. Works with both old and new library versions."""

    if not YT_API_AVAILABLE:
        raise TranscriptFetchError(
            "youtube-transcript-api not installed. Run: pip install youtube-transcript-api",
            code="dependency_missing"
        )

    try:
        languages = ["en", "hi", "en-IN", "en-US"]

        if YT_API_NEW:
            # ─── NEW API (v1.0+) — instance-based ───
            ytt = YouTubeTranscriptApi()
            transcript_list = ytt.list(video_id)

            transcript = None
            try:
                transcript = transcript_list.find_manually_created_transcript(languages)
            except NoTranscriptFound:
                try:
                    transcript = transcript_list.find_generated_transcript(languages)
                except NoTranscriptFound:
                    pass

            if transcript is None:
                for t in transcript_list:
                    transcript = t
                    break

            if transcript is None:
                raise TranscriptFetchError("No captions available for this video.", code="no_captions")

            fetched = transcript.fetch()
            # New API returns FetchedTranscript object with .snippets
            raw_segments = fetched.snippets if hasattr(fetched, "snippets") else fetched
            segments = _normalize_segments(raw_segments)
            language_code = transcript.language_code

        else:
            # ─── OLD API (<1.0) — static methods ───
            transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)

            transcript = None
            try:
                transcript = transcript_list.find_manually_created_transcript(languages)
            except NoTranscriptFound:
                try:
                    transcript = transcript_list.find_generated_transcript(languages)
                except NoTranscriptFound:
                    pass

            if transcript is None:
                for t in transcript_list:
                    transcript = t
                    break

            if transcript is None:
                raise TranscriptFetchError("No captions available for this video.", code="no_captions")

            segments = _normalize_segments(transcript.fetch())
            language_code = transcript.language_code

        full_text = " ".join(seg["text"] for seg in segments).strip()

        if not full_text or len(full_text) < 50:
            raise TranscriptFetchError("Transcript too short.", code="transcript_too_short")

        return {
            "text": full_text,
            "segments": segments,
            "language": language_code,
        }

    except TranscriptsDisabled:
        raise TranscriptFetchError("Captions are disabled on this video.", code="captions_disabled")
    except VideoUnavailable:
        raise TranscriptFetchError("Video is unavailable or private.", code="video_unavailable")
    except NoTranscriptFound:
        raise TranscriptFetchError("No transcript found for this video.", code="no_captions")
    except TranscriptFetchError:
        raise
    except Exception as e:
        logger.exception("Error fetching transcript")
        raise TranscriptFetchError(f"Failed to fetch transcript: {str(e)}", code="fetch_failed")


def fetch_video_data(url: str) -> dict:
    """Main entry point - fetches all video data."""
    video_id = extract_video_id(url)
    if not video_id:
        raise TranscriptFetchError("Invalid YouTube URL.", code="invalid_url")

    metadata = fetch_video_metadata(video_id)
    transcript_data = fetch_transcript(video_id)

    return {
        "video_id": video_id,
        "url": f"https://www.youtube.com/watch?v={video_id}",
        "title": metadata["title"],
        "channel": metadata["channel"],
        "channel_url": metadata["channel_url"],
        "thumbnail": metadata["thumbnail"],
        "transcript": transcript_data["text"],
        "transcript_segments": transcript_data["segments"],
        "language": transcript_data["language"],
        "transcript_word_count": len(transcript_data["text"].split()),
    }