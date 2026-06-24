"""
Modul pengambil komentar YouTube menggunakan yt-dlp.
"""
import re
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

try:
    from yt_dlp import YoutubeDL
    YT_DLP_AVAILABLE = True
except ImportError:
    YT_DLP_AVAILABLE = False
    logger.error("yt-dlp belum terinstall. Jalankan: pip install yt-dlp")


# ────────────────────────────────────────────────────────────────
# Validasi URL
# ────────────────────────────────────────────────────────────────

_YT_PATTERNS = [
    r"(https?://)?(www\.)?youtube\.com/watch\?v=[\w\-]+",
    r"(https?://)?youtu\.be/[\w\-]+",
    r"(https?://)?(www\.)?youtube\.com/shorts/[\w\-]+",
]


def is_valid_youtube_url(url: str) -> bool:
    """Periksa apakah URL adalah YouTube yang valid."""
    url = url.strip()
    return any(re.search(p, url) for p in _YT_PATTERNS)


# ────────────────────────────────────────────────────────────────
# Fungsi utama
# ────────────────────────────────────────────────────────────────

def get_youtube_comments(
    url: str,
    max_comments: int = 1000,
) -> List[Dict[str, Any]]:
    """
    Ambil komentar dari video YouTube.

    Parameters
    ----------
    url : str
        URL video YouTube.
    max_comments : int
        Jumlah maksimal komentar yang diambil (default 1000).

    Returns
    -------
    list of dict
        Setiap dict berisi: ``text``, ``author``, ``likes``.

    Raises
    ------
    ImportError
        Jika yt-dlp belum terinstall.
    ValueError
        Jika URL tidak valid.
    RuntimeError
        Jika terjadi kesalahan saat mengambil komentar.
    """
    if not YT_DLP_AVAILABLE:
        raise ImportError(
            "yt-dlp belum terinstall. Jalankan: pip install yt-dlp"
        )

    if not is_valid_youtube_url(url):
        raise ValueError(
            "URL YouTube tidak valid. "
            "Gunakan format: https://www.youtube.com/watch?v=..."
        )

    max_comments = min(int(max_comments), 1000)

    ydl_opts = {
        "getcomments": True,
        "extractor_args": {
            "youtube": {
                # Format: [max_comments, max_parents, max_replies, max_replies_per_thread]
                "max_comments": [f"{max_comments},all,all,all"],
            }
        },
        "quiet": True,
        "no_warnings": True,
        "skip_download": True,
        "ignoreerrors": True,
    }

    try:
        with YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
    except Exception as exc:
        raise RuntimeError(f"Gagal mengambil data dari YouTube: {exc}") from exc

    if info is None:
        raise RuntimeError("yt-dlp tidak dapat mengambil informasi video. "
                           "Periksa apakah video bersifat publik.")

    raw_comments: list = info.get("comments") or []

    if not raw_comments:
        return []

    comments: List[Dict[str, Any]] = []
    for c in raw_comments[:max_comments]:
        text = (c.get("text") or "").strip()
        if not text:
            continue
        comments.append(
            {
                "text": text,
                "author": c.get("author") or "Anonim",
                "likes": int(c.get("like_count") or 0),
            }
        )

    logger.info("Berhasil mengambil %d komentar dari %s", len(comments), url)
    return comments 