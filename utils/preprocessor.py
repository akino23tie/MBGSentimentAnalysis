"""
Preprocessing teks bahasa Indonesia untuk analisis sentimen MBG.
Pipeline: clean → normalize slang → remove stopwords → (optional stem)
"""
import re
import string
import logging
import unicodedata
from typing import List, Dict, Any

import pandas as pd

logger = logging.getLogger(__name__)

# ────────────────────────────────────────────────────────────────
# Sastrawi (stemmer Bahasa Indonesia) – opsional
# ────────────────────────────────────────────────────────────────
try:
    from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

    _stemmer = StemmerFactory().create_stemmer()
    SASTRAWI_AVAILABLE = True
except ImportError:
    SASTRAWI_AVAILABLE = False
    logger.warning("Sastrawi tidak ditemukan; stemming dinonaktifkan.")

# ────────────────────────────────────────────────────────────────
# NLTK stopwords Bahasa Indonesia – opsional
# ────────────────────────────────────────────────────────────────
_STOPWORDS: set = set()
try:
    import nltk

    try:
        from nltk.corpus import stopwords

        _STOPWORDS = set(stopwords.words("indonesian"))
    except LookupError:
        nltk.download("stopwords", quiet=True)
        from nltk.corpus import stopwords

        _STOPWORDS = set(stopwords.words("indonesian"))
except Exception:
    pass

# Tambah stopword manual yang sering muncul di komentar YouTube
_STOPWORDS.update(
    {
        "yg", "yang", "dgn", "dengan", "di", "ke", "dari", "ini", "itu",
        "dan", "atau", "tapi", "tp", "juga", "jg", "sih", "nih", "deh",
        "lah", "dong", "kah", "pun", "ya", "iya", "ok", "oke",
        "the", "is", "a", "of", "to",  # noise Inggris
        "nya", "mu", "ku",
    }
)

# ────────────────────────────────────────────────────────────────
# Kamus normalisasi kata gaul / singkatan Indonesia
# ────────────────────────────────────────────────────────────────
SLANG_DICT: Dict[str, str] = {
    # Umum
    "gk": "tidak", "ga": "tidak", "gak": "tidak", "ngga": "tidak",
    "nggak": "tidak", "tdk": "tidak", "tak": "tidak", "ndak": "tidak",
    "bkn": "bukan",
    "udh": "sudah", "udah": "sudah", "sdh": "sudah",
    "blm": "belum", "blum": "belum",
    "sm": "sama", "skrg": "sekarang", "skg": "sekarang",
    "jg": "juga", "bs": "bisa", "dr": "dari",
    "lg": "lagi", "lgi": "lagi", "msh": "masih",
    "emg": "memang", "emang": "memang", "mmg": "memang",
    "krn": "karena", "karna": "karena",
    "spy": "supaya", "biar": "supaya",
    "utk": "untuk", "tuk": "untuk",
    "gimana": "bagaimana", "gmn": "bagaimana",
    "knp": "kenapa", "ngapain": "mengapa",
    "hrs": "harus",
    "dgn": "dengan",
    "pdhl": "padahal",
    "tp": "tapi",
    "jd": "jadi", "jadi": "jadi",
    "ky": "kayak", "kyk": "kayak", "spt": "seperti",
    "gitu": "begitu", "gini": "begini",
    "org": "orang",
    "sy": "saya", "gue": "saya", "gw": "saya", "aku": "saya",
    "lu": "kamu", "lo": "kamu", "km": "kamu",
    "mrk": "mereka",
    "py": "punya",
    "kl": "kalau", "kalo": "kalau",
    # MBG spesifik
    "mbg": "makan bergizi gratis",
    "mkn": "makan",
    "bgs": "bagus",
    "bgst": "bagus sekali",
    "mantap": "bagus",
    "mantul": "bagus sekali",
    "keren": "bagus",
    "jos": "bagus",
    "gajelas": "tidak jelas",
    "asbun": "asal bunyi",
    "hoaks": "hoax",
    "boong": "bohong",
    "boongan": "bohong",
    "korup": "korupsi",
    "dikorup": "dikorupsi",
    "nyolong": "mencuri",
    "cuan": "untung",
}


# ────────────────────────────────────────────────────────────────
# Pipeline preprocessing
# ────────────────────────────────────────────────────────────────

def _strip_emoji(text: str) -> str:
    """Hapus emoji dan karakter non-ASCII yang tidak perlu."""
    # Normalisasi unicode terlebih dahulu
    text = unicodedata.normalize("NFKC", text)
    # Hapus karakter di luar rentang ASCII dasar
    return text.encode("ascii", errors="ignore").decode("ascii")


def clean_text(text: str) -> str:
    """
    Bersihkan teks mentah:
    lowercase → hapus URL/mention/hashtag/emoji → hapus karakter khusus.
    """
    if not isinstance(text, str) or not text.strip():
        return ""

    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)       # URL
    text = re.sub(r"@\w+", " ", text)                    # mention
    text = re.sub(r"#\w+", " ", text)                    # hashtag
    text = _strip_emoji(text)
    text = re.sub(r"[^a-z\s]", " ", text)               # angka & simbol
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_slang(text: str) -> str:
    """Ganti kata gaul / singkatan dengan bentuk baku."""
    words = text.split()
    return " ".join(SLANG_DICT.get(w, w) for w in words)


def remove_stopwords(text: str) -> str:
    """Hapus stopword Bahasa Indonesia."""
    if not _STOPWORDS:
        return text
    words = text.split()
    return " ".join(w for w in words if w not in _STOPWORDS and len(w) > 2)


def stem_text(text: str) -> str:
    """Stemming dengan Sastrawi (jika tersedia)."""
    if SASTRAWI_AVAILABLE:
        return _stemmer.stem(text)
    return text


def preprocess_text(text: str, apply_stemming: bool = False) -> str:
    """Pipeline preprocessing lengkap untuk satu teks."""
    text = clean_text(text)
    if not text:
        return ""
    text = normalize_slang(text)
    if apply_stemming:
        text = stem_text(text)
    return text


def preprocess_text_for_wordcloud(text: str) -> str:
    """Preprocessing lebih dalam untuk word cloud (buang stopword)."""
    text = preprocess_text(text, apply_stemming=False)
    text = remove_stopwords(text)
    return text


# ────────────────────────────────────────────────────────────────
# Fungsi utama: DataFrame preprocessing
# ────────────────────────────────────────────────────────────────

def preprocess_comments(
    comments: List[Dict[str, Any]],
    apply_stemming: bool = False,
) -> pd.DataFrame:
    """
    Bersihkan dan praproses list komentar menjadi DataFrame siap-pakai.

    Kolom output
    ------------
    text         : komentar asli
    author       : nama penulis
    likes        : jumlah likes
    clean_text   : teks setelah preprocessing (untuk model)
    wc_text      : teks untuk word cloud (tanpa stopword)
    """
    if not comments:
        return pd.DataFrame()

    df = pd.DataFrame(comments)

    # Pastikan kolom yang diperlukan ada
    for col in ("text", "author", "likes"):
        if col not in df.columns:
            df[col] = "" if col != "likes" else 0

    df["clean_text"] = df["text"].apply(
        lambda t: preprocess_text(t, apply_stemming=apply_stemming)
    )
    df["wc_text"] = df["text"].apply(preprocess_text_for_wordcloud)

    # Buang baris tanpa teks bersih
    df = df[df["clean_text"].str.strip().astype(bool)].reset_index(drop=True)

    logger.info("Preprocessing selesai: %d komentar valid", len(df))
    return df