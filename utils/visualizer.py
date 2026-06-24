"""
Modul visualisasi – menghasilkan Word Cloud sebagai gambar base64.
"""
import io
import base64
import logging
from typing import List, Optional

logger = logging.getLogger(__name__)

# Matplotlib harus non-interactive sebelum diimpor lebih lanjut
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from wordcloud import WordCloud
    WC_AVAILABLE = True
except ImportError:
    WC_AVAILABLE = False
    logger.warning("wordcloud belum terinstall – word cloud dinonaktifkan.")

from config import Config


def generate_wordcloud(
    texts: List[str],
    max_words: int = None,
) -> Optional[str]:
    """
    Buat Word Cloud dari list teks dan kembalikan sebagai data URI base64.

    Parameters
    ----------
    texts : list of str
        List teks yang sudah dipreproses (tanpa stopword).
    max_words : int, optional
        Jumlah maksimal kata (default dari Config.WORDCLOUD_MAX_WORDS).

    Returns
    -------
    str or None
        Data URI ``data:image/png;base64,...`` atau None jika gagal.
    """
    if not WC_AVAILABLE:
        return None

    if max_words is None:
        max_words = Config.WORDCLOUD_MAX_WORDS

    # Gabungkan semua teks
    combined = " ".join(str(t) for t in texts if t and str(t).strip())
    if not combined.strip():
        return None

    try:
        wc = WordCloud(
            width=Config.WORDCLOUD_WIDTH,
            height=Config.WORDCLOUD_HEIGHT,
            background_color="#0e1228",
            colormap="cool",
            max_words=max_words,
            prefer_horizontal=0.7,
            min_font_size=10,
            max_font_size=100,
            relative_scaling=0.5,
            collocations=False,
        ).generate(combined)

        fig, ax = plt.subplots(
            figsize=(Config.WORDCLOUD_WIDTH / 100, Config.WORDCLOUD_HEIGHT / 100),
            facecolor="#0e1228",
        )
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        fig.tight_layout(pad=0)

        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=130,
                    facecolor="#0e1228", bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)

        b64 = base64.b64encode(buf.read()).decode("utf-8")
        return f"data:image/png;base64,{b64}"

    except Exception as exc:
        logger.error("Gagal membuat word cloud: %s", exc)
        return None