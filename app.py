"""
SentimenMBG – Aplikasi analisis sentimen komentar YouTube
terhadap Program Makan Bergizi Gratis (MBG).

Jalankan:
    python app.py
"""
import io
import uuid
import logging

from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    send_file,
    jsonify,
)

from config import Config
from utils.youtube_scraper import get_youtube_comments
from utils.preprocessor import preprocess_comments
from utils.sentiment_predictor import predict_sentiment
from utils.visualizer import generate_wordcloud

# ────────────────────────────────────────────────────────────────
# Setup
# ────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config.from_object(Config)

# Penyimpanan in-memory untuk hasil analisis (key = session_id)
_store: dict = {}


# ────────────────────────────────────────────────────────────────
# Helper
# ────────────────────────────────────────────────────────────────

def _pct(n: int, total: int) -> float:
    return round(n / total * 100, 1) if total else 0.0


def _build_dashboard(df, url: str, session_id: str) -> dict:
    """Siapkan semua data yang dibutuhkan oleh template result.html."""
    total    = len(df)
    counts   = df["sentiment"].value_counts().to_dict()
    positive = counts.get("Positif", 0)
    neutral  = counts.get("Netral",  0)
    negative = counts.get("Negatif", 0)

    # Word cloud
    wc_texts = df["wc_text"].tolist() if "wc_text" in df.columns else df["clean_text"].tolist()
    wordcloud_img = generate_wordcloud(wc_texts)

    # Tabel (max 200 baris untuk UI, full data via CSV)
    table_rows = []
    for _, row in df.head(200).iterrows():
        raw_text = str(row.get("text", ""))
        table_rows.append(
            {
                "text": raw_text[:180] + "…" if len(raw_text) > 180 else raw_text,
                "author": row.get("author", "Anonim"),
                "sentiment": row.get("sentiment", "Netral"),
                "confidence": f"{float(row.get('confidence', 0)) * 100:.1f}",
                "likes": int(row.get("likes", 0)),
            }
        )

    return {
        "url": url,
        "session_id": session_id,
        "total": total,
        "positive": positive,
        "neutral": neutral,
        "negative": negative,
        "positive_pct": _pct(positive, total),
        "neutral_pct":  _pct(neutral,  total),
        "negative_pct": _pct(negative, total),
        "wordcloud_img": wordcloud_img,
        "table_rows": table_rows,
        "chart_labels": ["Positif", "Netral", "Negatif"],
        "chart_values": [positive, neutral, negative],
        "chart_colors": [
            Config.SENTIMENT_COLORS["Positif"],
            Config.SENTIMENT_COLORS["Netral"],
            Config.SENTIMENT_COLORS["Negatif"],
        ],
    }


# ────────────────────────────────────────────────────────────────
# Routes
# ────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    url          = (request.form.get("youtube_url") or "").strip()
    max_comments = min(
        int(request.form.get("max_comments", Config.DEFAULT_COMMENTS)),
        Config.MAX_COMMENTS,
    )

    if not url:
        return render_template("index.html",
                               error="URL YouTube tidak boleh kosong.")

    # 1. Ambil komentar ─────────────────────────────────────────
    try:
        comments = get_youtube_comments(url, max_comments=max_comments)
    except (ImportError, ValueError) as exc:
        return render_template("index.html", error=str(exc), url=url)
    except Exception as exc:
        logger.exception("Gagal mengambil komentar")
        return render_template(
            "index.html",
            error=f"Gagal mengambil komentar: {exc}",
            url=url,
        )

    if not comments:
        return render_template(
            "index.html",
            error=(
                "Tidak ada komentar ditemukan. "
                "Pastikan komentar diaktifkan pada video tersebut."
            ),
            url=url,
        )

    # 2. Preprocessing ──────────────────────────────────────────
    df = preprocess_comments(comments)
    if df.empty:
        return render_template(
            "index.html",
            error="Tidak ada komentar yang dapat diproses setelah preprocessing.",
            url=url,
        )

    # 3. Prediksi sentimen ──────────────────────────────────────
    df = predict_sentiment(df)

    # 4. Simpan ke store ────────────────────────────────────────
    session_id = str(uuid.uuid4())
    _store[session_id] = df

    # 5. Siapkan data dashboard ─────────────────────────────────
    data = _build_dashboard(df, url, session_id)

    return render_template("result.html", data=data)


@app.route("/export/<session_id>")
def export_csv(session_id: str):
    """Unduh hasil analisis sebagai CSV (UTF-8 BOM agar terbaca di Excel)."""
    df = _store.get(session_id)
    if df is None:
        return redirect(url_for("index"))

    export_df = df[
        [c for c in ("text", "author", "sentiment", "confidence", "likes")
         if c in df.columns]
    ].copy()

    col_rename = {
        "text": "Komentar",
        "author": "Penulis",
        "sentiment": "Sentimen",
        "confidence": "Kepercayaan",
        "likes": "Suka",
    }
    export_df.rename(columns=col_rename, inplace=True)

    buf = io.StringIO()
    export_df.to_csv(buf, index=False, encoding="utf-8")
    csv_bytes = ("\ufeff" + buf.getvalue()).encode("utf-8")

    return send_file(
        io.BytesIO(csv_bytes),
        mimetype="text/csv",
        as_attachment=True,
        download_name="hasil_sentimen_mbg.csv",
    )


@app.route("/api/status")
def api_status():
    """Endpoint cek kesehatan sederhana."""
    return jsonify({"status": "ok", "store_size": len(_store)})


# ────────────────────────────────────────────────────────────────
# Entry point
# ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)