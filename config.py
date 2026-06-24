"""
Konfigurasi aplikasi Flask - SentimenMBG
"""
import os
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = os.path.abspath(os.path.dirname(__file__))


class Config:
    SECRET_KEY = os.environ.get("SECRET_KEY", "sentimen-mbg-flask-2025-secret")
    MAX_COMMENTS = 1000
    DEFAULT_COMMENTS = 500
    MODEL_DIR = os.path.join(BASE_DIR, "model")
    MODEL_PATH = os.path.join(MODEL_DIR, "model.pkl")
    VECTORIZER_PATH = os.path.join(MODEL_DIR, "vectorizer.pkl")
    LABEL_ENCODER_PATH = os.path.join(MODEL_DIR, "label_encoder.pkl")
    # Label mapping dari integer → string
    LABEL_MAP = {0: "Negatif", 1: "Netral", 2: "Positif"}
    # Warna sentimen untuk Chart.js
    SENTIMENT_COLORS = {
        "Positif": "#00e676",
        "Netral":  "#ffd740",
        "Negatif": "#ff4757",
    }
    # Word cloud
    WORDCLOUD_WIDTH  = 900
    WORDCLOUD_HEIGHT = 400
    WORDCLOUD_MAX_WORDS = 200