import os
import torch
import pandas as pd

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)

MODEL_DIR = "model"

# =========================
# Load model sekali saat startup
# =========================

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_DIR
)

model.eval()

LABEL_MAP = {
    0: "Negatif",
    1: "Netral",
    2: "Positif"
}


# =========================
# Prediksi satu teks
# =========================

def predict_single(text):

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512
    )

    with torch.no_grad():
        outputs = model(**inputs)

    probs = torch.softmax(
        outputs.logits,
        dim=1
    )

    confidence = torch.max(probs).item()

    pred = torch.argmax(
        probs,
        dim=1
    ).item()

    return LABEL_MAP[pred], confidence


# =========================
# Prediksi DataFrame
# =========================

def predict_sentiment(df):

    sentiments = []
    confidences = []

    texts = df["clean_text"].fillna("").tolist()

    for text in texts:

        label, conf = predict_single(text)

        sentiments.append(label)
        confidences.append(round(conf, 4))

    df = df.copy()

    df["sentiment"] = sentiments
    df["confidence"] = confidences

    return df