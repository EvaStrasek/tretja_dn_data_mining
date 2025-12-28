from pathlib import Path
import pandas as pd

APP_DIR = Path(__file__).resolve().parent
DATA_DIR = APP_DIR / "data"

def main():
    reviews_path = DATA_DIR / "reviews.csv"
    out_path = DATA_DIR / "reviews_with_sentiment.csv"

    if not reviews_path.exists():
        print("No reviews.csv found. Skipping.")
        return

    df = pd.read_csv(reviews_path)
    if df.empty or "text" not in df.columns:
        print("reviews.csv empty or missing 'text'. Skipping.")
        return

    # Filter year 2023 if date column exists
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"])
        df = df[df["date"].dt.year == 2023].copy()

    texts = df["text"].astype(str).tolist()

    # REQUIRED by assignment:
    from transformers import pipeline

    clf = pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english",
        device=-1,
    )

    # Batch to reduce spikes
    batch_size = 8
    sentiments = []
    confidences = []

    for i in range(0, len(texts), batch_size):
        chunk = texts[i:i+batch_size]
        outs = clf(chunk, truncation=True, max_length=128)

        for o in outs:
            label = str(o.get("label", "")).upper()
            score = float(o.get("score", 0.0))
            label_norm = "Positive" if label.startswith("POS") else "Negative"
            sentiments.append(label_norm)
            confidences.append(score)

    df["sentiment"] = sentiments
    df["confidence"] = confidences

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Saved: {out_path} ({len(df)} rows)")

if __name__ == "__main__":
    main()
