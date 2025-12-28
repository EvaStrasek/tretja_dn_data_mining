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

    # Prepare inputs
    df["text"] = df["text"].astype(str).fillna("").astype(str)
    texts = df["text"].tolist()

    has_rating = "rating" in df.columns
    ratings = df["rating"].tolist() if has_rating else [None] * len(df)

    # REQUIRED by assignment:
    from transformers import pipeline

    clf = pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english",
        device=-1,
    )

    def fallback_from_rating(r):
        """Use rating as tie-breaker for ultra-short/ambiguous texts."""
        if r is None or (isinstance(r, float) and pd.isna(r)):
            return None
        try:
            r = float(r)
        except Exception:
            return None
        if r >= 4:
            return ("Positive", 0.99)
        if r <= 2:
            return ("Negative", 0.99)
        return None  # rating 3 -> leave to model

    batch_size = 8
    sentiments = [None] * len(texts)
    confidences = [None] * len(texts)

    i = 0
    while i < len(texts):
        batch_idx = []
        batch_txt = []

        for j in range(i, min(i + batch_size, len(texts))):
            t = (texts[j] or "").strip()
            word_count = len(t.split())

            # If the text is very short (e.g., "finally"), use rating if available
            if word_count < 3:
                fb = fallback_from_rating(ratings[j])
                if fb is not None:
                    lab, sc = fb
                    sentiments[j] = lab
                    confidences[j] = sc
                    continue

            # Otherwise classify with the transformer model
            batch_idx.append(j)
            batch_txt.append(t[:5000])  # guard against extremely long text

        if batch_txt:
            outs = clf(batch_txt, truncation=True, max_length=128)
            for k, j in enumerate(batch_idx):
                o = outs[k]
                label = str(o.get("label", "")).upper()
                score = float(o.get("score", 0.0))
                label_norm = "Positive" if label.startswith("POS") else "Negative"
                sentiments[j] = label_norm
                confidences[j] = score

        i += batch_size

    df["sentiment"] = sentiments
    df["confidence"] = confidences

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Saved: {out_path} ({len(df)} rows)")


if __name__ == "__main__":
    main()
