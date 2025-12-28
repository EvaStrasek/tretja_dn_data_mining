import sys
import subprocess
from pathlib import Path

import pandas as pd
import streamlit as st

from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt

st.set_page_config(page_title="HW3", layout="wide")
st.title("HW3 – Web Scraping & Sentiment Analysis (2023)")

APP_DIR = Path(__file__).resolve().parent
DATA_DIR = APP_DIR / "data"

# -------------------------
# Sidebar
# -------------------------
section = st.sidebar.radio("Go to", ["Products", "Testimonials", "Reviews"])

st.sidebar.divider()
if st.sidebar.button("Scrape / Refresh data"):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        [sys.executable, str(APP_DIR / "scrape_data.py")],
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        st.sidebar.success("Scraping finished ✅")
    else:
        st.sidebar.error("Scraping failed ❌")

    with st.sidebar.expander("Scraper log"):
        st.code((result.stdout or "") + "\n" + (result.stderr or ""))

# -------------------------
# Helpers
# -------------------------
@st.cache_data(show_spinner=False)
def load_csv_cached(path_str: str) -> pd.DataFrame:
    path = Path(path_str)
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def load_csv(name: str) -> pd.DataFrame:
    return load_csv_cached(str(DATA_DIR / name))


def month_label(dt: pd.Timestamp) -> str:
    return dt.strftime("%b %Y")


@st.cache_resource(show_spinner=True)
def get_sentiment_pipeline(model_name: str):
    """
    Hugging Face Transformers pipeline (required by assignment).
    Lazy-imports transformers + torch to reduce startup memory.
    """
    # Lazy imports to avoid loading torch/transformers at app startup
    import os
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    import torch
    from transformers import pipeline

    # Small, common sentiment model:
    # distilbert-base-uncased-finetuned-sst-2-english
    # (you can swap for another HF sentiment model if allowed)
    clf = pipeline(
        "sentiment-analysis",
        model=model_name,
        device=-1,  # CPU
    )

    # Optional: reduce CPU threads (can help a bit with overhead)
    try:
        torch.set_num_threads(1)
    except Exception:
        pass

    return clf


def run_sentiment(texts: list[str], model_name: str, batch_size: int = 8, max_length: int = 128) -> pd.DataFrame:
    """
    Classify every review as Positive/Negative using transformers pipeline.
    Returns dataframe with columns: sentiment, confidence
    """
    nlp = get_sentiment_pipeline(model_name)

    rows = []
    # Process in batches to avoid spikes
    for i in range(0, len(texts), batch_size):
        chunk = [str(t) for t in texts[i:i + batch_size]]

        outputs = nlp(
            chunk,
            truncation=True,
            max_length=max_length,
        )

        for out in outputs:
            label = str(out.get("label", "")).upper()
            score = float(out.get("score", 0.0))

            # Standardize to Positive/Negative
            if label.startswith("POS"):
                label_norm = "Positive"
            elif label.startswith("NEG"):
                label_norm = "Negative"
            else:
                # fallback (some models use LABEL_0/LABEL_1)
                # SST-2 is usually NEGATIVE/POSITIVE anyway
                label_norm = "Positive" if "1" in label else "Negative"

            rows.append({"sentiment": label_norm, "confidence": score})

    return pd.DataFrame(rows)


# -------------------------
# Pages
# -------------------------
if section == "Products":
    st.subheader("Products")

    df = load_csv("products.csv")
    if df.empty:
        st.info("No products loaded yet. Click 'Scrape / Refresh data' in the sidebar.")
    else:
        if "product_id" in df.columns:
            df = df.sort_values("product_id")
        df_unique = df.drop_duplicates(subset=["title"], keep="first")

        show_cols = [c for c in ["title", "price", "url"] if c in df_unique.columns]
        st.caption(f"Showing unique products by title: {len(df_unique)} (from {len(df)} product pages)")
        st.dataframe(df_unique[show_cols], use_container_width=True)

elif section == "Testimonials":
    st.subheader("Testimonials")

    df = load_csv("testimonials.csv")
    if df.empty:
        st.info("No testimonials loaded yet. Click 'Scrape / Refresh data' in the sidebar.")
    else:
        df_display = df.copy()
        df_display.insert(0, "No.", range(1, len(df_display) + 1))
        st.dataframe(df_display, use_container_width=True)

else:
    st.subheader("Reviews (2023)")

    df = load_csv("reviews.csv")
    if df.empty:
        st.info("No reviews loaded yet. Click 'Scrape / Refresh data' in the sidebar.")
        st.stop()

    prod = load_csv("products.csv")
    if not prod.empty and "url" in prod.columns:
        prod["product_id"] = (
            prod["url"]
            .str.extract(r"/product/(\d+)", expand=False)
            .astype(float)
            .astype("Int64")
        )

    df["date"] = pd.to_datetime(df.get("date"), errors="coerce")
    df = df.dropna(subset=["date"])
    df = df[df["date"].dt.year == 2023].copy()

    if not prod.empty and "product_id" in df.columns and "product_id" in prod.columns:
        df = df.merge(
            prod[["product_id", "title"]].dropna(),
            on="product_id",
            how="left",
        )
        df = df.rename(columns={"title": "product_title"})

    if df.empty:
        st.warning("No reviews from 2023 found in reviews.csv.")
        st.stop()

    # Month selector (Jan-Dec 2023)
    months = pd.date_range("2023-01-01", "2023-12-01", freq="MS")
    month_labels = [month_label(m) for m in months]

    selected_label = st.select_slider(
        "Select month (2023)",
        options=month_labels,
        value=month_labels[0],
    )
    selected_month = months[month_labels.index(selected_label)]

    start = selected_month
    end = selected_month + pd.offsets.MonthBegin(1)

    df_m = df[(df["date"] >= start) & (df["date"] < end)].copy()
    st.caption(f"Reviews in {selected_label}: {len(df_m)}")

    if df_m.empty:
        st.info("No reviews in the selected month.")
        st.stop()

    # Controls for sentiment (so it doesn't run on every rerun)
    st.write("### Sentiment Analysis (Hugging Face Transformers)")

    model_name = st.selectbox(
        "Model",
        options=[
            "distilbert-base-uncased-finetuned-sst-2-english",
            # If your assignment allows other HF sentiment models, add them here
        ],
        index=0,
    )
    batch_size = st.slider("Batch size", min_value=1, max_value=32, value=8, step=1)
    max_len = st.slider("Max token length (truncation)", min_value=32, max_value=256, value=128, step=8)

    run_now = st.button("Run sentiment for selected month")

    if "sentiment_cache" not in st.session_state:
        st.session_state["sentiment_cache"] = {}

    cache_key = (selected_label, model_name, batch_size, max_len)

    if run_now:
        with st.spinner("Running sentiment analysis..."):
            sentiments = run_sentiment(
                df_m["text"].astype(str).tolist(),
                model_name=model_name,
                batch_size=batch_size,
                max_length=max_len,
            )
        st.session_state["sentiment_cache"][cache_key] = sentiments

    sentiments = st.session_state["sentiment_cache"].get(cache_key)

    if sentiments is None:
        st.info("Click **Run sentiment for selected month** to compute sentiment.")
        st.stop()

    df_out = df_m.reset_index(drop=True).join(sentiments)

    # Deduplication
    if "product_title" in df_out.columns:
        subset_cols = [c for c in ["product_title", "date", "text", "rating"] if c in df_out.columns]
        if subset_cols:
            df_out = df_out.drop_duplicates(subset=subset_cols)

    # Summary tables/plots
    counts = (
        df_out["sentiment"]
        .value_counts()
        .rename_axis("sentiment")
        .reset_index(name="count")
    )
    avg_conf = (
        df_out.groupby("sentiment")["confidence"]
        .mean()
        .reset_index()
    )

    summary = pd.merge(counts, avg_conf, on="sentiment", how="left")
    summary["confidence"] = summary["confidence"].round(3)

    st.write("### Sentiment summary")
    st.dataframe(summary, use_container_width=True)

    st.write("### Positive vs Negative (count)")
    st.bar_chart(summary.set_index("sentiment")["count"])

    st.write("### Average confidence by sentiment")
    st.bar_chart(summary.set_index("sentiment")["confidence"])

    # Wordcloud
    st.write("### Word Cloud (selected month)")
    text_blob = " ".join(df_out["text"].astype(str).tolist()).strip()

    if not text_blob:
        st.info("No text available for word cloud.")
    else:
        wc = WordCloud(
            width=800,
            height=400,
            background_color="white",
            stopwords=set(STOPWORDS),
        ).generate(text_blob)

        fig, ax = plt.subplots()
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        st.pyplot(fig, use_container_width=True)

    # Reviews table
    st.write("### Reviews")
    show_cols = [
        c for c in ["date", "product_title", "rating", "text", "sentiment", "confidence"]
        if c in df_out.columns
    ]

    df_display = (
        df_out[show_cols]
        .sort_values("date")
        .reset_index(drop=True)
    )
    df_display.insert(0, "No.", range(1, len(df_display) + 1))
    st.dataframe(df_display, use_container_width=True)
