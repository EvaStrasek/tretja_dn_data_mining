import sys
import subprocess
from pathlib import Path

import pandas as pd
import streamlit as st
from transformers import pipeline

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
def load_csv(name: str) -> pd.DataFrame:
    path = DATA_DIR / name
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def month_label(dt: pd.Timestamp) -> str:
    return dt.strftime("%b %Y")


@st.cache_resource(show_spinner=False)
def get_sentiment_pipeline():
    # Model example requested by assignment
    return pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment-latest")


def run_sentiment(texts: list[str]) -> pd.DataFrame:
    """
    Returns dataframe with columns: label (Positive/Negative), score (0..1)
    """
    nlp = get_sentiment_pipeline()
    outputs = nlp(texts, truncation=True)
    rows = []
    for out in outputs:
        label = out.get("label", "")
        score = float(out.get("score", 0.0))
        # Normalize label names to Positive/Negative
        label_norm = "Positive" if label.upper().startswith("POS") else "Negative"
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
        # Keep unique products by title (what you asked for)
        if "product_id" in df.columns:
            df = df.sort_values("product_id")
        df_unique = df.drop_duplicates(subset=["title"], keep="first")

        show_cols = [c for c in ["title", "price", "url"] if c in df_unique.columns]
        st.caption(f"Showing unique products by title: {len(df_unique)} (from {len(df)} product pages)")
        df_display = df.copy()
        df_display.insert(0, "No.", range(1, len(df_display) + 1))
        st.dataframe(df_unique[show_cols], use_container_width=True)

elif section == "Testimonials":
    st.subheader("Testimonials")

    df = load_csv("testimonials.csv")
    if df.empty:
        st.info("No testimonials loaded yet. Click 'Scrape / Refresh data' in the sidebar.")
    else:
        df_display = df.copy()
        df_display.insert(0, "No.", range(1, len(df_display) + 1))
        st.dataframe(df, use_container_width=True)

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

    # Parse date
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

    # Sentiment analysis
    with st.spinner("Running sentiment analysis..."):
        sentiments = run_sentiment(df_m["text"].astype(str).tolist())

    df_out = df_m.reset_index(drop=True).join(sentiments)

    # >>> DODANO: deduplikacija po product_title + date + text
    if "product_title" in df_out.columns:
        df_out = df_out.drop_duplicates(
            subset=["product_title", "date", "text", "rating"]
        )

    # Visualization: counts + avg confidence
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

    st.write("### Reviews")

    show_cols = [
        c for c in
        ["date", "product_title", "rating", "text", "sentiment", "confidence"]
        if c in df_out.columns
    ]

    df_display = (
        df_out[show_cols]
        .sort_values("date")
        .reset_index(drop=True)
    )
    df_display.insert(0, "No.", range(1, len(df_display) + 1))

    st.dataframe(df_display, use_container_width=True)

