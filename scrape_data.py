from __future__ import annotations

import time
import re
import json
from pathlib import Path
from urllib.parse import urljoin

import pandas as pd
import requests
from bs4 import BeautifulSoup


BASE_URL = "https://web-scraping.dev"
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0.0.0 Safari/537.36"
)


# -------------------------
# Helpers
# -------------------------
def clean_text(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()


def try_parse_date(s: str) -> str:
    s = clean_text(s)
    if not s:
        return ""
    dt = pd.to_datetime(s, errors="coerce")
    if pd.isna(dt):
        return ""
    return dt.date().isoformat()


def get_session() -> requests.Session:
    sess = requests.Session()
    sess.headers.update({"User-Agent": USER_AGENT})
    return sess


# -------------------------
# PRODUCTS
# -------------------------
def parse_product_detail(sess: requests.Session, url: str) -> dict:
    r = sess.get(url, timeout=30)
    if r.status_code != 200:
        return {"title": "", "price": None}

    soup = BeautifulSoup(r.text, "html.parser")

    title_el = soup.select_one(".product-title")
    title = clean_text(title_el.get_text()) if title_el else ""

    price_el = soup.select_one(".product-price")
    price = None
    if price_el:
        m = re.search(r"(\d+(?:\.\d+)?)", price_el.get_text())
        if m:
            price = float(m.group(1))

    return {"title": title, "price": price}

def scrape_products(sess: requests.Session, max_pages: int = 50, sleep_s: float = 0.2) -> pd.DataFrame:
    rows = []
    seen = set()

    for page in range(1, max_pages + 1):
        r = sess.get(f"{BASE_URL}/products", params={"page": page} if page > 1 else None, timeout=30)
        if r.status_code != 200:
            break

        soup = BeautifulSoup(r.text, "html.parser")

        # collect product urls (absolute or relative)
        urls = []
        for a in soup.select("a[href]"):
            href = a.get("href") or ""
            if re.search(r"/product/\d+\b", href):
                urls.append(urljoin(BASE_URL, href))

        # keep only new urls
        new_urls = [u for u in urls if u not in seen]
        if not new_urls:
            break

        for u in new_urls:
            seen.add(u)
            rows.append({"url": u, "page": page})

        time.sleep(sleep_s)

    df = pd.DataFrame(rows).drop_duplicates(subset=["url"]).reset_index(drop=True)
    if df.empty:
        return df

    # Fetch details for each product (title + price)
    titles, prices = [], []
    for u in df["url"].tolist():
        info = parse_product_detail(sess, u)
        titles.append(info["title"])
        prices: list[float | None]
        prices.append(info["price"])
        time.sleep(0.05)

    df["title"] = titles
    df["price"] = prices

    return df[["title", "url", "price", "page"]]

# -------------------------
# TESTIMONIALS
# -------------------------
def scrape_testimonials(sess: requests.Session, max_pages: int = 200) -> pd.DataFrame:
    rows = []
    api_url = f"{BASE_URL}/api/testimonials"

    headers = {
        "X-Secret-Token": "secret123",
        "Referer": f"{BASE_URL}/testimonials",
        "User-Agent": USER_AGENT,
    }

    for page in range(1, max_pages + 1):
        r = sess.get(api_url, headers=headers, params={"page": page})
        if r.status_code != 200:
            break

        soup = BeautifulSoup(r.text, "html.parser")
        cards = soup.select(".testimonial")
        if not cards:
            break

        for c in cards:
            text = clean_text(c.get_text(" "))
            rating = len(c.select(".rating svg"))
            rows.append(
                {
                    "text": text,
                    "rating": rating if rating > 0 else None,
                    "page": page,
                }
            )

        time.sleep(0.2)

    return pd.DataFrame(rows).drop_duplicates()


# -------------------------
# REVIEWS (CORRECT WAY)
# -------------------------
def get_csrf_from_html(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    inp = soup.select_one('input[name="csrf-token"]')
    return inp.get("value") if inp else ""


def get_embedded_reviews(html: str) -> list[dict]:
    soup = BeautifulSoup(html, "html.parser")
    tag = soup.select_one("script#reviews-data")
    if not tag:
        return []
    try:
        return json.loads(tag.get_text())
    except Exception:
        return []


def scrape_reviews(sess: requests.Session, products_df: pd.DataFrame, max_pages: int = 50) -> pd.DataFrame:
    rows = []

    product_ids = (
        products_df["url"]
        .astype(str)
        .str.extract(r"/product/(\d+)", expand=False)
        .dropna()
        .astype(int)
        .unique()
        .tolist()
    )

    for pid in product_ids:
        product_url = f"{BASE_URL}/product/{pid}"
        r = sess.get(product_url)
        if r.status_code != 200:
            continue

        csrf = get_csrf_from_html(r.text)
        embedded = get_embedded_reviews(r.text)

        # page 1 (embedded JSON)
        for it in embedded:
            rows.append(
                {
                    "product_id": pid,
                    "id": it.get("id"),
                    "date": try_parse_date(it.get("date", "")),
                    "rating": it.get("rating"),
                    "text": clean_text(it.get("text", "")),
                    "page": 1,
                }
            )

        if not csrf:
            continue

        headers = {
            "x-csrf-token": csrf,
            "Referer": product_url,
            "User-Agent": USER_AGENT,
        }

        api_url = f"{BASE_URL}/api/reviews"
        page = 2

        while page <= max_pages:
            rr = sess.get(api_url, headers=headers, params={"product_id": pid, "page": page})
            if rr.status_code != 200:
                break

            data = rr.json()
            results = data.get("results", [])
            if not results:
                break

            for it in results:
                rows.append(
                    {
                        "product_id": pid,
                        "id": it.get("id"),
                        "date": try_parse_date(it.get("date", "")),
                        "rating": it.get("rating"),
                        "text": clean_text(it.get("text", "")),
                        "page": page,
                    }
                )

            if not data.get("next_url"):
                break

            page += 1
            time.sleep(0.2)

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # normalize
    df["id"] = df["id"].fillna("").astype(str)
    df["text"] = df["text"].fillna("").astype(str).str.strip()
    df["date"] = df["date"].fillna("").astype(str)

    # Prefer unique by (product_id, id) when id exists, otherwise fallback key
    df["_key"] = df.apply(
        lambda r: f"{r['product_id']}|ID|{r['id']}" if r["id"] else f"{r['product_id']}|DTXT|{r['date']}|{r['text']}",
        axis=1,
    )

    df = df.drop_duplicates(subset=["_key"]).drop(columns=["_key"])
    return df



# -------------------------
# SAVE ALL
# -------------------------
def scrape_and_save(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    sess = get_session()

    print("Scraping products...")
    products = scrape_products(sess)
    products.to_csv(output_dir / "products.csv", index=False)
    print(f"Saved {len(products)} rows -> products.csv")

    print("Scraping testimonials...")
    testimonials = scrape_testimonials(sess)
    testimonials.to_csv(output_dir / "testimonials.csv", index=False)
    print(f"Saved {len(testimonials)} rows -> testimonials.csv")

    print("Scraping reviews...")
    reviews = scrape_reviews(sess, products)
    reviews.to_csv(output_dir / "reviews.csv", index=False)
    print(f"Saved {len(reviews)} rows -> reviews.csv")

    print("Done.")
    


def main():
    here = Path(__file__).resolve().parent
    data_dir = here / "data"
    scrape_and_save(data_dir)


if __name__ == "__main__":
    main()
