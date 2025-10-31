import os
import json
import re
from collections import Counter
from types import SimpleNamespace
from typing import List, Dict, Any, Optional

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from tqdm import tqdm

import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from pydantic import BaseModel

from project_progress.part_1.data_preparation import ProcessedDocument

# ==============================
# EXPLORATORY ANALYSIS
# ==============================
def parse_numeric(value):
    """Convert values such as '1,499' or '55% off' into clean float values."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    s = str(value)
    s = re.sub(r"[^\d\.]", "", s)
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def normalize_product_details(details):
    """Convert lists of dictionaries into a single merged dictionary."""
    if isinstance(details, dict):
        return details
    if isinstance(details, list):
        merged = {}
        for d in details:
            if isinstance(d, dict):
                merged.update(d)
        return merged
    return {}


def run_exploration(data_path: str, outdir: str, sample_frac: float = 0.2):
    os.makedirs(outdir, exist_ok=True)
    print(f"Loading data from: {data_path}")
    df = pd.read_json(data_path)
    print("Rows loaded:", len(df))

    # Sampling to speed up the process (set 1.0 to use the full dataset)
    if sample_frac < 1.0:
        df = df.sample(frac=sample_frac, random_state=42).reset_index(drop=True)
        print(f"Using a {int(sample_frac * 100)}% sample → {len(df)} rows")

    processed_docs = []
    errors = 0

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing documents"):
        try:
            rdict = row.to_dict()

            # Normalization
            rdict["product_details"] = normalize_product_details(rdict.get("product_details"))
            for field in ["selling_price", "actual_price", "discount", "average_rating"]:
                rdict[field] = parse_numeric(rdict.get(field))

            doc_obj = SimpleNamespace(**rdict)
            pdoc = ProcessedDocument.from_document(doc_obj)
            pdoc.process_fields()
            processed_docs.append(pdoc)

        except Exception:
            errors += 1
            continue

    print(f"Successfully processed: {len(processed_docs)} / {len(df)} (Errors: {errors})")

    # Convert to DataFrame
    data = []
    for d in processed_docs:
        dct = d.model_dump()
        data.append({
            "pid": dct.get("pid"),
            "brand": dct.get("brand"),
            "category": dct.get("category"),
            "seller": dct.get("seller"),
            "out_of_stock": dct.get("out_of_stock"),
            "selling_price": dct.get("selling_price"),
            "discount": dct.get("discount"),
            "actual_price": dct.get("actual_price"),
            "average_rating": dct.get("average_rating"),
            "token_count": len(dct.get("search_text", []))
        })

    df_proc = pd.DataFrame(data)
    print("Final DataFrame:", df_proc.shape)

    # ==========================
    #  ANALYSIS & VISUALS
    # ==========================
    summary = {
        "n_docs": len(df_proc),
        "n_brands": df_proc["brand"].nunique(),
        "n_categories": df_proc["category"].nunique(),
        "avg_price": round(df_proc["selling_price"].mean(), 2),
        "avg_discount": round(df_proc["discount"].mean(), 2),
        "avg_rating": round(df_proc["average_rating"].mean(), 2),
        "out_of_stock_ratio": round(df_proc["out_of_stock"].mean(), 3)
    }

    with open(os.path.join(outdir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    sns.set(style="whitegrid")

    plt.figure(figsize=(8, 5))
    sns.histplot(df_proc["selling_price"].dropna(), bins=40, color="steelblue")
    plt.title("Selling Price Distribution")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "price_distribution.png"))
    plt.close()

    plt.figure(figsize=(8, 5))
    sns.histplot(df_proc["average_rating"].dropna(), bins=20, color="darkorange")
    plt.title("Average Rating Distribution")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "rating_distribution.png"))
    plt.close()

    plt.figure(figsize=(9, 5))
    top_brands = df_proc["brand"].value_counts().head(15)
    sns.barplot(x=top_brands.values, y=top_brands.index, palette="Blues_r")
    plt.title("Top 15 Brands")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "top_brands.png"))
    plt.close()

    plt.figure(figsize=(6, 6))
    stock_counts = df_proc["out_of_stock"].value_counts()
    stock_counts.index = ["In Stock", "Out of Stock"]
    stock_counts.plot(kind="pie", autopct="%1.1f%%", colors=["#8fd9b6", "#f6a6a6"])
    plt.title("Stock Availability")
    plt.ylabel("")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "stock_pie.png"))
    plt.close()

    all_tokens = []
    for d in processed_docs:
        all_tokens += d.search_text or []
    vocab = Counter(all_tokens)
    wc = WordCloud(width=1000, height=500, background_color="white").generate_from_frequencies(vocab)
    plt.figure(figsize=(12, 6))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.title("Most Frequent Words (WordCloud)")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "wordcloud.png"))
    plt.close()

    print("EDA completed. Results saved in:", outdir)


if __name__ == "__main__":
    # Simple configuration without argparse
    data_path = "data/fashion_products_dataset.json"
    outdir = "outputs"
    run_exploration(data_path, outdir, sample_frac=0.2)
