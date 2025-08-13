get_ipython().system('pip install pandas')
get_ipython().system('pip install matplotlib')

from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

@dataclass
class Config:
    n: int = 500
    start: str = "2021-01-01"
    outdir: Path = Path("outputs")
    seed: int = 42

# generate random data
def generate_random_data(cfg: Config) -> Dict[str, List]:
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)

    categories = ["Food", "Travel", "Fashion", "Fitness", "Music", "Culture", "Family", "Health"]

    dates = pd.date_range(cfg.start, periods=cfg.n)
    cat_choices = [random.choice(categories) for _ in range(cfg.n)]
    likes = np.random.randint(0, 10_000, size=cfg.n)

    return {
        "Date": dates,
        "Category": cat_choices,
        "Likes": likes,
    }

# load and explore data

def load_dataframe(data: Dict[str, List]) -> pd.DataFrame:
    return pd.DataFrame(data)


def explore_dataframe(df: pd.DataFrame) -> None:
    print("\n=== head() ===")
    print(df.head())

    print("\n=== info() ===")
    import io
    buf = io.StringIO()
    df.info(buf=buf)
    print(buf.getvalue())

    print("\n=== describe() ===")
    print(df.describe(include="all"))

    print("\n=== value_counts(Category) ===")
    print(df["Category"].value_counts())


# Clean the data
def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df_clean = df.dropna().drop_duplicates()
    df_clean["Date"] = pd.to_datetime(df_clean["Date"])
    df_clean["Likes"] = df_clean["Likes"].astype(int)
    return df_clean


# Visualize & Analyze
def visualize(df: pd.DataFrame, outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)

    # Histogram of Likes
    plt.figure()
    plt.hist(df["Likes"], bins=30, edgecolor="black")
    plt.title("Distribution of Likes")
    plt.xlabel("Likes")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(outdir / "likes_histogram.png", dpi=200)

    # Boxplot of Likes by Category
    plt.figure()
    categories = df["Category"].unique()
    data_to_plot = [df[df["Category"] == cat]["Likes"] for cat in categories]
    plt.boxplot(data_to_plot, labels=categories)
    plt.title("Likes by Category")
    plt.xlabel("Category")
    plt.ylabel("Likes")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(outdir / "likes_by_category_boxplot.png", dpi=200)


def analyze(df: pd.DataFrame, outdir: Path) -> None:
    mean_likes = df["Likes"].mean()
    print(f"\n=== Mean Likes (overall): {mean_likes:.2f} ===")

    by_cat = df.groupby("Category")["Likes"].mean().sort_values(ascending=False)
    print("\n=== Mean Likes by Category ===")
    print(by_cat)

    by_cat.to_csv(outdir / "mean_likes_by_category.csv", header=["mean_likes"])


def write_conclusions_template(outdir: Path) -> None:
    text = """# Conclusions (Template)

- Summary of steps: import, generate, explore, clean, visualize, analyze.
- Main observations.
- Which categories tend to perform best and possible reasons.
- Recommendations for strategy.
"""
    (outdir / "CONCLUSIONS.md").write_text(text, encoding="utf-8")


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description="Social Media Random Data Analysis Project")
    parser.add_argument("--n", type=int, default=500)
    parser.add_argument("--start", type=str, default="2021-01-01")
    parser.add_argument("--outdir", type=str, default="outputs")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(args=[])  # important for Jupyter
    return Config(n=args.n, start=args.start, outdir=Path(args.outdir), seed=args.seed)


def main() -> None:
    cfg = parse_args()
    cfg.outdir.mkdir(parents=True, exist_ok=True)

    data = generate_random_data(cfg)
    df = load_dataframe(data)
    explore_dataframe(df)
    df_clean = clean_dataframe(df)
    visualize(df_clean, cfg.outdir)
    analyze(df_clean, cfg.outdir)
    write_conclusions_template(cfg.outdir)
    print(f"\nArtifacts saved to: {cfg.outdir.resolve()}")

# run

cfg = Config(n=500, start="2021-01-01", outdir=Path("outputs"), seed=42)
data = generate_random_data(cfg)
df = load_dataframe(data)
explore_dataframe(df)
df_clean = clean_dataframe(df)
visualize(df_clean, cfg.outdir)
analyze(df_clean, cfg.outdir)
write_conclusions_template(cfg.outdir)
