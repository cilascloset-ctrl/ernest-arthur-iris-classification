"""Task 2: three-class sentiment analysis of airline tweets.

Author: Ernest Arthur
Source data: CrowdFlower/Figure Eight Twitter US Airline Sentiment (2015).
Run: python Task_2_Sentiment_Analysis/sentiment_analysis.py
"""
from __future__ import annotations

import argparse
import json
import re
import urllib.error
import urllib.request
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

HERE = Path(__file__).resolve().parent
DATA_URL = (
    "https://raw.githubusercontent.com/ruchitgandhi/"
    "Twitter-Airline-Sentiment-Analysis/master/Tweets.csv"
)
LABELS = ["negative", "neutral", "positive"]
SEED = 42


def clean_text(value: str) -> str:
    """Remove links, user handles, punctuation and extra spaces; keep words."""
    text = str(value).lower()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"@\w+", " ", text)
    text = re.sub(r"[^a-z\s']", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def load_dataset(path: Path, download: bool = True) -> pd.DataFrame:
    """Load the real labeled dataset; download it once if it is missing."""
    if not path.exists():
        if not download:
            raise FileNotFoundError(f"Dataset not found: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading dataset from {DATA_URL}")
        try:
            urllib.request.urlretrieve(DATA_URL, path)
        except (OSError, urllib.error.URLError) as exc:
            raise RuntimeError(
                "Dataset download failed. Download Tweets.csv from "
                "https://www.kaggle.com/datasets/crowdflower/twitter-airline-sentiment "
                f"and save it at {path}."
            ) from exc

    df = pd.read_csv(path, encoding="utf-8", encoding_errors="replace")
    required = {"text", "airline_sentiment"}
    if not required.issubset(df.columns):
        raise ValueError(
            f"Missing required columns {sorted(required - set(df.columns))}. "
            "Expected the US Airline Sentiment Tweets.csv file."
        )
    df = df[["text", "airline_sentiment"]].dropna().copy()
    df["airline_sentiment"] = df["airline_sentiment"].str.lower().str.strip()
    df = df[df["airline_sentiment"].isin(LABELS)]
    df["cleaned_text"] = df["text"].map(clean_text)
    df = df[df["cleaned_text"].str.len().gt(0)]
    # Avoid identical cleaned tweets appearing in both train and test sets.
    df = df.drop_duplicates(subset=["cleaned_text"]).reset_index(drop=True)
    if df["airline_sentiment"].nunique() != 3:
        raise ValueError("Dataset must contain negative, neutral, and positive labels.")
    if df["airline_sentiment"].value_counts().min() < 3:
        raise ValueError("Each class needs at least three examples for evaluation.")
    return df


def train_and_evaluate(df: pd.DataFrame, output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    x_train, x_test, y_train, y_test = train_test_split(
        df["cleaned_text"],
        df["airline_sentiment"],
        test_size=0.20,
        random_state=SEED,
        stratify=df["airline_sentiment"],
    )
    # The vectorizer learns its vocabulary on training data only: no leakage.
    model = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    stop_words="english",  # English stopword removal
                    token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z']+\b",  # tokenization
                    ngram_range=(1, 2),
                    min_df=2,
                    sublinear_tf=True,
                ),
            ),
            ("classifier", LogisticRegression(max_iter=1000, random_state=SEED)),
        ]
    )
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    baseline = DummyClassifier(strategy="most_frequent")
    baseline.fit(x_train.to_frame(), y_train)
    baseline_accuracy = accuracy_score(y_test, baseline.predict(x_test.to_frame()))

    result = {
        "author": "Ernest Arthur",
        "task": "Task 2 - Sentiment Analysis on Text Data",
        "dataset": "Twitter US Airline Sentiment (CrowdFlower/Figure Eight)",
        "random_state": SEED,
        "data_rows_after_cleaning": int(len(df)),
        "train_rows": int(len(x_train)),
        "test_rows": int(len(x_test)),
        "class_counts": {k: int(v) for k, v in df["airline_sentiment"].value_counts().sort_index().items()},
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "majority_class_baseline_accuracy": float(baseline_accuracy),
        "macro_precision": float(precision_score(y_test, y_pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_test, y_pred, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_test, y_pred, labels=LABELS).tolist(),
        "labels_order": LABELS,
    }
    (output_dir / "metrics.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    (output_dir / "classification_report.txt").write_text(
        f"Task 2: Sentiment Analysis\nAuthor: Ernest Arthur\n"
        f"Rows: {len(df)} | Train: {len(x_train)} | Test: {len(x_test)}\n"
        f"Accuracy: {result['accuracy']:.4f} ({result['accuracy']:.2%})\n"
        f"Majority baseline: {baseline_accuracy:.4f} ({baseline_accuracy:.2%})\n"
        f"Macro precision: {result['macro_precision']:.4f}\n"
        f"Macro recall: {result['macro_recall']:.4f}\n"
        f"Macro F1: {result['macro_f1']:.4f}\n\n"
        + classification_report(y_test, y_pred, labels=LABELS, zero_division=0)
        + "\nConfusion matrix (negative, neutral, positive):\n"
        + str(result["confusion_matrix"])
        + "\n",
        encoding="utf-8",
    )
    ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred, labels=LABELS, cmap="Blues", colorbar=False
    )
    plt.title("Sentiment classification - test set")
    plt.tight_layout()
    plt.savefig(output_dir / "confusion_matrix.png", dpi=160)
    plt.close()
    joblib.dump(model, output_dir / "sentiment_model.joblib")

    samples = [
        "The flight was smooth and the staff were friendly.",
        "My flight is delayed again and customer service is terrible.",
        "My flight leaves at 6 pm from the main terminal.",
    ]
    predictions = model.predict([clean_text(s) for s in samples])
    (output_dir / "example_predictions.txt").write_text(
        "\n".join(f"{label.upper()}: {sentence}" for sentence, label in zip(samples, predictions))
        + "\n",
        encoding="utf-8",
    )
    print((output_dir / "classification_report.txt").read_text(encoding="utf-8"))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Task 2: three-class tweet sentiment classifier")
    parser.add_argument("--data", type=Path, default=HERE / "data" / "Tweets.csv")
    parser.add_argument("--output-dir", type=Path, default=HERE / "results")
    parser.add_argument("--no-download", action="store_true", help="Require a local CSV")
    args = parser.parse_args()
    df = load_dataset(args.data, download=not args.no_download)
    train_and_evaluate(df, args.output_dir)


if __name__ == "__main__":
    main()
