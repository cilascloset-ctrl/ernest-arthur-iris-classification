# Task 2 — Sentiment Analysis on Text Data

**Student:** Ernest Arthur
**Dataset:** Twitter US Airline Sentiment (CrowdFlower/Figure Eight; February 2015)
**Goal:** Classify customer tweets as **positive**, **negative**, or **neutral**.

## Assignment requirements covered

1. **Dataset:** Airline customer tweets labeled positive, negative, or neutral.
2. **Preprocessing:** Lowercase; remove URLs, @mentions, punctuation, and duplicate tweets; tokenize and remove English stopwords using TF-IDF vectorizer.
3. **Vectorization:** TF-IDF with unigrams and bigrams, fitted only on training data.
4. **Training:** Scikit-learn LogisticRegression classifier, 80/20 stratified train/test split, fixed seed 42.
5. **Evaluation:** Test accuracy, macro precision/recall/F1, per-class classification report, majority-class baseline, and confusion matrix.

## How to run

From the **repository root**, install dependencies and execute:

    python -m pip install -r Task_2_Sentiment_Analysis/requirements.txt
    python Task_2_Sentiment_Analysis/sentiment_analysis.py

The program downloads Tweets.csv into Task_2_Sentiment_Analysis/data/ on its first run. Internet access is required for the first download. If automatic download fails, download Tweets.csv from the Kaggle dataset linked below and place it in that folder. Use --no-download to insist on a local file.

To use an existing CSV:

    python Task_2_Sentiment_Analysis/sentiment_analysis.py --data path/to/Tweets.csv

The program generates the following under Task_2_Sentiment_Analysis/results/:

- metrics.json — actual dataset sizes, measured accuracy and macro scores.
- classification_report.txt — complete per-class report.
- confusion_matrix.png — classification errors visualized.
- sentiment_model.joblib — saved model; **load only files you trust**.
- example_predictions.txt — sample predictions.

A GitHub Actions workflow also runs the program and uploads these as a downloadable **workflow artifact**.

## Explanation of the process

Each tweet is normalized to lowercase and cleaned using regular expressions to remove URLs, usernames, punctuation, and duplicate text. The dataset is split **before** the TF-IDF model is fitted to prevent test information from leaking into training. The vectorizer tokenizes the remaining text, excludes common English stopwords, and transforms the text into numerical TF-IDF features. A logistic regression model learns how these features correspond to negative, neutral, and positive labels.

The trained model predicts labels for the held-out 20% of tweets. **Test accuracy** is the fraction of correctly predicted labels. The confusion matrix and macro-F1 are also reported because the sentiment classes are imbalanced. For perspective, the script compares the model against a majority-class baseline.

## Accuracy and limitations

**The measured accuracy is written by the actual run** to results/metrics.json and results/classification_report.txt. Do not quote an unverified accuracy; see the GitHub Actions run artifact if you do not run it locally.

**Limitations:**
- The dataset is tweets about **US airlines from 2015**. Performance may not generalize to Ghanaian local languages, today's social-media style, or reviews of unrelated products.
- Tweets include **sarcasm, slang, typos, and mixed opinions**. A simple bag-of-words TF-IDF model may misunderstand these.
- The class distribution is unequal; high accuracy may hide poor **neutral** or **positive** predictions. Check macro-F1 and the confusion matrix, not just accuracy.
- Removing emojis and punctuation can discard important emotional cues.

## Dataset credits and usage

Source: [Kaggle — Twitter US Airline Sentiment](https://www.kaggle.com/datasets/crowdflower/twitter-airline-sentiment); originally from CrowdFlower/Figure Eight. The dataset is shared under **CC BY-NC-SA 4.0**; respect its noncommercial and share-alike terms. This repository contains code and source attribution; the third-party dataset is downloaded on demand instead of being republished here.

Data download mirror used by the script: [ruchitgandhi/Twitter-Airline-Sentiment-Analysis/Tweets.csv](https://github.com/ruchitgandhi/Twitter-Airline-Sentiment-Analysis/blob/master/Tweets.csv).
