# Task 2 - Evaluation Report

**Student:** Ernest Arthur  
**Assignment:** Sentiment Analysis on Text Data  
**Dataset:** Twitter US Airline Sentiment, CrowdFlower/Figure Eight (tweets from 2015)

## Aim and method

The objective is to classify customer reviews/tweets as **negative**, **neutral**, or **positive**. The project uses Python, Pandas, Scikit-learn and Matplotlib.

1. Read the labeled Twitter US Airline Sentiment dataset with the `text` and `airline_sentiment` columns.
2. Convert text to lowercase, remove URLs, user mentions, punctuation and duplicated cleaned tweets.
3. Split the cleaned data into **80% training / 20% testing**, with stratification and random state 42.
4. Tokenize and remove English stopwords; convert text to numeric **TF-IDF unigrams and bigrams** using only training-data vocabulary.
5. Fit **Logistic Regression** and evaluate on unseen test samples.

## Measured results

These numbers are **actual results**, reproduced successfully in [GitHub Actions run #1](https://github.com/cilascloset-ctrl/ernest-arthur-iris-classification/actions/runs/38750102106), not estimated scores.

| Metric | Value |
|---|---:|
| Cleaned, nonduplicate samples | 14,171 |
| Training samples | 11,336 |
| Testing samples | 2,835 |
| **Test accuracy** | **76.72%** |
| Majority-class baseline accuracy | 63.92% |
| Macro precision | 74.35% |
| Macro recall | 62.35% |
| Macro F1-score | 66.02% |

### Per-class classification

| Class | Precision | Recall | F1 | Test samples |
|---|---:|---:|---:|---:|
| Negative | 0.78 | 0.95 | 0.86 | 1,812 |
| Neutral | 0.64 | 0.36 | 0.46 | 586 |
| Positive | 0.81 | 0.56 | 0.66 | 437 |

### Confusion matrix

Rows are actual classes; columns are predicted classes. The order is **negative, neutral, positive**.

| Actual / predicted | Negative | Neutral | Positive |
|---|---:|---:|---:|
| Negative | 1,718 | 60 | 34 |
| Neutral | 349 | 212 | 25 |
| Positive | 135 | 57 | 245 |

The model performs better than a majority-class baseline by about **12.8 percentage points**. However, it frequently predicts neutral or positive comments as negative. This is why examining per-class recall and macro F1 is important in addition to accuracy.

## Limitations and improvements

- **Imbalanced dataset:** The negative class is much more common. A model can show reasonable accuracy while missing many neutral tweets; neutral recall was 0.36.
- **Limited domain and age:** Tweets are about US airline experiences from 2015. Predictions may be poorer on product reviews, Ghanaian English, local languages or modern writing styles.
- **Difficult language:** Sarcasm, slang, misspellings, mixed opinions, emojis and context are difficult for basic TF-IDF features. The cleaning step discards emoji and punctuation clues.
- **Model simplicity:** Logistic Regression considers weighted word patterns but not deep contextual meaning. Future work could compare class-weighted models, Naive Bayes, linear SVM or transformers on the same split.

## Reproduction and deliverables

From repository root:

    python -m pip install -r Task_2_Sentiment_Analysis/requirements.txt
    python Task_2_Sentiment_Analysis/sentiment_analysis.py

A fresh run writes `results/metrics.json`, `results/classification_report.txt`, `results/confusion_matrix.png`, example predictions, and a trained model. The [successful GitHub Actions run](https://github.com/cilascloset-ctrl/ernest-arthur-iris-classification/actions/runs/38750102106) also contains these five files in its downloadable **task2-sentiment-results** artifact.

## Source attribution

CrowdFlower/Figure Eight, *Twitter US Airline Sentiment*, [Kaggle](https://www.kaggle.com/datasets/crowdflower/twitter-airline-sentiment), CC BY-NC-SA 4.0. The dataset is downloaded separately, and is not redistributed in this repository.
