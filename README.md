# Iris Flower Classification — Ernest Arthur

This repository contains my **AI & Machine Learning Task 1: Classification Model on a Standard Dataset**.

## Student
**Ernest Arthur**

## Project
A classification model trained on the standard **Iris flower dataset** using Scikit-learn.

The program:
- loads and prepares the Iris dataset;
- splits the data into 80% training and 20% testing sets;
- standardizes the input features with `StandardScaler`;
- trains a `LogisticRegression` classifier;
- evaluates accuracy, macro precision, macro recall, a classification report, and a confusion matrix;
- saves the dataset, evaluation results, confusion-matrix image, and trained model.

## Results
Using `random_state=42` with stratified train/test splitting:

- **Accuracy:** 93.33%
- **Macro Precision:** 93.33%
- **Macro Recall:** 93.33%
- **Correct predictions:** 28/30

Confusion matrix:

```text
[[10, 0, 0],
 [ 0, 9, 1],
 [ 0, 1, 9]]
```

## How to run

1. Install Python 3.10 or later.
2. Clone or download this repository.
3. Install the dependencies:

```bash
python -m pip install -r requirements.txt
```

4. Run the program:

```bash
python iris_classification.py
```

The program will create:
- `iris_dataset.csv`
- `results.json`
- `evaluation_results.txt`
- `confusion_matrix.png`
- `iris_model.joblib`

## Dataset
The script uses the Iris dataset included with Scikit-learn. A separately hosted version of the standard dataset is also available on Kaggle:
https://www.kaggle.com/datasets/uciml/iris
