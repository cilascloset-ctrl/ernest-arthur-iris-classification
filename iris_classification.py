"""Task 1 - Iris flower classification | Ernest Arthur.
Run: python iris_classification.py
"""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
import joblib

OUT = Path(__file__).resolve().parent

# Load the standard Iris flower dataset.
iris = load_iris(as_frame=True)
X, y = iris.data, iris.target
names = list(iris.target_names)

# Save a readable CSV copy of the dataset.
dataset = X.copy()
dataset['species'] = y.map(dict(enumerate(names)))
dataset.to_csv(OUT / 'iris_dataset.csv', index=False)

# Split the data into 80% training and 20% testing sets.
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

# The scaler is fitted only on the training data through the pipeline,
# which helps prevent test-data leakage.
model = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', LogisticRegression(max_iter=1000, random_state=42)),
])

# Train the model.
model.fit(X_train, y_train)

# Make predictions on the test set.
predictions = model.predict(X_test)

# Evaluate the model.
results = {
    'author': 'Ernest Arthur',
    'dataset': 'Iris',
    'total_samples': len(X),
    'training_samples': len(X_train),
    'testing_samples': len(X_test),
    'class_names': names,
    'accuracy': accuracy_score(y_test, predictions),
    'precision_macro': precision_score(
        y_test, predictions, average='macro', zero_division=0
    ),
    'recall_macro': recall_score(
        y_test, predictions, average='macro', zero_division=0
    ),
    'confusion_matrix': confusion_matrix(y_test, predictions).tolist(),
    'classification_report': classification_report(
        y_test,
        predictions,
        target_names=names,
        zero_division=0,
    ),
}

# Save machine-readable results.
(OUT / 'results.json').write_text(
    json.dumps(results, indent=2),
    encoding='utf-8',
)

# Save a human-readable evaluation report.
(OUT / 'evaluation_results.txt').write_text(
    f"ERNEST ARTHUR | TASK 1: IRIS CLASSIFICATION\n\n"
    f"Dataset size: {len(X)} | Training: {len(X_train)} | Testing: {len(X_test)}\n"
    f"Accuracy: {results['accuracy']:.4f}\n"
    f"Macro precision: {results['precision_macro']:.4f}\n"
    f"Macro recall: {results['recall_macro']:.4f}\n\n"
    f"Classification report:\n{results['classification_report']}\n"
    f"Confusion matrix (rows=true, cols=predicted):\n"
    f"{confusion_matrix(y_test, predictions)}\n",
    encoding='utf-8',
)

# Create and save the confusion matrix chart.
ConfusionMatrixDisplay.from_predictions(
    y_test,
    predictions,
    display_labels=names,
    cmap='Blues',
    values_format='d',
)
plt.title('Iris Classification - Confusion Matrix')
plt.tight_layout()
plt.savefig(OUT / 'confusion_matrix.png', dpi=180)
plt.close()

# Save the trained model.
joblib.dump(model, OUT / 'iris_model.joblib')

# Print the evaluation results.
print((OUT / 'evaluation_results.txt').read_text(encoding='utf-8'))
