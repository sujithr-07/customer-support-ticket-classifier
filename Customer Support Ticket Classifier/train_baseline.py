
import os
import joblib
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# Create output directories
os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)

# Load datasets
train_df = pd.read_csv("train.csv")
val_df = pd.read_csv("validation.csv")

X_train = train_df["text"].fillna("")
X_val = val_df["text"].fillna("")

# Labels
targets = {
    "category": train_df["category"],
    "priority": train_df["priority"]
}

val_targets = {
    "category": val_df["category"],
    "priority": val_df["priority"]
}

# TF-IDF + Logistic Regression pipeline
pipeline = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("classifier", LogisticRegression(
        max_iter=2000,
        random_state=42
    ))
])

# Hyperparameter tuning
param_grid = {
    "tfidf__ngram_range": [(1, 1), (1, 2), (1, 3)],
    "classifier__C": [0.1, 1, 10]
}

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

results = {}

for label_name, y_train in targets.items():

    print("\n" + "=" * 60)
    print(f"TRAINING {label_name.upper()} CLASSIFIER")
    print("=" * 60)

    search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring="f1_macro",
        cv=cv,
        n_jobs=1,
        verbose=1
    )

    # Train and tune
    search.fit(X_train, y_train)

    best_model = search.best_estimator_

    # Validation predictions
    predictions = best_model.predict(X_val)

    # Evaluation
    accuracy = accuracy_score(
        val_targets[label_name],
        predictions
    )

    macro_f1 = f1_score(
        val_targets[label_name],
        predictions,
        average="macro"
    )

    print("\nBest Parameters:", search.best_params_)
    print("Best CV Macro F1:", round(search.best_score_, 4))

    print("\nValidation Results")
    print("Accuracy:", round(accuracy, 4))
    print("Macro F1:", round(macro_f1, 4))

    print("\nClassification Report:")
    print(classification_report(
        val_targets[label_name],
        predictions,
        zero_division=0
    ))

    # Confusion matrix
    labels = sorted(val_targets[label_name].unique())

    cm = confusion_matrix(
        val_targets[label_name],
        predictions,
        labels=labels
    )

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels
    )

    plt.title(f"{label_name.capitalize()} Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("Actual Label")
    plt.tight_layout()

    plt.savefig(
        f"results/{label_name}_confusion_matrix.png"
    )
    plt.close()

    # Identify the 10 most confusing examples
    probabilities = best_model.predict_proba(X_val)

    classes = list(best_model.classes_)

    true_labels = val_targets[label_name].tolist()

    true_probabilities = [
        probabilities[i][classes.index(true_labels[i])]
        for i in range(len(true_labels))
    ]

    confusing_df = pd.DataFrame({
        "text": X_val.tolist(),
        "actual": true_labels,
        "predicted": predictions,
        "actual_class_probability": true_probabilities,
        "correct": [
            true_labels[i] == predictions[i]
            for i in range(len(true_labels))
        ]
    })

    # Lowest probability assigned to the actual class
    confusing_df = confusing_df.sort_values(
        "actual_class_probability",
        ascending=True
    ).head(10)

    confusing_df.to_csv(
        f"results/{label_name}_confusing_examples.csv",
        index=False
    )

    print("\n10 Most Confusing Examples:")
    print(confusing_df.to_string(index=False))

    # Save trained model
    joblib.dump(
        best_model,
        f"models/{label_name}_classifier.joblib"
    )

    results[label_name] = {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "best_params": search.best_params_
    }

# Final summary
print("\n" + "=" * 60)
print("FINAL VALIDATION SUMMARY")
print("=" * 60)

for label, metrics in results.items():
    print(
        f"{label.capitalize()}: "
        f"Accuracy = {metrics['accuracy']:.4f}, "
        f"Macro F1 = {metrics['macro_f1']:.4f}"
    )

# Save metrics
pd.DataFrame(results).T.to_csv(
    "results/validation_metrics.csv"
)

print("\nModels and evaluation results saved successfully.")
