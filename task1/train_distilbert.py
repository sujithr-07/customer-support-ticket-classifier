
import os
import time
import joblib
import numpy as np
import pandas as pd
import torch

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding
)

from sklearn.metrics import accuracy_score, f1_score


# -------------------------------------------------
# 1. Configuration
# -------------------------------------------------

MODEL_NAME = "distilbert-base-uncased"

os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"

print("Using device:", device)


# -------------------------------------------------
# 2. Load datasets
# -------------------------------------------------

train_df = pd.read_csv("train.csv")
val_df = pd.read_csv("validation.csv")
test_df = pd.read_csv("test.csv")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


# -------------------------------------------------
# 3. Metrics
# -------------------------------------------------

def compute_metrics(eval_pred):

    logits, labels = eval_pred

    predictions = np.argmax(logits, axis=-1)

    return {
        "accuracy": accuracy_score(labels, predictions),
        "f1": f1_score(
            labels,
            predictions,
            average="macro",
            zero_division=0
        )
    }


# -------------------------------------------------
# 4. Inference timing
# -------------------------------------------------

def measure_latency(predict_function, texts):

    # Warm-up
    for text in texts[:min(5, len(texts))]:
        predict_function(text)

    start = time.perf_counter()

    for text in texts:
        predict_function(text)

    elapsed = time.perf_counter() - start

    return (elapsed / len(texts)) * 1000


# -------------------------------------------------
# 5. Train both classifiers
# -------------------------------------------------

all_results = []

for target in ["category", "priority"]:

    print("\n" + "=" * 60)
    print("TRAINING DISTILBERT:", target.upper())
    print("=" * 60)

    # Encode labels
    labels = sorted(train_df[target].unique())

    label2id = {
        label: index
        for index, label in enumerate(labels)
    }

    id2label = {
        index: label
        for label, index in label2id.items()
    }

    y_train = train_df[target].map(label2id)
    y_val = val_df[target].map(label2id)
    y_test = test_df[target].map(label2id)

    # Check that all validation/test labels exist in training
    assert val_df[target].isin(labels).all()
    assert test_df[target].isin(labels).all()

    # Prepare Hugging Face datasets
    def prepare_dataset(texts, target_labels):

        dataset = Dataset.from_dict({
            "text": texts.tolist(),
            "labels": target_labels.tolist()
        })

        def tokenize(batch):
            return tokenizer(
                batch["text"],
                truncation=True,
                max_length=128
            )

        return dataset.map(
            tokenize,
            batched=True,
            remove_columns=["text"]
        )

    train_dataset = prepare_dataset(
        train_df["text"],
        y_train
    )

    val_dataset = prepare_dataset(
        val_df["text"],
        y_val
    )

    test_dataset = prepare_dataset(
        test_df["text"],
        y_test
    )

    # Load pretrained DistilBERT
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(labels),
        id2label=id2label,
        label2id=label2id
    )

    # Training arguments
    training_args = TrainingArguments(
        output_dir=f"results/distilbert_{target}",
        num_train_epochs=3,
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        logging_steps=10,
        save_total_limit=1,
        report_to="none",
        fp16=torch.cuda.is_available(),
        seed=42
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(
            tokenizer=tokenizer
        ),
        compute_metrics=compute_metrics
    )

    # Train and measure time
    start_time = time.perf_counter()

    trainer.train()

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    training_time = time.perf_counter() - start_time

    print("Training time:", round(training_time, 2), "seconds")

    # Save model
    model_path = f"models/distilbert_{target}"

    trainer.save_model(model_path)
    tokenizer.save_pretrained(model_path)

    # Prediction function for single-ticket latency
    model.eval()
    model.to(device)

    def predict_one(text):

        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=128
        )

        inputs = {
            key: value.to(device)
            for key, value in inputs.items()
        }

        with torch.inference_mode():
            outputs = model(**inputs)

        return outputs.logits.argmax(dim=-1).item()

    latency = measure_latency(
        predict_one,
        val_df["text"].tolist()
    )

    print("Inference latency:", round(latency, 3), "ms/ticket")

    # Evaluate validation and test
    for split_name, dataset, original_df in [
        ("Validation", val_dataset, val_df),
        ("Test", test_dataset, test_df)
    ]:

        predictions = trainer.predict(dataset)

        predicted_labels = np.argmax(
            predictions.predictions,
            axis=-1
        )

        actual_labels = original_df[target].map(
            label2id
        ).to_numpy()

        accuracy = accuracy_score(
            actual_labels,
            predicted_labels
        )

        macro_f1 = f1_score(
            actual_labels,
            predicted_labels,
            average="macro",
            zero_division=0
        )

        print(f"\n{target} - {split_name}")
        print("Accuracy:", round(accuracy, 4))
        print("Macro F1:", round(macro_f1, 4))

        all_results.append({
            "Task": target,
            "Model": "DistilBERT",
            "Split": split_name,
            "Accuracy": accuracy,
            "Macro F1": macro_f1,
            "Training time (seconds)": training_time,
            "Inference time (ms/ticket)": latency
        })


# -------------------------------------------------
# 6. Evaluate Logistic Regression baseline
# -------------------------------------------------

for target in ["category", "priority"]:

    print("\nEvaluating baseline:", target)

    baseline = joblib.load(
        f"models/{target}_classifier.joblib"
    )

    for split_name, dataframe in [
        ("Validation", val_df),
        ("Test", test_df)
    ]:

        texts = dataframe["text"].fillna("")
        actual = dataframe[target]

        start = time.perf_counter()

        predicted = baseline.predict(texts)

        elapsed = time.perf_counter() - start

        latency = (elapsed / len(texts)) * 1000

        accuracy = accuracy_score(actual, predicted)

        macro_f1 = f1_score(
            actual,
            predicted,
            average="macro",
            zero_division=0
        )

        all_results.append({
            "Task": target,
            "Model": "TF-IDF + Logistic Regression",
            "Split": split_name,
            "Accuracy": accuracy,
            "Macro F1": macro_f1,
            "Training time (seconds)": np.nan,
            "Inference time (ms/ticket)": latency
        })


# -------------------------------------------------
# 7. Save comparison table
# -------------------------------------------------

comparison = pd.DataFrame(all_results)

comparison = comparison.sort_values(
    ["Task", "Split", "Model"]
)

comparison.to_csv(
    "results/model_comparison.csv",
    index=False
)

print("\n" + "=" * 70)
print("FINAL MODEL COMPARISON")
print("=" * 70)

print(comparison.to_string(index=False))

print("\nComparison saved to results/model_comparison.csv")
