
import os
import json
import time
import gc
import mlflow
import torch
import transformers
import pandas as pd
import matplotlib.pyplot as plt

from transformers import pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# --------------------------------------------------
# 1. CONFIGURATION
# --------------------------------------------------

MODEL_A = "distilbert-base-uncased-finetuned-sst-2-english"
MODEL_B = "textattack/bert-base-uncased-SST-2"

TRACKING_URI = "http://127.0.0.1:5000"

mlflow.set_tracking_uri(TRACKING_URI)
mlflow.set_experiment("sentiment_champion_challenger")

os.makedirs("outputs/model_A", exist_ok=True)
os.makedirs("outputs/model_B", exist_ok=True)
os.makedirs("outputs/comparison", exist_ok=True)

# --------------------------------------------------
# 2. LOAD EVALUATION DATA
# --------------------------------------------------

df = pd.read_csv("data/eval_inputs.csv")

texts = df["text"].astype(str).tolist()
actual = df["label"].tolist()

print("Total evaluation samples:", len(texts))

# --------------------------------------------------
# 3. MODEL CONFIGURATION
# --------------------------------------------------

models = {
    "A": {
        "id": MODEL_A,
        "name": "A_sentiment_distilbert",
        "role": "Candidate"
    },
    "B": {
        "id": MODEL_B,
        "name": "B_sentiment_textattack-bert",
        "role": "Candidate"
    }
}

results = {}

# --------------------------------------------------
# 4. EVALUATE BOTH MODELS
# --------------------------------------------------

for key, config in models.items():

    print("\n" + "=" * 60)
    print("Evaluating:", config["name"])

    classifier = pipeline(
        "sentiment-analysis",
        model=config["id"],
        tokenizer=config["id"],
        device=-1
    )

    # Measure total inference time
    start_time = time.perf_counter()

    predictions = classifier(
        texts,
        batch_size=8,
        truncation=True,
        max_length=256
    )

    total_time = time.perf_counter() - start_time

    # Convert predictions into standard labels
    predicted = []

    for item in predictions:

        label = item["label"]

        if label in ["LABEL_0", "NEGATIVE"]:
            predicted.append(0)
        elif label in ["LABEL_1", "POSITIVE"]:
            predicted.append(1)
        else:
            raise ValueError(f"Unknown model label: {label}")

    # Calculate metrics
    accuracy = accuracy_score(actual, predicted)

    precision = precision_score(
        actual, predicted, average="macro", zero_division=0
    )

    recall = recall_score(
        actual, predicted, average="macro", zero_division=0
    )

    macro_f1 = f1_score(
        actual, predicted, average="macro", zero_division=0
    )

    avg_latency = (total_time / len(texts)) * 1000

    metrics = {
        "accuracy": accuracy,
        "precision_macro": precision,
        "recall_macro": recall,
        "macro_f1": macro_f1,
        "avg_latency_ms": avg_latency,
        "total_inference_s": total_time
    }

    results[key] = {
        "config": config,
        "metrics": metrics,
        "predicted": predicted,
        "confidence": [p["score"] for p in predictions]
    }

    # Save individual predictions
    output_df = df.copy()

    output_df["predicted_label"] = [
        "POSITIVE" if p == 1 else "NEGATIVE"
        for p in predicted
    ]

    output_df["confidence"] = [
        p["score"] for p in predictions
    ]

    output_path = f"outputs/model_{key}/predictions_{key}.csv"
    output_df.to_csv(output_path, index=False)

    # Generate confusion matrix
    cm = confusion_matrix(actual, predicted)

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["NEGATIVE", "POSITIVE"]
    )

    display.plot()
    plt.title(f"Confusion Matrix - Model {key}")
    plt.tight_layout()

    plot_path = f"outputs/model_{key}/confusion_matrix_{key}.png"
    plt.savefig(plot_path)
    plt.close()

    # Log experiment in MLflow
    with mlflow.start_run(run_name=config["name"]):

        mlflow.log_params({
            "hf_model_id": config["id"],
            "task": "sentiment_classification",
            "role": config["role"],
            "max_length": 256,
            "batch_size": 8,
            "eval_samples": len(texts),
            "seed": 42,
            "device": "CPU"
        })

        mlflow.log_metrics(metrics)

        mlflow.set_tags({
            "student_usn": "1MS26CI413-T",
            "dataset_name": "SST-2 validation",
            "transformers_version": transformers.__version__,
            "torch_version": torch.__version__,
            "mlflow_version": mlflow.__version__
        })

        mlflow.log_artifact(output_path)
        mlflow.log_artifact(plot_path)

    print("\nResults:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")

    del classifier
    gc.collect()

# --------------------------------------------------
# 5. SAVE COMPARISON RESULTS
# --------------------------------------------------

summary = {
    "Model A": results["A"]["metrics"],
    "Model B": results["B"]["metrics"]
}

with open("outputs/comparison/metrics_summary.json", "w") as file:
    json.dump(summary, file, indent=4)

# --------------------------------------------------
# 6. GENERATE METRIC COMPARISON CHART
# --------------------------------------------------

metric_names = [
    "accuracy",
    "precision_macro",
    "recall_macro",
    "macro_f1"
]

model_a_values = [
    results["A"]["metrics"][m] for m in metric_names
]

model_b_values = [
    results["B"]["metrics"][m] for m in metric_names
]

x = range(len(metric_names))
width = 0.35

plt.figure(figsize=(10, 5))

plt.bar(
    [i - width / 2 for i in x],
    model_a_values,
    width,
    label="Model A"
)

plt.bar(
    [i + width / 2 for i in x],
    model_b_values,
    width,
    label="Model B"
)

plt.xticks(x, metric_names)
plt.ylabel("Score")
plt.title("Champion vs Challenger Comparison")
plt.legend()
plt.tight_layout()

comparison_path = "outputs/comparison/metric_comparison.png"

plt.savefig(comparison_path)
plt.close()

# --------------------------------------------------
# 7. SAVE FIVE DISAGREEMENT EXAMPLES
# --------------------------------------------------

disagreements = []

for i in range(len(texts)):

    pred_a = results["A"]["predicted"][i]
    pred_b = results["B"]["predicted"][i]

    if pred_a != pred_b:

        disagreements.append({
            "text": texts[i],
            "actual": actual[i],
            "model_A_prediction": pred_a,
            "model_B_prediction": pred_b,
            "model_A_confidence": results["A"]["confidence"][i],
            "model_B_confidence": results["B"]["confidence"][i]
        })

disagreement_df = pd.DataFrame(disagreements[:5])

disagreement_path = "outputs/comparison/disagreement_examples.csv"
disagreement_df.to_csv(disagreement_path, index=False)

# --------------------------------------------------
# 8. LOG COMPARISON RUN IN MLFLOW
# --------------------------------------------------

with mlflow.start_run(run_name="compare_sentiment"):

    mlflow.log_param("primary_metric", "macro_f1")
    mlflow.log_param("minimum_improvement_margin", 0.01)
    mlflow.log_param("latency_guardrail_multiplier", 5)

    mlflow.log_metric(
        "delta_primary_metric",
        results["B"]["metrics"]["macro_f1"]
        - results["A"]["metrics"]["macro_f1"]
    )

    mlflow.log_artifact("outputs/comparison/metrics_summary.json")
    mlflow.log_artifact(comparison_path)
    mlflow.log_artifact(disagreement_path)

# --------------------------------------------------
# 9. FINAL OUTPUT
# --------------------------------------------------

print("\n" + "=" * 60)
print("EVALUATION COMPLETED SUCCESSFULLY!")
print("=" * 60)

print("\nModel A Macro F1:",
      round(results["A"]["metrics"]["macro_f1"], 4))

print("Model B Macro F1:",
      round(results["B"]["metrics"]["macro_f1"], 4))

print("\nAll results saved inside the outputs folder.")
print("Check MLflow at:", TRACKING_URI)
