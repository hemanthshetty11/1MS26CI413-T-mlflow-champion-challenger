import json
import os
import mlflow
from mlflow.tracking import MlflowClient

# Connect to MLflow
TRACKING_URI = "http://127.0.0.1:5000"

mlflow.set_tracking_uri(TRACKING_URI)

client = MlflowClient(tracking_uri=TRACKING_URI)

EXPERIMENT_NAME = "sentiment_champion_challenger"

experiment = client.get_experiment_by_name(EXPERIMENT_NAME)

if experiment is None:
    raise Exception("MLflow experiment not found!")

mlflow.set_experiment(EXPERIMENT_NAME)

# Load actual evaluation results
with open(
    "outputs/comparison/metrics_summary.json",
    encoding="utf-8-sig"
) as f:
    results = json.load(f)

model_a = results["Model A"]
model_b = results["Model B"]


# Function to avoid creating duplicate runs
def run_exists(run_name):
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=f"tags.mlflow.runName = '{run_name}'"
    )
    return len(runs) > 0


# -------------------------------------------------
# 1. LOG MODEL B EVALUATION
# -------------------------------------------------

run_name = "B_sentiment_textattack_bert"

if not run_exists(run_name):

    with mlflow.start_run(run_name=run_name):

        mlflow.log_params({
            "model_name": "textattack/bert-base-uncased-SST-2",
            "model_role": "Candidate",
            "dataset": "SST-2 validation",
            "eval_samples": 200,
            "seed": 42,
            "device": "CPU"
        })

        mlflow.log_metrics(model_b)

        mlflow.set_tags({
            "student_usn": "1MS26CI413-T",
            "role": "Candidate",
            "source": "complete_mlflow_runs.py"
        })

        mlflow.log_artifact(
            "outputs/model_B/predictions_B.csv"
        )

        mlflow.log_artifact(
            "outputs/model_B/confusion_matrix_B.png"
        )

    print("Model B evaluation run logged successfully!")

else:
    print("Model B evaluation run already exists. Skipping.")


# -------------------------------------------------
# 2. LOG MODEL COMPARISON
# -------------------------------------------------

run_name = "Champion_Challenger_Comparison"

if not run_exists(run_name):

    with mlflow.start_run(run_name=run_name):

        # Log comparison metrics
        comparison_metrics = {}

        for metric in model_a:
            comparison_metrics[f"Model_A_{metric}"] = model_a[metric]
            comparison_metrics[f"Model_B_{metric}"] = model_b[metric]

        mlflow.log_metrics(comparison_metrics)

        # Log decision rule
        mlflow.log_params({
            "primary_metric": "macro_f1",
            "minimum_f1_improvement": 0.01,
            "latency_limit_multiplier": 5,
            "selected_champion": "Model B",
            "selected_challenger": "Model A",
            "f1_improvement": round(
                model_b["macro_f1"] - model_a["macro_f1"], 6
            ),
            "latency_ratio": round(
                model_b["avg_latency_ms"] /
                model_a["avg_latency_ms"], 4
            )
        })

        mlflow.set_tags({
            "student_usn": "1MS26CI413-T",
            "comparison_type": "Champion-Challenger",
            "winner": "Model B",
            "source": "complete_mlflow_runs.py"
        })

        # Log comparison artifacts
        artifacts = [
            "outputs/comparison/metrics_summary.json",
            "outputs/comparison/metric_comparison.png",
            "outputs/comparison/disagreement_examples.csv",
            "outputs/model_A/predictions_A.csv",
            "outputs/model_A/confusion_matrix_A.png",
            "outputs/model_B/predictions_B.csv",
            "outputs/model_B/confusion_matrix_B.png",
            "data/eval_inputs.csv"
        ]

        for artifact in artifacts:
            if os.path.exists(artifact):
                mlflow.log_artifact(artifact)

    print("Comparison run logged successfully!")

else:
    print("Comparison run already exists. Skipping.")


print("\nMLflow recovery completed!")