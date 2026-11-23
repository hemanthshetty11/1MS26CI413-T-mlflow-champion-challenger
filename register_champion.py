
import json
import mlflow
import pandas as pd

from mlflow import MlflowClient
from transformers import pipeline
import torch

# --------------------------------------------------
# 1. CONFIGURATION
# --------------------------------------------------

TRACKING_URI = "http://127.0.0.1:5000"
REGISTERED_MODEL_NAME = "Sentiment_Classifier"

mlflow.set_tracking_uri(TRACKING_URI)
mlflow.set_registry_uri(TRACKING_URI)

client = MlflowClient()

# Read actual evaluation results
with open("outputs/comparison/metrics_summary.json", encoding="utf-8-sig") as f:
    results = json.load(f)

# Apply the predefined Champion rule
a = results["Model A"]
b = results["Model B"]

f1_improvement = b["macro_f1"] - a["macro_f1"]
latency_ratio = b["avg_latency_ms"] / a["avg_latency_ms"]

if f1_improvement >= 0.01 and latency_ratio <= 5:
    winner = "B"
else:
    winner = "A"

print("Selected Champion:", winner)

# --------------------------------------------------
# 2. PYFUNC MODEL WRAPPER
# --------------------------------------------------

class SentimentModel(mlflow.pyfunc.PythonModel):

    def __init__(self, model_id):
        self.model_id = model_id

    def load_context(self, context):

        self.classifier = pipeline(
            "sentiment-analysis",
            model=self.model_id,
            tokenizer=self.model_id,
            device=-1
        )

    def predict(self, context, model_input, params=None):

        texts = model_input["text"].astype(str).tolist()

        predictions = self.classifier(
            texts,
            truncation=True,
            max_length=256,
            batch_size=8
        )

        output = []

        for item in predictions:

            label = item["label"]

            if label in ["LABEL_0", "NEGATIVE"]:
                label = "NEGATIVE"
            else:
                label = "POSITIVE"

            output.append({
                "predicted_label": label,
                "confidence": item["score"]
            })

        return pd.DataFrame(output)


# --------------------------------------------------
# 3. REGISTER BOTH MODELS
# --------------------------------------------------

models = {
    "A": {
        "id": "distilbert-base-uncased-finetuned-sst-2-english",
        "role": "Champion" if winner == "A" else "Challenger"
    },
    "B": {
        "id": "textattack/bert-base-uncased-SST-2",
        "role": "Champion" if winner == "B" else "Challenger"
    }
}

versions = {}

for key, config in models.items():

    print("\nRegistering Model", key)

    with mlflow.start_run(run_name=f"register_model_{key}"):

        model_info = mlflow.pyfunc.log_model(
            name=f"sentiment_model_{key}",
            python_model=SentimentModel(config["id"]),
            registered_model_name=REGISTERED_MODEL_NAME,
            pip_requirements=[
                "mlflow==3.18.0",
                "transformers",
                "torch",
                "pandas"
            ],
            metadata={
                "hf_model_id": config["id"],
                "role": config["role"],
                "task": "sentiment_classification"
            }
        )

    version = str(model_info.registered_model_version)
    versions[key] = version

    # Add model-version tags
    client.set_model_version_tag(
        REGISTERED_MODEL_NAME,
        version,
        "hf_model_id",
        config["id"]
    )

    client.set_model_version_tag(
        REGISTERED_MODEL_NAME,
        version,
        "role",
        config["role"]
    )

    print("Model", key, "registered as version", version)


# --------------------------------------------------
# 4. ASSIGN MODEL ALIASES
# --------------------------------------------------

champion_key = winner
challenger_key = "A" if winner == "B" else "B"

client.set_registered_model_alias(
    REGISTERED_MODEL_NAME,
    "champion",
    versions[champion_key]
)

client.set_registered_model_alias(
    REGISTERED_MODEL_NAME,
    "challenger",
    versions[challenger_key]
)

# --------------------------------------------------
# 5. DISPLAY FINAL RESULT
# --------------------------------------------------

print("\n" + "=" * 50)
print("MODEL REGISTRATION COMPLETED")
print("=" * 50)

print("Registered model:", REGISTERED_MODEL_NAME)
print("Champion version:", versions[champion_key])
print("Challenger version:", versions[challenger_key])

print("\nChampion HF ID:", models[champion_key]["id"])
print("Challenger HF ID:", models[challenger_key]["id"])

print("\nChampion URI:")
print(f"models:/{REGISTERED_MODEL_NAME}@champion")

print("\nChallenger URI:")
print(f"models:/{REGISTERED_MODEL_NAME}@challenger")
