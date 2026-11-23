
import mlflow
import pandas as pd

from mlflow import MlflowClient

# --------------------------------------------------
# 1. CONFIGURATION
# --------------------------------------------------

TRACKING_URI = "http://127.0.0.1:5000"
MODEL_NAME = "Sentiment_Classifier"

mlflow.set_tracking_uri(TRACKING_URI)
mlflow.set_registry_uri(TRACKING_URI)

client = MlflowClient()

# --------------------------------------------------
# 2. GET CHAMPION MODEL INFORMATION
# --------------------------------------------------

model_version = client.get_model_version_by_alias(
    MODEL_NAME,
    "champion"
)

print("Champion Model Version:", model_version.version)

hf_model_id = model_version.tags.get("hf_model_id")

print("Hugging Face Model ID:", hf_model_id)

# --------------------------------------------------
# 3. LOAD MODEL DIRECTLY FROM MLFLOW
# --------------------------------------------------

model_uri = f"models:/{MODEL_NAME}@champion"

print("\nLoading model from:", model_uri)

model = mlflow.pyfunc.load_model(model_uri)

print("Champion model loaded successfully!")

# --------------------------------------------------
# 4. NEW SENTENCES (NOT FROM EVALUATION DATA)
# --------------------------------------------------

new_sentences = [
    "This movie was absolutely fantastic and entertaining.",
    "The story was boring and I did not enjoy it.",
    "The acting was brilliant and the direction was excellent.",
    "I regret wasting my time watching this terrible movie.",
    "A wonderful experience with a beautiful storyline.",
    "The film was disappointing and poorly executed.",
    "I loved every moment of this amazing movie.",
    "The movie was dull and completely uninteresting."
]

new_data = pd.DataFrame({
    "text": new_sentences
})

# --------------------------------------------------
# 5. MAKE PREDICTIONS
# --------------------------------------------------

predictions = model.predict(new_data)

# Combine inputs and predictions
output = pd.concat(
    [new_data, predictions],
    axis=1
)

# --------------------------------------------------
# 6. DISPLAY RESULTS
# --------------------------------------------------

print("\n" + "=" * 60)
print("CHAMPION MODEL PREDICTIONS")
print("=" * 60)

print("Model Version:", model_version.version)
print("Model ID:", hf_model_id)

for index, row in output.iterrows():

    print(f"\nExample {index + 1}")
    print("Text:", row["text"])
    print("Prediction:", row["predicted_label"])
    print("Confidence:", round(row["confidence"], 4))

# --------------------------------------------------
# 7. SAVE OUTPUT
# --------------------------------------------------

output.to_csv(
    "outputs/new_data_predictions.csv",
    index=False
)

print("\nPredictions saved successfully!")
print("File: outputs/new_data_predictions.csv")
