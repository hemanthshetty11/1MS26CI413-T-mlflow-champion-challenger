
import pandas as pd
from transformers import pipeline
import gc

# Load 10 samples for initial testing
df = pd.read_csv("data/eval_inputs.csv").head(10)

texts = df["text"].tolist()

models = {
    "Model A": "distilbert-base-uncased-finetuned-sst-2-english",
    "Model B": "textattack/bert-base-uncased-SST-2"
}

for name, model_id in models.items():

    print("\n" + "=" * 60)
    print("Testing:", name)
    print("Model ID:", model_id)

    # Load model on CPU
    classifier = pipeline(
        "sentiment-analysis",
        model=model_id,
        tokenizer=model_id,
        device=-1
    )

    # Generate predictions
    predictions = classifier(
        texts,
        truncation=True,
        max_length=256,
        batch_size=4
    )

    # Display results
    for i, prediction in enumerate(predictions):
        raw_label = prediction["label"]

        # Normalize labels
        if raw_label in ["LABEL_0"]:
            label = "NEGATIVE"
        elif raw_label in ["LABEL_1"]:
            label = "POSITIVE"
        else:
            label = raw_label.upper()

        print(f"\nExample {i + 1}")
        print("Actual:    ", df.iloc[i]["label_name"])
        print("Predicted: ", label)
        print("Confidence:", round(prediction["score"], 4))

    # Free memory before loading next model
    del classifier
    gc.collect()

print("\nBoth models tested successfully!")
