
from datasets import load_dataset
import pandas as pd

# Load SST-2 validation dataset
dataset = load_dataset(
    "nyu-mll/glue",
    "sst2",
    split="validation"
)

# Select 200 reproducible samples
dataset = dataset.shuffle(seed=42).select(range(200))

# Create DataFrame
df = pd.DataFrame({
    "text": dataset["sentence"],
    "label": dataset["label"]
})

# Convert numerical labels into readable labels
df["label_name"] = df["label"].map({
    0: "NEGATIVE",
    1: "POSITIVE"
})

# Save CSV
df.to_csv("data/eval_inputs.csv", index=False)

print("Dataset created successfully!")
print("Total samples:", len(df))
print(df.head())
print("\nLabel distribution:")
print(df["label_name"].value_counts())
