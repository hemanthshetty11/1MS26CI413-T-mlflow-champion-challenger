# Champion–Challenger Model Comparison with Hugging Face and MLflow

## 1. Project Overview

This individual project compares two Hugging Face sentiment classification models using MLflow.

**Task:** Binary Sentiment Classification

**Dataset:** SST-2 validation dataset (200 samples, seed 42)

**Models:**
- Model A: `distilbert-base-uncased-finetuned-sst-2-english`
- Model B: `textattack/bert-base-uncased-SST-2`

**Objective:** Evaluate both models using the same dataset, compare their performance, select a Champion, and register both models in MLflow.

## 2. Champion Selection Rule

The primary evaluation metric is Macro F1, where a higher score is better.

The selection criteria are:

1. The Challenger must improve Macro F1 by at least 0.01.
2. Its average inference latency must not exceed five times the existing Champion's latency.
3. If both conditions are satisfied, the Challenger becomes the new Champion. Otherwise, the existing Champion remains.

Both models were evaluated using the same 200 SST-2 validation samples in the same order.

## 3. Evaluation Results

| Metric | Model A | Model B |
|---|---:|---:|
| Accuracy | 0.9150 | 0.9500 |
| Precision (Macro) | 0.9183 | 0.9505 |
| Recall (Macro) | 0.9121 | 0.9491 |
| Macro F1 | 0.9141 | 0.9497 |
| Average Latency (ms) | 37.57 | 80.23 |
| Total Inference Time (s) | 7.51 | 16.05 |

## 4. Champion Decision

Model B achieved a Macro F1 improvement of approximately 0.0356, exceeding the required improvement of 0.01.

Its average latency was approximately 2.14 times that of Model A, which is within the permitted five-times limit.

Therefore, Model B was selected as the Champion.

Model B provides better classification performance, while Model A offers faster inference. This demonstrates the trade-off between prediction quality and speed.

## 5. MLflow Model Registry

**Registered Model Name:** `Sentiment_Classifier`

| Role | Model | Version |
|---|---|---:|
| Champion | TextAttack BERT | 2 |
| Challenger | DistilBERT | 1 |

**Champion URI:**
`models:/Sentiment_Classifier@champion`

**Challenger URI:**
`models:/Sentiment_Classifier@challenger`

Both aliases were successfully verified using the MLflow Model Registry.

## 6. New Data Prediction

The registered Champion was loaded directly from MLflow using its alias.

Eight new movie-review sentences were supplied for prediction.

The script displays:
- MLflow model version
- Hugging Face model ID
- Predicted sentiment
- Prediction confidence

The predictions are saved to:

`outputs/new_data_predictions.csv`

## 7. Project Structure

```text
1MS26CI413-T-mlflow-champion-challenger/
│
├── data/
│   └── eval_inputs.csv
│
├── outputs/
│   ├── model_A/
│   │   ├── predictions_A.csv
│   │   └── confusion_matrix_A.png
│   │
│   ├── model_B/
│   │   ├── predictions_B.csv
│   │   └── confusion_matrix_B.png
│   │
│   ├── comparison/
│   │   ├── metrics_summary.json
│   │   ├── metric_comparison.png
│   │   └── disagreement_examples.csv
│   │
│   └── new_data_predictions.csv
│
├── screenshots/
│   ├── mlflow_comparison.png
│   ├── mlflow_run.png
│   ├── model_registry.png
│   ├── model_registry_verified.png
│   ├── prediction_output_1.png
│   └── prediction_output_2.png
│
├── evaluate.py
├── register_champion.py
├── predict_from_mlflow.py
├── complete_mlflow_runs.py
├── create_comparison_chart.py
├── prepare_data.py
├── test_models.py
├── requirements.txt
└── README.md
```

## 8. How to Run

### Install dependencies

```bash
pip install -r requirements.txt
```

### Start MLflow tracking server

```bash
mlflow server --host 127.0.0.1 --port 5000
```

Keep the MLflow server running in a separate terminal.

### Run model evaluation

```bash
python evaluate.py
```

### Register models and assign aliases

```bash
python register_champion.py
```

### Predict new examples using the registered Champion

```bash
python predict_from_mlflow.py
```

## 9. Disagreement Analysis

Both models produced different predictions on 11 out of 200 examples.

Five disagreement examples were selected for further analysis.

Among these five examples:
- Model B correctly classified four examples.
- Model A correctly classified one example.

Model B correctly handled several difficult positive reviews that Model A incorrectly classified as negative. However, Model B also made an incorrect prediction on one positive review.

These disagreements demonstrate that different models can interpret the same sentences differently. Higher overall Macro F1 does not mean every individual prediction is correct.

## 10. Limitations

- The evaluation uses only 200 samples.
- Both