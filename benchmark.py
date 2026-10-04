import json
import time
import platform

import numpy as np
import pandas as pd
import lightgbm as lgb

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score
)


# ============================================================
# 1. LOAD DATASET
# ============================================================

DATASET_PATH = "creditcard.csv"

print("Loading dataset...")
df = pd.read_csv(DATASET_PATH)

print(f"Dataset shape: {df.shape}")
print(f"Fraud transactions: {df['Class'].sum()}")
print(f"Normal transactions: {(df['Class'] == 0).sum()}")

X = df.drop(columns=["Class"])
y = df["Class"]


# ============================================================
# 2. TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print(f"Train size: {len(X_train)}")
print(f"Test size : {len(X_test)}")


# ============================================================
# 3. CREATE LIGHTGBM MODEL
# ============================================================

model = lgb.LGBMClassifier(
    n_estimators=300,
    learning_rate=0.05,
    num_leaves=31,
    random_state=42,
    n_jobs=-1,
    verbosity=-1
)


# ============================================================
# 4. TRAINING TIME
# ============================================================

print("\nTraining LightGBM...")

start_train = time.perf_counter()

model.fit(X_train, y_train)

end_train = time.perf_counter()

training_time = end_train - start_train

print(f"Training time: {training_time:.4f} seconds")


# ============================================================
# 5. MODEL EVALUATION
# ============================================================

print("\nEvaluating...")

y_prob = model.predict_proba(X_test)[:, 1]
y_pred = (y_prob >= 0.5).astype(int)

auc = roc_auc_score(y_test, y_prob)
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)

print(f"AUC-ROC   : {auc:.6f}")
print(f"Accuracy  : {accuracy:.6f}")
print(f"F1-Score  : {f1:.6f}")
print(f"Precision : {precision:.6f}")
print(f"Recall    : {recall:.6f}")


# ============================================================
# 6. INFERENCE LATENCY
# ============================================================

# Lấy 1 sample
sample = X_test.iloc[[0]]

# Warm-up để tránh lần predict đầu ảnh hưởng benchmark
for _ in range(10):
    model.predict_proba(sample)

# Đo nhiều lần để kết quả latency ổn định hơn
num_latency_runs = 100

start_latency = time.perf_counter()

for _ in range(num_latency_runs):
    model.predict_proba(sample)

end_latency = time.perf_counter()

average_latency_sec = (
    end_latency - start_latency
) / num_latency_runs

latency_ms = average_latency_sec * 1000

print(f"\nInference latency (1 sample): {latency_ms:.6f} ms")


# ============================================================
# 7. INFERENCE THROUGHPUT
# ============================================================

num_samples = min(1000, len(X_test))
batch = X_test.iloc[:num_samples]

# Warm-up
model.predict_proba(batch)

start_throughput = time.perf_counter()

model.predict_proba(batch)

end_throughput = time.perf_counter()

batch_time = end_throughput - start_throughput

throughput = num_samples / batch_time

print(
    f"Inference throughput ({num_samples} samples): "
    f"{throughput:.2f} samples/sec"
)

print(f"Batch inference time: {batch_time * 1000:.4f} ms")


# ============================================================
# 8. SAVE RESULT TO JSON
# ============================================================

results = {
    "dataset": {
        "name": "Credit Card Fraud Detection",
        "total_samples": int(len(df)),
        "num_features": int(X.shape[1]),
        "train_samples": int(len(X_train)),
        "test_samples": int(len(X_test)),
        "fraud_samples": int(y.sum()),
        "normal_samples": int((y == 0).sum())
    },

    "model": {
        "name": "LightGBM",
        "type": "LGBMClassifier",
        "n_estimators": 300,
        "learning_rate": 0.05,
        "num_leaves": 31
    },

    "metrics": {
        "auc_roc": float(auc),
        "accuracy": float(accuracy),
        "f1_score": float(f1),
        "precision": float(precision),
        "recall": float(recall)
    },

    "performance": {
        "training_time_seconds": float(training_time),
        "inference_latency_ms_per_sample": float(latency_ms),
        "throughput_samples_per_second": float(throughput),
        "throughput_batch_size": int(num_samples),
        "batch_inference_time_ms": float(batch_time * 1000)
    },

    "system": {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "cpu_count": __import__("os").cpu_count(),
        "lightgbm_version": lgb.__version__
    }
}

with open("benchmark_result.json", "w") as f:
    json.dump(results, f, indent=4)

print("\n========================================")
print("Benchmark completed!")
print("Result saved to benchmark_result.json")
print("========================================")
