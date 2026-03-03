---
title: "MLflow: Experiment Tracking & Model Management"
slug: mlflow
summary: "Track experiments, manage models, and deploy ML pipelines with MLflow — the open-source ML lifecycle platform."
tags: ["MLflow", "experiment-tracking", "model-registry", "ML-infrastructure", "reproducibility"]
visibility: public
---

# MLflow: Experiment Tracking & Model Management

## Overview

**MLflow** is an open-source platform for managing the ML lifecycle, including experimentation, reproducibility, and deployment. It solves the "experiment chaos" problem — tracking which hyperparameters, datasets, and code produced which results.

**Core components:**
- **MLflow Tracking:** Log experiments, parameters, metrics, artifacts
- **MLflow Projects:** Package ML code in reusable format
- **MLflow Models:** Package models for deployment
- **MLflow Registry:** Central model store with versioning and staging

---

## MLflow Tracking

### Core Concepts

| Concept | Description |
|---------|-------------|
| **Run** | Single execution of ML code |
| **Experiment** | Group of related runs |
| **Parameters** | Input configuration (learning_rate, batch_size) |
| **Metrics** | Logged values over time (loss, accuracy) |
| **Artifacts** | Files saved (model, plots, data) |
| **Tags** | Key-value metadata |

### Basic Usage

```python
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Set experiment
mlflow.set_experiment("iris-classification")

with mlflow.start_run(run_name="rf-experiment-1"):
    # Log parameters
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("max_depth", 5)
    mlflow.log_param("random_state", 42)

    # Train model
    model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate
    train_acc = accuracy_score(y_train, model.predict(X_train))
    val_acc = accuracy_score(y_val, model.predict(X_val))

    # Log metrics
    mlflow.log_metric("train_accuracy", train_acc)
    mlflow.log_metric("val_accuracy", val_acc)

    # Log model
    mlflow.sklearn.log_model(model, "random-forest-model")

    # Log artifacts
    mlflow.log_artifact("confusion_matrix.png")
    mlflow.log_artifact("feature_importance.csv")

    print(f"Run ID: {mlflow.active_run().info.run_id}")
```

### Logging Metrics Over Training

```python
with mlflow.start_run():
    for epoch in range(num_epochs):
        # Training step
        train_loss = train_one_epoch(model, train_loader)
        val_loss = validate(model, val_loader)

        # Log per-step metrics
        mlflow.log_metrics({
            "train_loss": train_loss,
            "val_loss": val_loss,
            "learning_rate": scheduler.get_last_lr()[0]
        }, step=epoch)
```

### PyTorch / Deep Learning Integration

```python
import mlflow.pytorch

with mlflow.start_run():
    mlflow.log_params({
        "batch_size": 32,
        "learning_rate": 1e-4,
        "optimizer": "AdamW",
        "model": "bert-base-uncased",
        "num_epochs": 10
    })

    for epoch in range(num_epochs):
        loss = train_epoch(model, optimizer)
        val_acc = evaluate(model, val_loader)

        mlflow.log_metrics({
            "train_loss": loss,
            "val_accuracy": val_acc
        }, step=epoch)

    # Save model with signature
    signature = mlflow.models.infer_signature(X_train, model(X_train).detach())
    mlflow.pytorch.log_model(model, "bert-classifier", signature=signature)
```

### HuggingFace Transformers Integration

```python
from transformers import TrainerCallback

class MLflowCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if state.is_world_process_zero:
            mlflow.log_metrics(logs, step=state.global_step)

trainer = Trainer(
    model=model,
    args=training_args,
    callbacks=[MLflowCallback()]
)

with mlflow.start_run():
    mlflow.log_params({
        "model_name": "bert-base-uncased",
        "learning_rate": training_args.learning_rate,
        "num_train_epochs": training_args.num_train_epochs
    })
    trainer.train()
```

---

## MLflow UI

Launch the tracking server:

```bash
# Start MLflow UI (reads from ./mlruns by default)
mlflow ui

# With remote backend
mlflow ui --backend-store-uri postgresql://user:pass@localhost/mlflow \
          --default-artifact-root s3://my-bucket/mlflow-artifacts

# Access at http://localhost:5000
```

The UI shows:
- All experiments and runs
- Metric plots over time
- Parameter comparison across runs
- Artifact downloads

---

## MLflow Model Registry

### Register a Model

```python
with mlflow.start_run() as run:
    # ... training code ...
    mlflow.pytorch.log_model(model, "model")

# Register to Model Registry
model_uri = f"runs:/{run.info.run_id}/model"
mv = mlflow.register_model(model_uri, "BertClassifier")
print(f"Version: {mv.version}")  # Version: 1
```

### Model Lifecycle Stages

```python
from mlflow.tracking import MlflowClient

client = MlflowClient()

# Staging (ready for testing)
client.transition_model_version_stage(
    name="BertClassifier",
    version=1,
    stage="Staging"
)

# Production (live)
client.transition_model_version_stage(
    name="BertClassifier",
    version=1,
    stage="Production"
)

# Archive old version
client.transition_model_version_stage(
    name="BertClassifier",
    version=0,
    stage="Archived"
)
```

### Load a Model from Registry

```python
# Load latest Production model
model = mlflow.pytorch.load_model("models:/BertClassifier/Production")

# Load specific version
model = mlflow.pytorch.load_model("models:/BertClassifier/3")
```

---

## MLflow Projects

Standardize ML code packaging:

```yaml
# MLproject
name: bert-finetuning

conda_env: conda.yaml

entry_points:
  main:
    parameters:
      learning_rate: {type: float, default: 1e-4}
      num_epochs: {type: int, default: 3}
      model_name: {type: str, default: "bert-base-uncased"}
    command: "python train.py --lr {learning_rate} --epochs {num_epochs} --model {model_name}"
```

```bash
# Run a project
mlflow run . -P learning_rate=2e-5 -P num_epochs=5

# Run from GitHub
mlflow run https://github.com/user/bert-project -P learning_rate=1e-4
```

---

## Hyperparameter Search with MLflow

```python
import optuna

def objective(trial):
    lr = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
    dropout = trial.suggest_float("dropout", 0.1, 0.5)
    num_layers = trial.suggest_int("num_layers", 2, 6)

    with mlflow.start_run(nested=True):
        mlflow.log_params({"lr": lr, "dropout": dropout, "num_layers": num_layers})
        val_loss = train_and_evaluate(lr, dropout, num_layers)
        mlflow.log_metric("val_loss", val_loss)

    return val_loss

with mlflow.start_run(run_name="hyperparameter-search"):
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=50)

    mlflow.log_params(study.best_params)
    mlflow.log_metric("best_val_loss", study.best_value)
```

---

## Production Deployment

### Serve a Model as REST API

```bash
# Serve from model registry
mlflow models serve -m "models:/BertClassifier/Production" -p 5001

# Or from run artifacts
mlflow models serve -m "runs:/abc123/model" -p 5001
```

```bash
# Test the served model
curl http://localhost:5001/invocations \
    -H "Content-Type: application/json" \
    -d '{"inputs": [{"text": "This movie is great!"}]}'
```

### Docker Deployment

```bash
# Build Docker image from model
mlflow models build-docker -m "models:/BertClassifier/Production" -n bert-api

docker run -p 5001:8080 bert-api
```

---

## Key Takeaways

1. **MLflow Tracking** logs parameters, metrics, and artifacts for every run — essential for reproducibility
2. **Compare runs** in the UI to identify winning hyperparameter combinations
3. **Model Registry** provides versioning and staging (Staging → Production → Archived)
4. **MLflow Projects** standardize code packaging for reproducible execution
5. **Nested runs** enable logging individual trials within a hyperparameter sweep
6. **Serve directly** with `mlflow models serve` or build Docker images for deployment

## References

- MLflow Documentation — https://mlflow.org/docs/latest/
- MLflow GitHub — https://github.com/mlflow/mlflow
