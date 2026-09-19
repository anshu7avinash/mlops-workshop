"""Train, evaluate, and track the product recommender.

Run this file from the repository root:
    python model/train_model.py

The script is deliberately small so it can be read in a workshop.  DVC calls
it when a dependency changes; MLflow records every execution as an experiment.
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
import yaml
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.neighbors import NearestNeighbors

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "model"
DATA_PATH = MODEL_DIR / "products.csv"
MODEL_PATH = MODEL_DIR / "recommendation_model.pkl"
METRICS_PATH = MODEL_DIR / "metrics.json"
METADATA_PATH = MODEL_DIR / "model_metadata.json"


def sha256(path: Path) -> str:
    """Return a short, human-readable fingerprint for the training data."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_params() -> dict:
    with (ROOT / "params.yaml").open(encoding="utf-8") as file:
        return yaml.safe_load(file)["model"]


def recommendation_metrics(model: Pipeline, products: pd.DataFrame, top_k: int) -> dict:
    """Compute workshop-friendly proxy metrics for an unlabeled recommender.

    This catalogue does not contain click or purchase labels, so these are not
    accuracy metrics.  They make the effect of a data or parameter change
    visible, while giving us an opportunity to explain what production teams
    would measure with real user-feedback data.
    """
    features = products[["category", "price", "rating"]]
    transformed = model.named_steps["preprocessor"].transform(features)
    neighbours = min(top_k + 1, len(products))
    _, indices = model.named_steps["model"].kneighbors(transformed, n_neighbors=neighbours)

    category_matches = []
    recommended_ratings = []
    recommended_ids = set()

    for row_index, row_indices in enumerate(indices):
        recommendations = [index for index in row_indices if index != row_index][:top_k]
        if not recommendations:
            continue
        selected_category = products.iloc[row_index]["category"]
        category_matches.extend(
            products.iloc[index]["category"] == selected_category for index in recommendations
        )
        recommended_ratings.extend(products.iloc[index]["rating"] for index in recommendations)
        recommended_ids.update(products.iloc[index]["product_id"] for index in recommendations)

    return {
        "catalog_size": int(len(products)),
        "unique_categories": int(products["category"].nunique()),
        "category_match_at_k": round(sum(category_matches) / len(category_matches), 4),
        "average_recommended_rating": round(sum(recommended_ratings) / len(recommended_ratings), 4),
        "recommendation_coverage": round(len(recommended_ids) / len(products), 4),
    }

def main() -> None:
    params = load_params()
    products = pd.read_csv(DATA_PATH)
    features = products[["category", "price", "rating"]]
    data_hash = sha256(DATA_PATH)
    n_neighbors = min(int(params["n_neighbors"]), len(products))

    preprocessor = ColumnTransformer(
        transformers=[
            ("category", OneHotEncoder(handle_unknown="ignore"), ["category"]),
            ("numeric", StandardScaler(), ["price", "rating"]),
        ]
    )
    recommendation_model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", NearestNeighbors(n_neighbors=n_neighbors, metric=params["metric"])),
        ]
    )

    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns"))
    mlflow.set_experiment(params["experiment_name"])

    with mlflow.start_run(run_name=f"catalog-{data_hash[:8]}") as run:
        recommendation_model.fit(features)
        metrics = recommendation_metrics(recommendation_model, products, int(params["top_k"]))
        metadata = {
            "model_name": "content-based-nearest-neighbours",
            "model_version": run.info.run_id[:8],
            "mlflow_run_id": run.info.run_id,
            "training_data_sha256": data_hash,
            "training_data_rows": int(len(products)),
            "parameters": params,
            "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        }

        joblib.dump({"model": recommendation_model, "products": products}, MODEL_PATH)
        METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

        mlflow.log_params({
            "n_neighbors": n_neighbors,
            "distance_metric": params["metric"],
            "top_k": int(params["top_k"]),
            "data_sha256": data_hash,
        })
        mlflow.log_metrics(metrics)
        mlflow.log_dict(metadata, "model_metadata.json")
        mlflow.log_artifact(str(METRICS_PATH), artifact_path="evaluation")
        mlflow.sklearn.log_model(recommendation_model, artifact_path="recommendation_model")

    print(f"Model saved: {MODEL_PATH}")
    print(f"MLflow run: {run.info.run_id}")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
