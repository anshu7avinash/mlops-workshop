"""FastAPI inference service for the workshop recommendation model."""

import json
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI(title="E-commerce Product Recommendation API", version="1.0.0")

# Load model and product data
MODEL_DIRECTORY = Path("model")
model_data = joblib.load(MODEL_DIRECTORY / "recommendation_model.pkl")
recommendation_model = model_data["model"]
products = model_data["products"]


def get_model_metadata() -> dict:
    metadata_file = MODEL_DIRECTORY / "model_metadata.json"
    if metadata_file.exists():
        return json.loads(metadata_file.read_text(encoding="utf-8"))
    return {
        "model_version": "baseline-untracked",
        "message": "Run model/train_model.py to create tracked metadata.",
    }

class RecommendationRequest(BaseModel):
    product_id: int

@app.get("/")
def home():
    return {
        "message": "E-commerce Product Recommendation API is running"
    }

@app.get("/products")
def get_products():
    return products.to_dict(orient="records")


@app.get("/model-info")
def model_info():
    """Lets learners verify exactly which model is serving traffic."""
    return get_model_metadata()

@app.post("/recommend")
def recommend_products(request: RecommendationRequest):
    product_id = request.product_id

    selected_product = products[products["product_id"] == product_id]

    if selected_product.empty:
        raise HTTPException(
            status_code=404,
            detail="Product not found"
        )

    # Prepare selected product features
    selected_features = selected_product[["category", "price", "rating"]]

    # Find nearest products
    distances, indices = recommendation_model.named_steps["model"].kneighbors(
        recommendation_model.named_steps["preprocessor"].transform(selected_features)
    )

    recommended_indices = indices[0]

    recommendations = products.iloc[recommended_indices]

    # Remove the selected product itself from recommendations
    recommendations = recommendations[
        recommendations["product_id"] != product_id
    ]

    return {
        "selected_product": selected_product.to_dict(orient="records")[0],
        "recommendations": recommendations.to_dict(orient="records"),
        "model": get_model_metadata(),
    }
