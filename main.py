# main.py

import os
import math
import joblib
import numpy as np
import pandas as pd
from itertools import product
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, Any

# --- 1. SETUP AND MODEL LOADING ---

# Initialize the FastAPI app
app = FastAPI(
    title="Crop Yield Prediction & Optimization API",
    description="An API to predict crop yield and recommend optimal fertilizer and pesticide usage.",
    version="1.0.0"
)

# Define the path to the model artifact
MODEL_PATH = os.path.join("artifacts", "yield_model_pipeline.joblib")

# Load the trained model pipeline when the application starts
try:
    model_pipeline = joblib.load(MODEL_PATH)
    print(f"Model pipeline loaded successfully from {MODEL_PATH}")
except FileNotFoundError:
    print(f"ERROR: Model file not found at {MODEL_PATH}. The API will not work.")
    model_pipeline = None
except Exception as e:
    print(f"An error occurred while loading the model: {e}")
    model_pipeline = None

# Define the full list of features your model pipeline expects
MODEL_FEATURES = [
    'Crop', 'Crop_Year', 'Season', 'State',
    'Area', 'Annual_Rainfall', 'Fertilizer', 'Pesticide',
    'group_avg_yield', 'group_avg_fertilizer', 'group_avg_pesticide', 'group_avg_rainfall',
    'Temp', 'Humidity', 'WindSpeed',
    'Soil_Moisture', 'Soil_Temperature', 'Soil_pH', 'Soil_OrganicCarbon'
]


# --- 2. PYDANTIC MODELS FOR INPUT DATA VALIDATION ---

# Defines the required inputs for a prediction request.
class FarmProfile(BaseModel):
    crop: str
    crop_year: int
    season: str
    state: str
    area: float
    fertilizer: float
    pesticide: float
    annual_rainfall_mm: float


# Extends the basic profile with parameters for optimization.
class OptimizationProfile(FarmProfile):
    cost_lambda: float = Field(default=0.05, ge=0, le=1.0, description="Penalty for changing inputs (0=none, 1=max).")
    top_k: int = Field(default=5, gt=0, le=20, description="Number of top recommendations to return.")
    n_grid: int = Field(default=21, gt=5, le=51, description="Grid search points for optimization.")


# --- 3. HELPER FUNCTIONS ---

def build_input_dataframe(profile: Dict[str, Any]) -> pd.DataFrame:
    """Constructs a single-row DataFrame for the model pipeline."""
    # Create a dictionary with all expected features, initialized to NaN
    row = {feature: np.nan for feature in MODEL_FEATURES}
    # Update the dictionary with values from the user's profile
    row.update({
        'Crop': profile["crop"],
        'Crop_Year': profile["crop_year"],
        'Season': profile["season"],
        'State': profile["state"],
        'Area': profile["area"],
        'Annual_Rainfall': profile["annual_rainfall_mm"],
        'Fertilizer': profile["fertilizer"],
        'Pesticide': profile["pesticide"],
    })
    # The model's imputer will handle the other missing values
    return pd.DataFrame([row])


def optimize_inputs(pipeline, profile_dict: Dict, n_grid: int, cost_lambda: float, top_k: int):
    """Grid-search to find optimal Fertilizer & Pesticide values."""
    base_profile_df = build_input_dataframe(profile_dict)
    
    curr_fert = float(profile_dict.get('fertilizer', 0))
    curr_pest = float(profile_dict.get('pesticide', 0))
    eps = 1e-8

    fert_bounds = (max(0.0, curr_fert * 0.5), curr_fert * 1.5)
    pest_bounds = (max(0.0, curr_pest * 0.5), curr_pest * 1.5)

    fert_grid = np.linspace(fert_bounds[0], fert_bounds[1], n_grid)
    pest_grid = np.linspace(pest_bounds[0], pest_bounds[1], n_grid)
    
    rows = []
    base_row_dict = base_profile_df.iloc[0].to_dict()
    for f, p in product(fert_grid, pest_grid):
        row = base_row_dict.copy()
        row['Fertilizer'] = f
        row['Pesticide'] = p
        rows.append(row)
    
    cand_df = pd.DataFrame(rows, columns=MODEL_FEATURES)
    
    preds = pipeline.predict(cand_df)
    
    rel_fert_change = np.abs((cand_df['Fertilizer'] - curr_fert) / (curr_fert + eps))
    rel_pest_change = np.abs((cand_df['Pesticide'] - curr_pest) / (curr_pest + eps))
    penalty = cost_lambda * (rel_fert_change + rel_pest_change)
    objective = preds - penalty
    
    cand_df['predicted_yield'] = preds
    cand_df['objective_score'] = objective
    
    top_results = cand_df.sort_values(by='objective_score', ascending=False).head(top_k)
    return top_results[['Fertilizer', 'Pesticide', 'predicted_yield']]


# --- 4. API ENDPOINTS ---

@app.on_event("startup")
async def startup_event():
    if model_pipeline is None:
        raise RuntimeError("Model could not be loaded. Please check the logs.")

@app.get("/", tags=["Status"])
def read_root():
    """Check if the API is running."""
    return {"status": "OK", "model_loaded": model_pipeline is not None}


@app.post("/predict", tags=["Prediction"])
def predict_yield(profile: FarmProfile) -> Dict[str, Any]:
    """Predicts the crop yield for a given set of farm inputs."""
    
    profile_dict = profile.dict()
    input_df = build_input_dataframe(profile_dict)
    prediction = model_pipeline.predict(input_df)[0]
    
    return {
        "predicted_yield_tons_per_hectare": round(float(prediction), 4),
        "inputs_used": profile_dict
    }


@app.post("/recommend", tags=["Recommendation"])
def get_recommendations(profile: OptimizationProfile) -> Dict[str, Any]:
    """
    Runs an optimization to find the best Fertilizer and Pesticide
    amounts to maximize yield, balanced by a cost factor.
    """
    profile_dict = profile.dict()
    
    current_yield = model_pipeline.predict(build_input_dataframe(profile_dict))[0]
    
    recommendations_df = optimize_inputs(
        pipeline=model_pipeline,
        profile_dict=profile_dict,
        n_grid=profile.n_grid,
        cost_lambda=profile.cost_lambda,
        top_k=profile.top_k
    )
    
    return {
        "baseline_yield_prediction": round(float(current_yield), 4),
        "top_recommendations": recommendations_df.to_dict(orient="records"),
        "optimization_inputs": {
            "cost_lambda": profile.cost_lambda,
            "top_k": profile.top_k,
            "grid_size": profile.n_grid
        }
    }
