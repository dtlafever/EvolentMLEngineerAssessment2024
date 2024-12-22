from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import joblib
import os
import pandas as pd

# Initialize FastAPI
app = FastAPI()


# Define the input data schema
class PredictionInput(BaseModel):
    features: List[dict]




# Define a dictionary to store loaded models
loaded_models = {}

# Define the models folder
# TODO: make this a better path
models_folder = "models/"


@app.get("/")
def root():
    return {"message": "Welcome to the model prediction API. Use /predict to make predictions!"}

@app.get("/models")
def get_models():
    return {"models": os.listdir(models_folder)}

@app.post("/predict/{model_name}")
def predict(model_name: str, input_data: PredictionInput):
    # Ensure the model is available
    model_path = os.path.join(models_folder, model_name)
    if not os.path.exists(model_path):
        raise HTTPException(status_code=404, detail="Model not found")

    # Load the model if not already cached
    if model_name not in loaded_models:
        try:
            print(f"Loading model: {model_name}")
            # TODO: fix bug when trying to load a model
            loaded_models[model_name] = joblib.load(model_path)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error loading model: {e}")

    model = loaded_models[model_name]

    # Convert input data to a DataFrame
    try:
        df = pd.DataFrame(input_data.features)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid input data format: {e}")

    # Make predictions
    try:
        predictions = model.predict(df)
        # Convert predictions to a list if they are numpy arrays
        predictions = predictions.tolist() if hasattr(predictions, "tolist") else predictions
        return {"predictions": predictions}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error making predictions: {e}")
