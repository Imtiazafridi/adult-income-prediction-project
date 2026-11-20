
from pathlib import Path
import joblib
import pandas as pd
from fastapi import FastAPI

# Load saved files
BASE_DIR = Path(__file__).resolve().parent

model = joblib.load(BASE_DIR / "best_automl_model.pkl")
preprocessor = joblib.load(BASE_DIR / "preprocessor.pkl")

# Create FastAPI application
app = FastAPI(title="Adult Income Prediction API")

@app.get("/")
def home():
    return {"message": "Adult Income Prediction API is running"}

@app.post("/predict")
def predict(data: dict):
    input_data = pd.DataFrame([data])

    processed_data = preprocessor.transform(input_data)
    prediction = model.predict(processed_data)[0]

    result = ">50K" if prediction == 1 else "<=50K"

    return {"prediction": result}
