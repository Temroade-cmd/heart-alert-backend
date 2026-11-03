"""
Heart Disease Prediction Model Loader
Loads the trained XGBoost model and makes predictions
"""

import joblib
import numpy as np
import os
import xgboost as xgb
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'xgboost_model.pkl')

try:
    model = joblib.load(MODEL_PATH)
    print(f"✅ Model loaded successfully")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None

def predict_heart_disease(age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope):
    if model is None:
        return {
            "error": "Model not loaded",
            "risk_score": 0.5,
            "risk_category": "moderate",
            "has_disease": False,
            "recommendation": "Model unavailable. Please try again later."
        }
    
    # The API receives cp in 0-3 and slope in 0-2 (Cleveland coding, used by the app and the thesis).
    # The model was trained on the combined dataset, where cp is 1-4 and slope is 1-3,
    # so both are shifted by one before calling the model.
    features = np.array([[
        age, sex, cp + 1, trestbps, chol, fbs, restecg, 
        thalach, exang, oldpeak, slope + 1
    ]])
    
    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0][1]
    
    risk_score = float(probability)
    
    if risk_score < 0.3:
        category = "low"
        recommendation = "Low risk profile. Continue healthy lifestyle and routine screenings."
    elif risk_score < 0.6:
        category = "moderate"
        recommendation = "Monitor closely. Consider lifestyle modifications and regular check-ups."
    else:
        category = "high"
        recommendation = "Immediate cardiology consultation recommended. Further diagnostic tests may be necessary."
    
    return {
        "risk_score": round(risk_score, 2),
        "risk_category": category,
        "has_disease": bool(prediction),
        "recommendation": recommendation
    }