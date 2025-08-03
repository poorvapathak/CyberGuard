# flask_api.py
from flask import Flask, request, jsonify
import pandas as pd
import pickle

app = Flask(__name__)  # Fixed: Changed 'name' to '__name__'

# Load model and feature column order
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("feature_columns.pkl", "rb") as f:
    feature_columns = pickle.load(f)

# Define numeric columns
numeric_cols = [col for col in feature_columns if col != 'label']

# Load defaults for missing values (optional, using zeros for simplicity)
# Note: In production, you might want to use training means, but ensure no leakage
input_data_default = pd.DataFrame(0, index=[0], columns=feature_columns)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json

    # Create input DataFrame with default values
    input_data = input_data_default.copy()

    # Update with user-provided values
    for key in data:
        if key in input_data.columns:
            input_data[key] = data[key]

    # Scale the input data
    input_data[numeric_cols] = scaler.transform(input_data[numeric_cols])

    # Predict
    prediction = model.predict(input_data)[0]
    confidence = model.predict_proba(input_data)[0][prediction]

    return jsonify({
        "prediction": "Phishing" if prediction == 1 else "Legit",
        "confidence": round(confidence, 3)
    })

if __name__ == "__main__":
    app.run(debug=True)