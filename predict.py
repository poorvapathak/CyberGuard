# predict.py
import pandas as pd
import pickle

# Load the model, scaler, feature column order, and training medians
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("feature_columns.pkl", "rb") as f:
    feature_columns = pickle.load(f)

with open("train_medians.pkl", "rb") as f:
    train_medians = pickle.load(f)

# Define numeric columns (based on training feature set)
numeric_cols = feature_columns  # All selected features are numeric after scaling

# Select top N features for user input (based on feature importance)
user_features = feature_columns  # Directly use the list of top 20 features

def predict_url(user_input: dict):
    # Create a DataFrame with all features, initialized to training medians
    input_data = pd.DataFrame([train_medians], columns=feature_columns)

    # Replace with user-provided values (only for selected features)
    for key, value in user_input.items():
        if key in input_data.columns:
            input_data[key] = value

    # Ensure all features are present (fill missing with medians)
    for col in feature_columns:
        if col not in user_input:
            input_data[col] = train_medians[col]

    # Scale the input data
    input_data[numeric_cols] = scaler.transform(input_data[numeric_cols])

    # Make prediction
    prediction = model.predict(input_data)[0]
    confidence = model.predict_proba(input_data)[0][prediction]

    label = "Phishing" if prediction == 1 else "Legit"
    return label, round(confidence, 3)

# Test block with cases aligned to training medians and PhiUSIIL patterns
if __name__ == "__main__":
    test_cases = [
        {
            'name': 'Legit-Looking URL',
            'input': {
                'URLLength': 60,  # Near median value
                'URLSimilarityIndex': 30.0,
                'NoOfExternalRef': 4,
                'LineOfCode': 80,
                'IsHTTPS': 1,
                'NoOfSelfRef': 3,
                'NoOfImage': 6,
                'NoOfJS': 2,
                'NoOfCSS': 1,
                'HasSocialNet': 1,
                'URL_Entropy': 0.2  # Adjusted for realism
            }
        },
        {
            'name': 'Suspicious Phishing URL',
            'input': {
                'URLLength': 170,
                'URLSimilarityIndex': 85.0,
                'NoOfExternalRef': 55,
                'LineOfCode': 15,
                'IsHTTPS': 0,
                'NoOfSelfRef': 1,
                'NoOfImage': 0,
                'NoOfJS': 0,
                'NoOfCSS': 0,
                'HasSocialNet': 0,
                'URL_Entropy': 0.55
            }
        },
        {
            'name': 'Extreme Phishing URL',
            'input': {
                'URLLength': 200,
                'URLSimilarityIndex': 95.0,
                'NoOfExternalRef': 80,
                'LineOfCode': 5,
                'IsHTTPS': 0,
                'NoOfSelfRef': 0,
                'NoOfImage': 0,
                'NoOfJS': 0,
                'NoOfCSS': 0,
                'HasSocialNet': 0,
                'URL_Entropy': 0.65
            }
        }
    ]

    for case in test_cases:
        label, confidence = predict_url(case['input'])
        print(f"\n🔎 Test Case: {case['name']}")
        print(f"Prediction: {label} (Confidence: {confidence})")