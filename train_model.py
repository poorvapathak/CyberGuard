# train_model.py

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

# Load dataset
df = pd.read_csv("PhiUSIIL_Phishing_URL_Dataset.csv")

# Drop non-numeric / unnecessary columns
drop_cols = ['URL', 'Domain', 'TLD', 'Title']
df.drop(columns=drop_cols, inplace=True)

# Define features and label
X = df.drop(columns=['label'])
y = df['label']

# Optional: Log transform extreme columns (like URLLength)
# X['URLLength'] = X['URLLength'].apply(lambda x: np.log1p(x))

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save model
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Model saved as model.pkl")

# Predict and evaluate
y_pred = model.predict(X_test)

print("\n🔍 Classification Report:")
print(classification_report(y_test, y_pred))

# Plot confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=['Phishing', 'Legit'], yticklabels=['Phishing', 'Legit'])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
print("📊 Confusion matrix saved as confusion_matrix.png")

# Feature importance
importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
top_features = importances.head(15)

plt.figure(figsize=(8, 6))
sns.barplot(x=top_features, y=top_features.index)
plt.title("Top 15 Important Features")
plt.xlabel("Importance")
plt.tight_layout()
plt.savefig("feature_importance.png")
print("📊 Feature importance plot saved as feature_importance.png")
