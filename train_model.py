# train_model.py
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import RobustScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV

warnings.filterwarnings("ignore")

# === 1. Load Dataset and Create Holdout Set ===
df = pd.read_csv("PhiUSIIL_Phishing_URL_Dataset.csv")
df.drop_duplicates(inplace=True)

# Reserve 15% as holdout set
df_holdout, df_train = train_test_split(df, test_size=0.85, stratify=df['label'], random_state=42)
print(f"📊 Holdout shape: {df_holdout.shape}, Training shape: {df_train.shape}")

# Drop non-feature columns
drop_cols = ['URL', 'Domain', 'TLD', 'Title']
df_train.drop(columns=[col for col in drop_cols if col in df_train.columns], inplace=True)
df_holdout.drop(columns=[col for col in drop_cols if col in df_holdout.columns], inplace=True)

# === 2. Class Balance Check ===
label_dist = df_train['label'].value_counts(normalize=True)
print("\n⚖️ Class Balance (Training):")
print(label_dist.rename("proportion"))

# === 3. Separate Features and Labels ===
X_train_full = df_train.drop(columns=['label'])
y_train_full = df_train['label']
X_holdout_full = df_holdout.drop(columns=['label'])
y_holdout = df_holdout['label']

# === 4. Handle Outliers with RobustScaler ===
numeric_cols = X_train_full.select_dtypes(include='number').columns
scaler = RobustScaler()
X_train_full[numeric_cols] = scaler.fit_transform(X_train_full[numeric_cols])
X_holdout_full[numeric_cols] = scaler.transform(X_holdout_full[numeric_cols])

# === 5. Initial Decision Tree Fit for Feature Importance ===
model_initial = DecisionTreeClassifier(random_state=42, class_weight='balanced')
model_initial.fit(X_train_full, y_train_full)
importances = pd.Series(model_initial.feature_importances_, index=X_train_full.columns)
top_features = importances.sort_values(ascending=False).head(20).index

# Subset all datasets to top 20 features
X_train_full = X_train_full[top_features]
X_holdout_full = X_holdout_full[top_features]

# Reapply scaling on selected features
scaler = RobustScaler()
X_train_full = pd.DataFrame(scaler.fit_transform(X_train_full), columns=X_train_full.columns)
X_holdout_full = pd.DataFrame(scaler.transform(X_holdout_full), columns=X_holdout_full.columns)

# === 6. Train-Test Split on Training Data with Validation Set ===
X_train, X_temp, y_train, y_temp = train_test_split(X_train_full, y_train_full, test_size=0.3, stratify=y_train_full, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)
print(f"Training shape: {X_train.shape}, Validation shape: {X_val.shape}, Test shape: {X_test.shape}")

# === 7. Save Training Medians for Default Values ===
train_medians = X_train.median()
with open("train_medians.pkl", "wb") as f:
    pickle.dump(train_medians, f)

# === 8. Hyperparameter Tuning with GridSearchCV on Sample ===
X_train_sample = X_train.sample(frac=0.2, random_state=42)
y_train_sample = y_train[X_train_sample.index]
param_grid = {
    'max_depth': [5, 10, 15],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
model = DecisionTreeClassifier(random_state=42, class_weight='balanced')
grid_search = GridSearchCV(model, param_grid, cv=3, scoring='f1', n_jobs=2, verbose=1)
grid_search.fit(X_train_sample, y_train_sample)
best_model = grid_search.best_estimator_
print(f"✅ Best parameters: {grid_search.best_params_}")

# === 9. Cross-Validation ===
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(best_model, X_train_full, y_train_full, cv=cv, scoring='f1')
print(f"\n🧪 Cross-validated F1 scores: {cv_scores}")
print(f"📈 Mean F1 score: {cv_scores.mean():.4f}")

# === 10. Train on Full Training Set ===
best_model.fit(X_train, y_train)

# === 11. Evaluation on Test Set ===
y_pred = best_model.predict(X_test)
print("\n🔍 Classification Report (Test Set):")
print(classification_report(y_test, y_pred, digits=4))

# === 12. Confusion Matrix ===
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=['Phishing', 'Legit'],
            yticklabels=['Phishing', 'Legit'])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix (Test Set)")
plt.tight_layout()
plt.savefig("confusion_matrix_test.png")
print("📊 Confusion matrix saved as confusion_matrix_test.png")

# === 13. Feature Importance ===
importances = pd.Series(best_model.feature_importances_, index=X_train.columns)
top_features_final = importances.sort_values(ascending=False).head(20)

plt.figure(figsize=(8, 6))
sns.barplot(x=top_features_final, y=top_features_final.index)
plt.title("Top 20 Feature Importances")
plt.tight_layout()
plt.savefig("feature_importance.png")
print("📊 Feature importance saved as feature_importance.png")

# === 14. Save Model, Scaler, and Columns ===
with open("model.pkl", "wb") as f:
    pickle.dump(best_model, f)

with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

with open("feature_columns.pkl", "wb") as f:
    pickle.dump(X_train.columns.tolist(), f)

print("✅ Model, scaler, train_medians, and feature column list saved.")

# === 15. Holdout Evaluation ===
y_holdout_pred = best_model.predict(X_holdout_full)
print("\n🔍 Classification Report (Holdout Set):")
print(classification_report(y_holdout, y_holdout_pred, digits=4))

cm_holdout = confusion_matrix(y_holdout, y_holdout_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm_holdout, annot=True, fmt="d", cmap="Blues",
            xticklabels=['Phishing', 'Legit'],
            yticklabels=['Phishing', 'Legit'])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix (Holdout Set)")
plt.tight_layout()
plt.savefig("confusion_matrix_holdout.png")
print("📊 Holdout confusion matrix saved as confusion_matrix_holdout.png")