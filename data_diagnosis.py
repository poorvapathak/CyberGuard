import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Load dataset
df = pd.read_csv("PhiUSIIL_Phishing_URL_Dataset.csv")

# Step 2: Preview data
print("\n🔹 First 5 rows:")
print(df.head())

# Step 3: Show column names
print("\n🔹 Column names:")
print(df.columns.tolist())

# Step 4: Check for null/missing values
print("\n🔹 Null value counts:")
print(df.isnull().sum())

# Step 5: Data types
print("\n🔹 Data types:")
print(df.dtypes)

# Step 6: Unique value counts for target label
label_col = [col for col in df.columns if 'label' in col.lower() or 'class' in col.lower()]
if label_col:
    label_col = label_col[0]
    print(f"\n🔹 Value counts for target column: {label_col}")
    print(df[label_col].value_counts())
else:
    print("\n⚠️ No column with 'label' found. Please check column names.")

# Step 7: Plot target distribution
if label_col:
    sns.countplot(data=df, x=label_col)
    plt.title("Class Distribution")
    plt.xlabel("Label")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig("label_distribution.png")
    print("\n📊 Saved class distribution plot as 'label_distribution.png'")

# Step 8: Detect outliers with boxplots (for a few sample columns)
sample_cols = df.select_dtypes(include='number').columns[:5]  # Take first 5 numeric columns
for col in sample_cols:
    plt.figure(figsize=(5, 3))
    sns.boxplot(x=df[col])
    plt.title(f"Outlier check: {col}")
    plt.tight_layout()
    plt.savefig(f"outlier_{col}.png")
    print(f"📊 Saved boxplot for {col} as 'outlier_{col}.png'")

