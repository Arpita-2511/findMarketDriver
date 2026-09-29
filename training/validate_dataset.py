import pandas as pd

# ==========================================
# Load Dataset
# ==========================================

df = pd.read_csv("data/final_stock_dataset.csv")

print("=" * 60)
print("DATASET VALIDATION")
print("=" * 60)

print("\nDataset Shape")
print(df.shape)

print("\nColumns")
print(df.columns.tolist())

print("\nMissing Values")
print(df.isnull().sum())

print("\nDuplicate Rows")
print(df.duplicated().sum())

print("\nData Types")
print(df.dtypes)

print("\nTarget Statistics")
print(df["Target"].describe())

print("\nLast Five Rows")
print(df.tail())