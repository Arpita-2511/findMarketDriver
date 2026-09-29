import pandas as pd

df = pd.read_csv("data/final_stock_dataset.csv")

corr = (
    df.corr(numeric_only=True)["Target"]
    .sort_values(ascending=False)
)

print("=" * 60)
print("CORRELATION WITH TARGET")
print("=" * 60)

print(corr)