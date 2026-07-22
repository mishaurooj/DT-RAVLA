import pandas as pd

df = pd.read_csv(r"D:\other\DT-RAVLA\Dataset\SWaT\merged.csv", low_memory=False)

print("Shape:", df.shape)
print("\nFirst 20 columns:")
print(df.columns[:20].tolist())

print("\nLast 20 columns:")
print(df.columns[-20:].tolist())

print("\nData types:")
print(df.dtypes.tail(20))

print("\nFirst 5 rows:")
print(df.head())