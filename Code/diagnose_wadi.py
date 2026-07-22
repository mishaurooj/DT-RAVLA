from pathlib import Path
import pandas as pd

PATH = Path(r"D:\other\DT-RAVLA\Dataset\WADI\WADI_attackdataLABLE.csv")

for header in [0, 1, 2]:
    print("\n" + "=" * 100)
    print("HEADER:", header)
    try:
        df = pd.read_csv(PATH, header=header, low_memory=False)
    except Exception as exc:
        print("ERROR:", exc)
        continue

    print("SHAPE:", df.shape)
    for i, col in list(enumerate(df.columns))[-25:]:
        s = df[col]
        vals = s.dropna().astype(str).str.strip().unique()[:12]
        print(i, repr(str(col)), "nunique=", s.nunique(dropna=True), "sample=", vals.tolist())
