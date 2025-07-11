# dataset_check.py – show per‑class counts & a quick self‑test
import glob, os, pandas as pd, joblib, numpy as np, collections
from pathlib import Path

data_dir = Path("dataset")
files    = glob.glob(str(data_dir / "*.csv"))
print("Found", len(files), "CSV files")

counts = collections.Counter()
rows, labels = [], []

for f in files:
    lab = Path(f).stem.split("_")[0]
    try:
        df = pd.read_csv(f, header=None)
    except pd.errors.EmptyDataError:
        print("⚠️ empty:", f); continue
    counts[lab] += len(df)
    rows.append(df.values); labels += [lab]*len(df)

print("\nPer‑class row counts:")
for lab, n in counts.items():
    print(f"  {lab:<8} {n}")

X = np.vstack(rows); y = np.array(labels)

# quick fit to see which classes blend
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=.2, stratify=y, random_state=1)
clf = RandomForestClassifier().fit(X_tr, y_tr)
print("\nQuick self‑test accuracy:", clf.score(X_te, y_te))
