import os, pandas as pd, joblib
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# ✅ Correct dataset path
data_dir = Path("dataset")  # this resolves to: C:\Users\Vaishnavi\sign_language_translator\dataset

all_data = []
all_labels = []

print(f"📂  Looking for CSVs in:  {data_dir.resolve()}")
for file in os.listdir(data_dir):
    if not file.endswith(".csv"):
        continue

    file_path = data_dir / file

    try:
        df = pd.read_csv(file_path, header=None)
        if df.empty:
            print(f"❌ Corrupt or empty: {file} – skipped")
            continue

        label = file.split("_")[0]
        all_data.extend(df.values.tolist())
        all_labels.extend([label] * len(df))

    except pd.errors.EmptyDataError:
        print(f"❌ Couldn’t read: {file} – empty or broken.")
        continue

print(f"📝  Loaded {len(all_data)} samples across {len(set(all_labels))} gestures\n")

# 🎓 Train model
X_train, X_test, y_train, y_test = train_test_split(all_data, all_labels, test_size=0.2, random_state=42)
model = RandomForestClassifier()
model.fit(X_train, y_train)

# 📊 Evaluate
predictions = model.predict(X_test)
print("🔍  Validation report:\n", classification_report(y_test, predictions))

# 💾 Save model
models_dir = Path("models")
models_dir.mkdir(exist_ok=True)
joblib.dump(model, models_dir / "gesture_model.pkl")
print(f"\n✅  Model saved to:  {models_dir / 'gesture_model.pkl'}")
