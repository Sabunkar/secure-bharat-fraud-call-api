import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from feature_extractor import extract_features

CSV_PATH = "fraud_numbers.csv"
MODEL_PATH = "fraud_model.pkl"

def build_dataframe(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, dtype={"phone_number": str})
    rows = []
    for _, row in df.iterrows():
        feats = extract_features(row["phone_number"])
        feats["label"] = 1 if str(row["label"]).strip().lower() == "scam" else 0
        rows.append(feats)
    return pd.DataFrame(rows)

def main():
    df = build_dataframe(CSV_PATH)
    X = df.drop(columns=["label"])
    y = df["label"]

    model = RandomForestClassifier(n_estimators=60, random_state=42)
    model.fit(X, y)

    preds = model.predict(X)
    print(classification_report(y, preds, target_names=["legit","scam"]))

    joblib.dump(model, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")

if __name__ == "__main__":
    main()
