import pandas as pd
import joblib
from pathlib import Path
import yaml

CONFIG_PATH = Path(__file__).parent / "config.yaml"
MODEL_PATH = Path(__file__).parent / "model.joblib"

def load_config():
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)

def predict(test_path: str, output_path: str):
    cfg = load_config()
    df = pd.read_csv(test_path)
    model = joblib.load(MODEL_PATH)

    X = df[cfg["features"]["numeric"]]
    preds = model.predict(X)
    df["prediction"] = preds
    df.to_csv(output_path, index=False)
    return df

if __name__ == "__main__":
    predict("data/test_round0.csv", "data/test_round0_pred.csv")