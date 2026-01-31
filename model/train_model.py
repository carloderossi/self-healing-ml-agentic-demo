import pandas as pd
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
from pathlib import Path

CONFIG_PATH = Path(__file__).parent / "config.yaml"
MODEL_PATH = Path(__file__).parent / "model.joblib"

def load_config():
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)

def train(train_path: str):
    print(f"[TRAIN] Starting training with data from {train_path}")
    cfg = load_config()
    df = pd.read_csv(train_path)

    X = df[cfg["features"]["numeric"]]
    y = df[cfg["target"]]

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_val)
    acc = accuracy_score(y_val, y_pred)
    print(f"[TRAIN] Validation accuracy: {acc:.3f}")

    joblib.dump(model, MODEL_PATH)
    return acc

if __name__ == "__main__":
    train("data/train.csv")