import pandas as pd
import numpy as np

def inject_drift(input_path: str, output_path: str, shift: float = 10.0):
    df = pd.read_csv(input_path)
    if "age" in df.columns:
        df["age"] = df["age"] + shift
    if "income" in df.columns:
        df["income"] = df["income"] * (1 + shift / 100.0)
    df.to_csv(output_path, index=False)
    print(f"[DRIFT] Wrote drifted data to {output_path}")

if __name__ == "__main__":
    inject_drift("data/test_round0.csv", "data/test_round1_drift.csv", shift=10.0)