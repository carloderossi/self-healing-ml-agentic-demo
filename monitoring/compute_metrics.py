import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
import json
from pathlib import Path
import yaml

CONFIG_PATH = Path(__file__).parents[1] / "model" / "config.yaml"

def load_config():
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)
    
'''
    PSI — Population Stability Index — is a statistical measure used in ML monitoring to quantify 
    how much a feature’s distribution has shifted between two datasets, typically:
    - Baseline (training data or a stable past period)
    - Current (new production data)
    It’s one of the most widely used drift metrics in risk modeling, credit scoring, 
    and production ML systems because it’s simple, interpretable, and works well for both continuous and categorical features.
'''
def population_stability_index(expected, actual, bins=10):
    e_perc, _ = np.histogram(expected, bins=bins)
    a_perc, _ = np.histogram(actual, bins=bins)
    e_perc = e_perc / len(expected)
    a_perc = a_perc / len(actual)
    e_perc = np.where(e_perc == 0, 1e-6, e_perc)
    a_perc = np.where(a_perc == 0, 1e-6, a_perc)
    psi = np.sum((a_perc - e_perc) * np.log(a_perc / e_perc))
    return float(psi)

def compute_metrics(
    baseline_path: str,
    current_path: str,
    drift_report_path: str,
    baseline_acc: float,
):
    cfg = load_config()
    base = pd.read_csv(baseline_path)
    curr = pd.read_csv(current_path)

    # assume both have target + prediction
    acc = accuracy_score(curr[cfg["target"]], curr["prediction"])
    acc_drop = baseline_acc - acc

    psi_by_feature = {}
    for feat in cfg["features"]["numeric"]:
        psi_by_feature[feat] = population_stability_index(
            base[feat].values, curr[feat].values
        )

    report = {
        "baseline_accuracy": baseline_acc,
        "current_accuracy": acc,
        "accuracy_drop": acc_drop,
        "psi_by_feature": psi_by_feature,
    }

    Path(drift_report_path).write_text(json.dumps(report, indent=2))
    print(f"[MONITORING] Report written to {drift_report_path}")
    return report

if __name__ == "__main__":
    compute_metrics(
        "data/test_round0_pred.csv",
        "data/test_round1_pred.csv",
        "monitoring/drift_report_round1.json",
        baseline_acc=0.9,
    )