from pathlib import Path
from model.train_model import train
from model.predict import predict
from monitoring.compute_metrics import compute_metrics
from agents.orchestrator import Orchestrator
from agents.memory_store import MEMORY_PATH
import shutil

DATA_DIR = Path("data")
MON_DIR = Path("monitoring")

# Clean memory before each simulation run: this ensures no prior state affects the demo
open(MEMORY_PATH, "w").close()
shutil.copy("data/train_original.csv", "data/train.csv")

def run_demo():
    # Round 0: baseline
    print("\n=== ROUND 0: BASELINE ===")
    baseline_acc = train(str(DATA_DIR / "train.csv"))
    predict(str(DATA_DIR / "test_round0.csv"), str(DATA_DIR / "test_round0_pred.csv"))

    # Round 1: drift
    print("\n=== ROUND 1: DRIFT ===")
    predict(str(DATA_DIR / "test_round1_drift.csv"), str(DATA_DIR / "test_round1_pred.csv"))
    report1 = compute_metrics(
        str(DATA_DIR / "test_round0_pred.csv"),
        str(DATA_DIR / "test_round1_pred.csv"),
        str(MON_DIR / "drift_report_round1.json"),
        baseline_acc=baseline_acc,
    )

    orch = Orchestrator()
    orch.run_round(
        drift_report_path=str(MON_DIR / "drift_report_round1.json"),
        round_id="1",
        baseline_acc=report1["baseline_accuracy"],
        current_acc=report1["current_accuracy"],
    )

    # Round 2: another drift (you can generate a second drifted file)
    print("\n=== ROUND 2: DRIFT ===")
    predict(str(DATA_DIR / "test_round2_drift.csv"), str(DATA_DIR / "test_round2_pred.csv"))
    report2 = compute_metrics(
        str(DATA_DIR / "test_round0_pred.csv"),
        str(DATA_DIR / "test_round2_pred.csv"),
        str(MON_DIR / "drift_report_round2.json"),
        baseline_acc=baseline_acc,
    )

    orch.run_round(
        drift_report_path=str(MON_DIR / "drift_report_round2.json"),
        round_id="2",
        baseline_acc=report2["baseline_accuracy"],
        current_acc=report2["current_accuracy"],
    )

if __name__ == "__main__":
    run_demo()