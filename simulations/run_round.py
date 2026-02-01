from pathlib import Path
from model.train_model import train
from model.predict import predict
from monitoring.compute_metrics import compute_metrics
from agents.orchestrator import Orchestrator
from agents.memory_store import MEMORY_PATH
import shutil
from agents.workflow import build_workflow
from agents.memory_store import MemoryStore

DATA_DIR = Path("data")
MON_DIR = Path("monitoring")

# Clean memory before each simulation run: this ensures no prior state affects the demo
open(MEMORY_PATH, "w").close()
shutil.copy("data/train_original.csv", "data/train.csv")

from pathlib import Path
from model.train_model import train
from model.predict import predict
from monitoring.compute_metrics import compute_metrics
from agents.workflow import build_workflow
from agents.memory_store import MemoryStore

def run_demo():
    # === Round 0: Baseline training ===
    print("\n=== ROUND 0: BASELINE ===")
    baseline_acc = train(str(DATA_DIR / "train.csv"))
    predict(str(DATA_DIR / "test_round0.csv"), str(DATA_DIR / "test_round0_pred.csv"))

    # === Round 1: Drift ===
    print("\n=== ROUND 1: DRIFT ===")
    predict(str(DATA_DIR / "test_round1_drift.csv"), str(DATA_DIR / "test_round1_pred.csv"))

    report1 = compute_metrics(
        str(DATA_DIR / "test_round0_pred.csv"),
        str(DATA_DIR / "test_round1_pred.csv"),
        str(MON_DIR / "drift_report_round1.json"),
        baseline_acc=baseline_acc,
    )

    # Build LangGraph workflow
    workflow = build_workflow()

    # Prepare state for Round 1
    state1 = {
        "drift_report_path": str(MON_DIR / "drift_report_round1.json"),
        "round_id": "1",
        "baseline_accuracy": report1["baseline_accuracy"],
        "current_accuracy": report1["current_accuracy"],
    }

    # Run the graph
    final1 = workflow.invoke(state1)

    # === Round 2: Drift ===
    print("\n=== ROUND 2: DRIFT ===")
    predict(str(DATA_DIR / "test_round2_drift.csv"), str(DATA_DIR / "test_round2_pred.csv"))

    report2 = compute_metrics(
        str(DATA_DIR / "test_round0_pred.csv"),
        str(DATA_DIR / "test_round2_pred.csv"),
        str(MON_DIR / "drift_report_round2.json"),
        baseline_acc=baseline_acc,
    )

    # Prepare state for Round 2
    state2 = {
        "drift_report_path": str(MON_DIR / "drift_report_round2.json"),
        "round_id": "2",
        "baseline_accuracy": report2["baseline_accuracy"],
        "current_accuracy": report2["current_accuracy"],
    }

    # Run the graph
    final2 = workflow.invoke(state2)


if __name__ == "__main__":
    run_demo()