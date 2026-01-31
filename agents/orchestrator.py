from pathlib import Path

import yaml, json
from .llm_client import LLMClient
from .monitoring_interpreter import MonitoringInterpreter
from .code_config_critic import CodeConfigCritic
from .config_critic import ConfigCritic
from .data_pipeline_analyst import DataPipelineAnalyst
from .memory_store import MemoryStore
from agents.retrainer import Retrainer
from agents.data_generator import SyntheticDataGenerator
import pandas as pd

class Orchestrator:
    def __init__(self):
        self.config_path = Path("model/config.yaml")
        self.data_generator = SyntheticDataGenerator()
        self.config = yaml.safe_load(open(self.config_path))
        self.llm = LLMClient()
        self.retrainer = Retrainer()
        self.monitoring_interpreter = MonitoringInterpreter(self.llm)
        #self.code_critic = CodeConfigCritic(self.llm)
        self.code_critic = ConfigCritic(self.llm, self.config)
        self.data_analyst = DataPipelineAnalyst(self.llm)
        self.memory = MemoryStore()

    def run_round(
        self,
        drift_report_path: str,
        round_id: str,
        baseline_acc: float,
        current_acc: float,
    ):
        print(f"\n[ORCH] === Round {round_id} ===")
        diagnosis = self.monitoring_interpreter.interpret(drift_report_path)
        print("[ORCH] Diagnosis:", diagnosis)

        config_suggestion = self.code_critic.suggest_changes(diagnosis)
        print("[ORCH] Config suggestion:", config_suggestion)

        should_retrain = config_suggestion.get("should_retrain", False)
        print("[ORCH] Should retrain:", should_retrain)

        # === Prevent infinite retraining loops ===
        last_incident = self.memory.load_last()
        if should_retrain and last_incident and last_incident.get("retrained", False):
            print("[ORCH] Skipping retraining — already retrained last round.")
            should_retrain = False        

        data_suggestion = self.data_analyst.suggest_data_fixes(diagnosis)
        print("[ORCH] Data suggestion:", data_suggestion)

        # In a real system, you'd ask for human approval.
        # For the demo, auto-accept config patch:
        self.code_critic.apply_patch(config_suggestion)

        with open(self.config_path, "w") as f:
            yaml.safe_dump(self.config, f)

        incident = {
            "round_id": round_id,
            "diagnosis": diagnosis,
            "config_suggestion": config_suggestion,
            "data_suggestion": data_suggestion,
            "baseline_accuracy": baseline_acc,
            "current_accuracy": current_acc,
            "should_retrain": should_retrain,
        }
        if should_retrain:
            print("[ORCH] Retraining triggered...")
            new_acc = self.retrainer.retrain()

            # Store retraining result in the incident
            incident["retrained"] = True
            incident["post_retrain_accuracy"] = new_acc

            # === Check if retraining helped ===
            if new_acc <= current_acc:
                print("[ORCH] Retraining ineffective — acquiring new data...")

                # Load drift report
                drift_report = json.loads(open(drift_report_path).read())

                # Generate synthetic new data
                new_data = self.data_generator.generate(drift_report, n_samples=500)

                # Append to training set
                train_df = pd.read_csv("data/train.csv")
                updated_train = pd.concat([train_df, new_data], ignore_index=True)
                updated_train.to_csv("data/train.csv", index=False)

                incident["new_data_acquired"] = True
                incident["new_data_samples"] = len(new_data)

                # Retrain again on expanded dataset
                print("[ORCH] Retraining with new data...")
                improved_acc = self.retrainer.retrain()
                incident["post_newdata_accuracy"] = improved_acc
            else:
                incident["new_data_acquired"] = False
        else:
            incident["retrained"] = False
            incident["new_data_acquired"] = False

        self.memory.append_incident(incident)
        print("[ORCH] Incident stored in memory.")