from pathlib import Path
from .llm_client import LLMClient
from .monitoring_interpreter import MonitoringInterpreter
from .code_config_critic import CodeConfigCritic
from .data_pipeline_analyst import DataPipelineAnalyst
from .memory_store import MemoryStore

class Orchestrator:
    def __init__(self):
        self.llm = LLMClient()
        self.monitoring_interpreter = MonitoringInterpreter(self.llm)
        self.code_critic = CodeConfigCritic(self.llm)
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

        data_suggestion = self.data_analyst.suggest_data_fixes(diagnosis)
        print("[ORCH] Data suggestion:", data_suggestion)

        # In a real system, you'd ask for human approval.
        # For the demo, auto-accept config patch:
        self.code_critic.apply_patch(config_suggestion)

        incident = {
            "round_id": round_id,
            "diagnosis": diagnosis,
            "config_suggestion": config_suggestion,
            "data_suggestion": data_suggestion,
            "baseline_accuracy": baseline_acc,
            "current_accuracy": current_acc,
        }
        self.memory.append_incident(incident)
        print("[ORCH] Incident stored in memory.")