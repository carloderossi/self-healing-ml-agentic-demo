import json
import pandas as pd

from .graph_state import AgentState
from .monitoring_interpreter import MonitoringInterpreter
from .config_critic import ConfigCritic
from .data_pipeline_analyst import DataPipelineAnalyst
from .memory_store import MemoryStore
from .retrainer import Retrainer
from .data_generator import SyntheticDataGenerator
from .memory_summarizer import MemorySummarizer
from pathlib import Path
import yaml


# Shared singletons (mirroring your Orchestrator.__init__)
_config_path = Path("model/config.yaml")
_config = yaml.safe_load(open(_config_path))

_llm = None
_monitor = None
_critic = None
_analyst = None
_memory = None
_retrainer = None
_generator = None

_summarizer = None

def _ensure_singletons():
    global _llm, _monitor, _critic, _analyst, _memory, _retrainer, _generator
    from .llm_client import LLMClient

    global _summarizer
    if _summarizer is None:
        _summarizer = MemorySummarizer()

    if _llm is None:
        _llm = LLMClient()
        from .monitoring_interpreter import MonitoringInterpreter
        from .config_critic import ConfigCritic
        from .data_pipeline_analyst import DataPipelineAnalyst
        from .memory_store import MemoryStore
        from .retrainer import Retrainer
        from .data_generator import SyntheticDataGenerator

        _monitor = MonitoringInterpreter(_llm)
        _critic = ConfigCritic(_llm, _config)
        _analyst = DataPipelineAnalyst(_llm)
        _memory = MemoryStore()
        _retrainer = Retrainer()
        _generator = SyntheticDataGenerator()


def node_monitor(state: AgentState) -> AgentState:
    _ensure_singletons()
    print(f"\n[GRAPH] === Round {state['round_id']} ===")
    diagnosis = _monitor.interpret(state["drift_report_path"])
    print("[GRAPH] Diagnosis:", diagnosis)
    state["diagnosis"] = diagnosis
    return state


def node_config_critic(state: AgentState) -> AgentState:
    _ensure_singletons()
    config_suggestion = _critic.suggest_changes(state["diagnosis"])
    print("[GRAPH] Config suggestion:", config_suggestion)
    state["config_suggestion"] = config_suggestion

    should_retrain = config_suggestion.get("should_retrain", False)
    print("[GRAPH] Should retrain:", should_retrain)

    # Prevent infinite retraining loops (same as orchestrator)
    last_incident = _memory.load_last()
    if should_retrain and last_incident and last_incident.get("retrained", False):
        print("[GRAPH] Skipping retraining — already retrained last round.")
        should_retrain = False

    state["should_retrain"] = should_retrain

    # Apply config patch + persist
    _critic.apply_patch(config_suggestion)
    with open(_config_path, "w") as f:
        yaml.safe_dump(_config, f)

    return state


def node_data_analyst(state: AgentState) -> AgentState:
    _ensure_singletons()
    data_suggestion = _analyst.suggest_data_fixes(
        state["diagnosis"],
        state.get("memory_summary", {})
    )
    print("[GRAPH] Data suggestion:", data_suggestion)
    state["data_suggestion"] = data_suggestion
    return state


def node_retrain(state: AgentState) -> AgentState:
    _ensure_singletons()
    print("[GRAPH] Retraining triggered...")
    new_acc = _retrainer.retrain()
    state["retrained"] = True
    state["post_retrain_accuracy"] = new_acc
    return state


def node_new_data(state: AgentState) -> AgentState:
    _ensure_singletons()
    print("[GRAPH] Retraining ineffective — acquiring new data...")

    drift_report_path = state["drift_report_path"]
    drift_report = json.loads(open(drift_report_path).read())

    new_data = _generator.generate(drift_report, n_samples=500)

    train_df = pd.read_csv("data/train.csv")
    updated_train = pd.concat([train_df, new_data], ignore_index=True)
    updated_train.to_csv("data/train.csv", index=False)

    state["new_data_acquired"] = True
    state["new_data_samples"] = len(new_data)

    print("[GRAPH] Retraining with new data...")
    improved_acc = _retrainer.retrain()
    state["post_newdata_accuracy"] = improved_acc

    return state


def node_memory(state: AgentState) -> AgentState:
    _ensure_singletons()
    incident = {
        "round_id": state["round_id"],
        "diagnosis": state["diagnosis"],
        "config_suggestion": state["config_suggestion"],
        "data_suggestion": state["data_suggestion"],
        "baseline_accuracy": state["baseline_accuracy"],
        "current_accuracy": state["current_accuracy"],
        "should_retrain": state["should_retrain"],
        "retrained": state.get("retrained", False),
        "post_retrain_accuracy": state.get("post_retrain_accuracy"),
        "new_data_acquired": state.get("new_data_acquired", False),
        "new_data_samples": state.get("new_data_samples"),
        "post_newdata_accuracy": state.get("post_newdata_accuracy"),
    }
    _memory.append_incident(incident)
    print("[GRAPH] Incident stored in memory.")
    return state

def node_summarize_memory(state: AgentState) -> AgentState:
    _ensure_singletons()
    incidents = _memory.load_all()
    summary = _summarizer.summarize(incidents)
    print("[GRAPH] Memory summary generated.")
    state["memory_summary"] = summary
    return state