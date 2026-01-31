import yaml
from pathlib import Path
from .llm_client import LLMClient

CONFIG_PATH = Path(__file__).parents[1] / "model" / "config.yaml"

class CodeConfigCritic:
    def __init__(self, llm: LLMClient):
        self.llm = llm

    def suggest_changes(self, diagnosis: dict) -> dict:
        cfg = yaml.safe_load(open(CONFIG_PATH))
        system_prompt = (
            "You are an expert MLOps engineer. "
            "Given a diagnosis and config.yaml, suggest minimal, safe changes."
        )
        user_prompt = f"""
Diagnosis:
{diagnosis}

Current config.yaml:
{yaml.dump(cfg)}

Return JSON with:
- config_patch: description of suggested changes (human-readable)
- rationale: why these changes help
"""
        response = self.llm.chat(system_prompt, user_prompt)
        # For now, just return a dummy suggestion
        return {
            "config_patch": "Lower psi_threshold from 0.3 to 0.2",
            "rationale": "More sensitive to drift after recent incident.",
        }

    def apply_patch(self, suggestion: dict):
        cfg = yaml.safe_load(open(CONFIG_PATH))
        # naive example: if suggestion mentions psi_threshold, change it
        if "psi_threshold" in suggestion["config_patch"]:
            cfg["monitoring"]["psi_threshold"] = 0.2
        with open(CONFIG_PATH, "w") as f:
            yaml.safe_dump(cfg, f)
        print("[CRITIC] Applied config patch:", suggestion["config_patch"])