import json
from .llm_client import LLMClient
from .memory_store import MemoryStore


class ConfigCritic:
    def __init__(self, llm: LLMClient, config: dict):
        self.llm = llm
        self.config = config
        self.memory = MemoryStore()

    def _summarize_memory(self, incidents):
        if not incidents:
            return "No past incidents available."

        recent = incidents[-3:]
        return json.dumps(
            [
                {
                    "round_id": inc.get("round_id"),
                    "diagnosis": inc.get("diagnosis"),
                    "config_suggestion": inc.get("config_suggestion"),
                }
                for inc in recent
            ],
            indent=2,
        )

    def suggest_changes(self, diagnosis: dict) -> dict:
        past_incidents = self.memory.load_all()
        memory_summary = self._summarize_memory(past_incidents)

        system_prompt = (
            "You are an ML reliability engineer. "
            "Your job is to propose configuration changes and decide whether retraining is needed."
        )

        user_prompt = f"""
Current config:
{json.dumps(self.config, indent=2)}

Diagnosis for this round:
{json.dumps(diagnosis, indent=2)}

Use the historical summary to avoid repeating ineffective patches and to detect long-term trends:
{memory_summary}

Return ONLY a JSON object with EXACTLY these keys:
- "changes": a dictionary of config fields to update (e.g. {{"monitoring.psi_threshold": 0.2}})
- "rationale": a short explanation
- "should_retrain": true or false
"""

        response = self.llm.chat(system_prompt, user_prompt)

        try:
            return json.loads(response)
        except Exception:
            return {
                "changes": {},
                "rationale": "Fallback config suggestion (LLM returned invalid JSON).",
                "should_retrain": False,
            }

    def apply_patch(self, suggestion: dict):
        """Apply the structured patch to the in-memory config."""
        changes = suggestion.get("changes", {})
        for key, value in changes.items():
            section, field = key.split(".")
            if section in self.config and field in self.config[section]:
                self.config[section][field] = value
            else:
                print(f"[CRITIC] Warning: Unknown config key {key}")