import json
from .llm_client import LLMClient

class MonitoringInterpreter:
    def __init__(self, llm: LLMClient):
        self.llm = llm

    def interpret(self, drift_report_path: str) -> dict:
        report = json.loads(open(drift_report_path).read())
        system_prompt = (
            "You are an ML reliability engineer. "
            "Given monitoring metrics, you diagnose issues in structured JSON."
        )
        user_prompt = f"""
Monitoring report:
{json.dumps(report, indent=2)}

Return a JSON object with:
- issue_type: one of ["data_drift", "concept_drift", "pipeline_issue", "unknown"]
- suspect_features: list of feature names
- severity: "low" | "medium" | "high"
- reasoning: short explanation
"""
        response = self.llm.chat(system_prompt, user_prompt)
        # In a real impl, parse JSON from response. For now, naive fallback:
        try:
            diagnosis = json.loads(response)
        except Exception:
            diagnosis = {
                "issue_type": "data_drift",
                "suspect_features": list(report["psi_by_feature"].keys()),
                "severity": "medium",
                "reasoning": "Fallback diagnosis (LLM not wired).",
            }
        return diagnosis