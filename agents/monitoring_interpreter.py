import json
import ast
from .llm_client import LLMClient
from .memory_store import MemoryStore


class MonitoringInterpreter:
    def __init__(self, llm: LLMClient):
        self.llm = llm
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

    def _parse_llm_json(self, text: str, fallback_report: dict) -> dict:
        # Debug: see what the model actually returned
        print("\n[MONITORING_INTERPRETER] Raw LLM response:")
        print(text)
        print()

        # 1) Try strict JSON
        try:
            return json.loads(text)
        except Exception:
            pass

        # 2) Try to extract a JSON-ish block
        try:
            start = text.index("{")
            end = text.rindex("}") + 1
            candidate = text[start:end]
            return json.loads(candidate)
        except Exception:
            pass

        # 3) Try Python literal (handles single quotes)
        try:
            obj = ast.literal_eval(candidate if "candidate" in locals() else text)
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass

        # 4) Fallback
        return {
            "issue_type": "data_drift",
            "suspect_features": list(fallback_report["psi_by_feature"].keys()),
            "severity": "medium",
            "reasoning": "Fallback diagnosis (LLM returned invalid JSON).",
        }

    def interpret(self, drift_report_path: str) -> dict:
        report = json.loads(open(drift_report_path).read())

        past_incidents = self.memory.load_all()
        memory_summary = self._summarize_memory(past_incidents)

        system_prompt = (
            "You are an ML reliability engineer. "
            "You diagnose drift issues using monitoring metrics and past incidents."
        )

        user_prompt = f"""
You are diagnosing ML drift.

You MUST:
1. Read the monitoring report carefully.
2. Use the exact PSI values from the report.
3. Use the exact accuracy_drop from the report.
4. Identify which features have PSI > 0.1.
5. Compare this round with past incidents.
6. Explain differences explicitly.
7. Return ONLY a JSON object.

Monitoring report:
{json.dumps(report, indent=2)}

Recent past incidents:
{memory_summary}

Return ONLY a JSON object with EXACTLY these keys:
- "issue_type": one of ["data_drift", "concept_drift", "pipeline_issue", "unknown"]
- "suspect_features": list of feature names with PSI > 0.1
- "severity": "low" | "medium" | "high"
- "reasoning": a short explanation that MUST reference:
    - the PSI values,
    - the accuracy_drop,
    - and differences vs past incidents.
"""

        response = self.llm.chat(system_prompt, user_prompt)
        diagnosis = self._parse_llm_json(response, report)
        return diagnosis