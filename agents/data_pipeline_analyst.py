from .llm_client import LLMClient

class DataPipelineAnalyst:
    def __init__(self, llm: LLMClient):
        self.llm = llm

    def suggest_data_fixes(self, diagnosis: dict, memory_summary: dict) -> dict:
        system_prompt = (
            "You are a data engineer. "
            "Given a diagnosis, suggest data quality or pipeline checks."
        )
        user_prompt = f"""
Diagnosis:
{diagnosis}

Here is a summary of past data issues and drift patterns:
{memory_summary}

Return JSON with:
- data_checks: list of suggested new checks or validations
- rationale: short explanation
"""
        response = self.llm.chat(system_prompt, user_prompt)
        # Dummy fallback
        return {
            "data_checks": [
                "Add validation: income >= 0",
                "Monitor null rate for age",
            ],
            "rationale": "Common issues when drift appears on numeric features.",
        }