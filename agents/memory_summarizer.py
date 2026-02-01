from typing import List, Dict, Any
from .llm_client import LLMClient

class MemorySummarizer:
    def __init__(self):
        self.llm = LLMClient()

    def summarize(self, incidents: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not incidents:
            return {"summary": "No past incidents.", "patterns": [], "recommendations": []}

        prompt = f"""
You are an ML reliability analyst. Summarize the following incidents into a compact JSON structure.

INCIDENTS:
{incidents}

Provide:
- drift_trends: recurring drift patterns
- retraining_effectiveness: when retraining helped or failed
- config_changes: notable config patches
- data_quality_issues: recurring data issues
- recommendations: 3 short actionable suggestions

Respond ONLY with valid JSON.
"""

        system_prompt = (
    "You are an ML reliability analyst. "
    "Summarize historical incidents into a compact JSON structure."
)

        response = self.llm.chat(system_prompt, prompt)
        return response