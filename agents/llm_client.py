import yaml
import requests
from pathlib import Path


CONFIG_PATH = Path(__file__).parents[1] / "model" / "config.yaml"


class LLMClient:
    def __init__(self, config_path: Path = CONFIG_PATH):
        self.config = self._load_config(config_path)
        self.provider = self.config["llm"]["provider"]
        self.model = self.config["llm"]["model"]
        self.temperature = self.config["llm"].get("temperature", 0.2)
        self.max_tokens = self.config["llm"].get("max_tokens", 512)
        self.endpoint = self.config["llm"].get("endpoint")

    def _load_config(self, path: Path):
        with open(path, "r") as f:
            return yaml.safe_load(f)

    def chat(self, system_prompt: str, user_prompt: str) -> str:
        """
        Unified chat interface for all providers.
        """
        if self.provider == "ollama":
            return self._chat_ollama(system_prompt, user_prompt)

        elif self.provider == "openai":
            return self._chat_openai(system_prompt, user_prompt)

        elif self.provider == "kimi":
            return self._chat_kimi(system_prompt, user_prompt)

        elif self.provider == "deepseek":
            return self._chat_deepseek(system_prompt, user_prompt)

        else:
            raise ValueError(f"Unknown LLM provider: {self.provider}")

    # -----------------------------
    # OLLAMA IMPLEMENTATION
    # -----------------------------
    def _chat_ollama(self, system_prompt: str, user_prompt: str) -> str:
        url = f"{self.endpoint}/api/chat"

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_tokens,
            },
            "stream": False,
        }

        try:
            response = requests.post(url, json=payload, timeout=60)
            response.raise_for_status()
            data = response.json()
            return data.get("message", {}).get("content", "")
        except Exception as e:
            return f"[LLM ERROR] Ollama request failed: {e}"

    # -----------------------------
    # PLACEHOLDERS FOR FUTURE PROVIDERS
    # -----------------------------
    def _chat_openai(self, system_prompt: str, user_prompt: str) -> str:
        return "[OpenAI provider not yet implemented]"

    def _chat_kimi(self, system_prompt: str, user_prompt: str) -> str:
        return "[Kimi provider not yet implemented]"

    def _chat_deepseek(self, system_prompt: str, user_prompt: str) -> str:
        return "[DeepSeek provider not yet implemented]"