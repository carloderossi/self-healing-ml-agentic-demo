import json
from pathlib import Path
from typing import Dict, Any, List

MEMORY_PATH = Path(__file__).parent / "memory.jsonl"

class MemoryStore:
    def __init__(self, path: Path = MEMORY_PATH):
        self.path = path
        self.path.touch(exist_ok=True)

    def append_incident(self, incident: Dict[str, Any]):
        with self.path.open("a") as f:
            f.write(json.dumps(incident) + "\n")

    def load_all(self) -> List[Dict[str, Any]]:
        incidents = []
        with self.path.open("r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                incidents.append(json.loads(line))
        return incidents
    
    def load_last(self) -> Dict[str, Any] | None:
        """Return the most recent incident, or None if memory is empty."""
        incidents = self.load_all()
        if not incidents:
            return None
        return incidents[-1]

