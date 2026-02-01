from typing import TypedDict, Optional, Dict, Any


class AgentState(TypedDict, total=False):
    drift_report_path: str
    round_id: str
    baseline_accuracy: float
    current_accuracy: float

    diagnosis: Dict[str, Any]
    config_suggestion: Dict[str, Any]
    data_suggestion: Dict[str, Any]

    should_retrain: bool
    retrained: bool
    post_retrain_accuracy: Optional[float]

    new_data_acquired: bool
    new_data_samples: Optional[int]
    post_newdata_accuracy: Optional[float]