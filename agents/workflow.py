from langgraph.graph import StateGraph, END

from .graph_state import AgentState
from .nodes import (
    node_monitor,
    node_config_critic,
    node_data_analyst,
    node_retrain,
    node_new_data,
    node_memory,
)


def build_workflow():
    graph = StateGraph(AgentState)

    graph.add_node("monitor", node_monitor)
    graph.add_node("critic", node_config_critic)
    graph.add_node("data_analyst", node_data_analyst)
    graph.add_node("retrain", node_retrain)
    graph.add_node("new_data", node_new_data)
    graph.add_node("memory", node_memory)

    graph.set_entry_point("monitor")

    graph.add_edge("monitor", "critic")
    graph.add_edge("critic", "data_analyst")

    # After data_analyst: branch on should_retrain
    def branch_after_data_analyst(state: AgentState) -> str:
        return "retrain" if state.get("should_retrain", False) else "memory"

    graph.add_conditional_edges(
        "data_analyst",
        branch_after_data_analyst,
        {
            "retrain": "retrain",
            "memory": "memory",
        },
    )

    # After retrain: check if retraining helped
    def branch_after_retrain(state: AgentState) -> str:
        post = state.get("post_retrain_accuracy")
        current = state.get("current_accuracy")
        if post is None or current is None:
            return "memory"
        return "new_data" if post <= current else "memory"

    graph.add_conditional_edges(
        "retrain",
        branch_after_retrain,
        {
            "new_data": "new_data",
            "memory": "memory",
        },
    )

    # After new_data → always memory
    graph.add_edge("new_data", "memory")

    graph.add_edge("memory", END)

    return graph.compile()