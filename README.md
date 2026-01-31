# Self-Healing ML Agentic Demo

This project demonstrates a minimal agentic system that:
- trains a simple ML model,
- monitors drift and performance degradation,
- uses LLM-based agents to interpret monitoring signals,
- suggests code/config/data fixes,
- stores incidents in memory,
- and improves suggestions over time.

Run the demo:

```bash
uv run python -m simulations.run_round