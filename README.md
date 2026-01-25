# AI Agent Evaluation and Simulation Toolkit

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![DeepEval](https://img.shields.io/badge/built%20on-DeepEval-brightgreen)](https://deepeval.com/)

Stress-test AI agents with simulated user personas and LLM-as-judge evaluation. Plug in your agent, define scenarios in YAML, simulate conversations, get evaluation reports.

<!-- ## Demo

![CLI Demo](assets/cli_demo.gif)

## Sample Output

![Report Preview](assets/report_preview.png) -->

---

## Quick Start

### 1. Install

```bash
git clone https://github.com/emfrg/agent-eval-toolkit
cd agent-eval-toolkit

python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# core
pip install -e .

# for the example service
pip install -e ".[service]"

# for tests (optional)
pip install -e ".[dev]"
```

### 2. Set up environment

```bash
cp .env.example .env
```

Edit `.env` and add your OpenAI API key:

```
OPENAI_API_KEY=sk-your-key-here
```

> Required for user simulation + LLM-as-judge evaluation. The mock agent itself makes no API calls.

### 3. Try the demo

```bash
agent-eval-toolkit quickstart
```

This runs a complete simulation + evaluation with an example agent. Check `quickstart_output/` for results.

---

## How to Use With Your Agent

### Architecture Patterns

This framework supports two integration patterns:

1. **Service + Adapter Pattern (for existing projects)**:

   - Your agent runs as a separate service (FastAPI, Flask, etc.)
   - Create an adapter file that connects to your service
   - Example: `examples/example_agent/` (service) + `example_agent_adapter.py` (adapter)

2. **Standalone Pattern**:
   - All agent logic in a single file
   - Direct API calls within the callback function
   - Example: `examples/mock_agent.py` (mock for demonstration)

### Try the Example First

Before creating your own adapter, test with the included example:

**Terminal 1 - Start the example agent service:**
```bash
python examples/example_agent/run.py
```

**Terminal 2 - Run simulations:**
```bash
agent-eval-toolkit simulate --agent examples/example_agent_adapter.py
```

Then evaluate:
```bash
agent-eval-toolkit evaluate
```

Check `reports/` for results. See [examples/example_agent/README.md](examples/example_agent/README.md) for details.

---

### Step 1: Create your agent adapter

Create `my_agent_adapter.py` (following the recommended Service + Adapter pattern):

```python
from typing import List
from deepeval.test_case import Turn
import httpx

async def agent_callback(input: str, turns: List[Turn], thread_id: str) -> Turn:
    """Adapter that connects your agent service to the eval framework."""

    # Connect to your agent service
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://localhost:8000/chat",  # Your agent service URL
            json={
                "message": input,
                "thread_id": thread_id,
                "history": [{"role": t.role, "content": t.content} for t in turns]
            },
            timeout=30.0
        )

        data = response.json()
        return Turn(role="assistant", content=data["message"])
```

### Step 2: Run simulations

```bash
agent-eval-toolkit simulate --agent my_agent_adapter.py
```

> **Note:** Your agent service must be running before you run simulations. The adapter connects to your service via HTTP.

### Step 3: Evaluate

```bash
agent-eval-toolkit evaluate
```

### Step 4: Check reports

Check `reports/` for evaluation results.

---

## Customization

### Define your personas

Edit `config/personas.yaml`:

```yaml
personas:
  busy_professional:
    scenario: "User wants quick pricing info"
    expected_outcome: "Get answer in under 5 messages"
    user_description: "Time-poor professional who values concise answers"
    conversations: 10
    max_turns: 12
```

### Custom evaluation criteria

Edit `config/judge_prompt.md` to define what "good" looks like for your agent.

### Custom metrics

Add your own metrics programmatically:

```python
from src.evaluation.judge import JudgeEvaluator
from src.metrics import ConversationLengthMetric

evaluator = JudgeEvaluator(
    custom_metrics=[
        ConversationLengthMetric(max_turns=12)
    ]
)
```

See [src/metrics/custom_examples.py](src/metrics/custom_examples.py) for examples.

<!--
## Case Study

We used this toolkit internally to evaluate a brainstorming agent before deployment.

One finding: a "disinterested" persona took 3x more turns to produce an idea compared to engaged users. This led us to adjust the agent to be more proactive when it detects low engagement — a fix we wouldn't have caught without simulation.
-->

---

## CLI Reference

### simulate

```bash
agent-eval-toolkit simulate --agent <path>

Options:
  --agent PATH         Your agent file (required)
  --config PATH        Persona config (default: config/personas.yaml)
  --output PATH        Output directory (default: logs)
  --format TEXT        json, jsonl, or both (default: jsonl)
  --max-concurrent N   Max parallel sims (default: 10)
```

### evaluate

```bash
agent-eval-toolkit evaluate

Options:
  --logs-dir PATH      Conversation logs (default: logs)
  --output PATH        Reports directory (default: reports)
  --prompt PATH        Judge prompt (default: config/judge_prompt.md)
  --threshold FLOAT    Pass/fail threshold (default: 0.7)
```

### quickstart

Runs complete demo with mock agent
NOTE: simulated users and judge still use API calls

```bash
agent-eval-toolkit quickstart
```

---

## Project Structure

```
agent-eval-toolkit/
├── examples/
│   ├── example_agent/             # Example LangGraph agent service
│   │   ├── agent_service.py       # FastAPI application
│   │   ├── agent.py               # LangGraph agent logic
│   │   ├── models.py              # Pydantic request/response models
│   │   ├── tools.py               # Agent tools
│   │   └── run.py                 # Service launcher
│   ├── example_agent_adapter.py   # HTTP adapter for the service
│   └── custom_metrics_example.py  # Custom metrics examples
├── config/
│   ├── personas.yaml              # User personas
│   └── judge_prompt.md            # Evaluation criteria
├── src/
│   ├── core/                      # Core types & logging
│   ├── simulation/                # Conversation simulation
│   ├── evaluation/                # LLM-as-a-judge
│   └── metrics/                   # Custom metrics
├── tests/                         # Test suite
└── pyproject.toml
```

---

## Advanced: Programmatic Usage

If you need more control:

```python
import asyncio
from src.simulation.config_loader import load_personas_from_yaml
from src.simulation.runner import SimulationRunner
from src.evaluation.judge import JudgeEvaluator
from src.core.logging import ConversationLogger
from src.evaluation.report import ReportGenerator

# Your agent
async def my_agent(input, turns, thread_id):
    # ... your logic
    return Turn(role="assistant", content="response")

async def main():
    goldens, configs = load_personas_from_yaml("config/personas.yaml")

    runner = SimulationRunner(agent_callback=my_agent)
    test_cases = await runner.run_simulations(goldens, configs)

    logger = ConversationLogger(output_dir="logs")
    logger.save_as_jsonl(test_cases)

    evaluator = JudgeEvaluator(threshold=0.7)
    results = evaluator.evaluate_conversations(test_cases)

    summary = evaluator.get_summary(results)
    print(f"Pass rate: {summary['pass_rate']*100:.1f}%")

asyncio.run(main())
```

<!--
## Contributing

PRs welcome! Areas of interest:

- Standardized adapters pattern
- Standardized custom metrics
- Better reporting/visualizations
-->
