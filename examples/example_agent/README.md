# Example Agent Service

This directory contains a **reference FastAPI service** that exposes a LangGraph agent over HTTP. This demonstrates the **correct architectural pattern** for separating concerns:

- **Service** (this directory): Handles LLM API calls, tool execution, business logic
- **Adapter** (`example_agent_adapter.py`): Handles HTTP calls and format conversion

## Architecture

```
┌─────────────────────────────────────────────┐
│ Agent Service (FastAPI + LangGraph)         │
│ - LLM API integration                       │
│ - Tool execution                            │
│ - Business logic                            │
│ - Can be deployed independently             │
└─────────────────────────────────────────────┘
                    ↓ HTTP
┌─────────────────────────────────────────────┐
│ Adapter                                     │
│ - HTTP client                               │
│ - Format conversion (JSON → Turn)           │
│ - Error handling                            │
└─────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────┐
│ LLM Evals Framework                         │
│ - Simulation runner                         │
│ - Evaluation                                │
└─────────────────────────────────────────────┘
```

## Files

- **`agent_service.py`**: FastAPI application with REST endpoints
- **`agent.py`**: LangGraph agent logic with StateGraph and tool calling
- **`tools.py`**: Agent tools decorated with `@tool` (pricing info, docs search)
- **`models.py`**: Pydantic models for request/response validation
- **`run.py`**: Launcher script for the service
- **`README.md`**: This file

## Installation

### 1. Install Service Dependencies

```bash
# From project root
pip install -e ".[service]"
```

This installs:
- `langgraph` - Agent framework with StateGraph
- `langchain` & `langchain-openai` - Core LLM integration
- `fastapi` & `uvicorn` - Web framework
- `httpx` - HTTP client for the adapter

### 2. Set OpenAI API Key

```bash
# Add to .env file in project root
echo "OPENAI_API_KEY=sk-your-key-here" >> .env
```

## Usage

### Starting the Service

```bash
# From project root
python examples/example_agent/run.py
```

The service will start at `http://localhost:5555` by default.

To use a different port:
```bash
PORT=3000 python examples/example_agent/run.py
```

### API Documentation

Once running, visit:
- **Swagger UI**: http://localhost:5555/docs
- **ReDoc**: http://localhost:5555/redoc

### Testing the Service

#### Using curl:

```bash
# Health check
curl http://localhost:5555/health

# Chat request
curl -X POST http://localhost:5555/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "What are your pricing plans?",
    "conversation_id": "test-123",
    "history": []
  }'
```

#### Using Python:

```bash
# Test the adapter
python examples/example_agent_adapter.py
```

#### Using the Framework:

```bash
# Run simulations using the adapter
llm-evals-starter simulate \
  --agent examples/example_agent_adapter.py \
  --config config/personas.yaml
```

## API Endpoints

### `GET /health`

Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "agent_configured": true
}
```

### `POST /chat`

Chat with the agent.

**Request:**
```json
{
  "message": "What's the Pro plan pricing?",
  "conversation_id": "user-123",
  "history": [
    {"role": "user", "content": "Hello"},
    {"role": "assistant", "content": "Hi! How can I help?"}
  ]
}
```

**Response:**
```json
{
  "message": "The Pro plan costs $25/month and includes...",
  "conversation_id": "user-123",
  "tool_calls": [
    {
      "name": "get_pricing_info",
      "arguments": {"plan_name": "pro"},
      "result": null
    }
  ],
  "metadata": {
    "model": "gpt-4o",
    "temperature": 0.7
  }
}
```

## Available Tools

The agent has access to these tools:

### 1. `get_pricing_info(plan_name: str)`

Get pricing information for subscription plans (basic, pro, enterprise).

### 2. `search_documentation(query: str)`

Search product documentation for specific topics.

## Customization

### Adding New Tools

1. Create a new function in `tools.py`:

```python
from langchain.tools import tool

@tool
def your_new_tool(param: str) -> dict:
    """Tool description for the agent."""
    # Your implementation
    return {"result": "data"}
```

2. Add to `TOOLS` list:

```python
TOOLS = [get_pricing_info, search_documentation, your_new_tool]
```

### Changing the Model

Edit `agent.py`:

```python
agent = OpenAIAgent(
    model="gpt-4o-mini",  # Use a different model
    temperature=0.5        # Adjust creativity
)
```

### Custom System Prompt

Pass it in the request:

```json
{
  "message": "Hello",
  "system_prompt": "You are a specialized pricing assistant..."
}
```

## Deployment

### Docker

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY . .

RUN pip install -e ".[service]"

EXPOSE 5555
CMD ["uvicorn", "examples.example_agent.agent_service:app", "--host", "0.0.0.0", "--port", "5555"]
```

### Environment Variables

Required:
- `OPENAI_API_KEY`: Your OpenAI API key

Optional:
- `PORT`: Server port (default: 5555)
- `HOST`: Server host (default: 0.0.0.0)

## Troubleshooting

### "Module not found" errors

```bash
# Make sure you're running from project root
export PYTHONPATH=.

# Or install in editable mode
pip install -e ".[service]"
```

### "OPENAI_API_KEY not set"

```bash
# Check your .env file
cat .env | grep OPENAI_API_KEY

# Or export directly
export OPENAI_API_KEY=sk-your-key-here
```

### Service won't start

```bash
# Check if port 5555 is already in use
lsof -i :5555

# Use a different port
uvicorn examples.example_agent.agent_service:app --port 5556
```

## Why This Architecture?

### ✅ Advantages

1. **Separation of Concerns**
   - Service handles business logic
   - Adapter handles communication
   - Easy to test each independently

2. **Deployment Flexibility**
   - Service can run anywhere (cloud, on-prem)
   - Multiple adapters can call the same service
   - Service can be scaled independently

3. **Technology Independence**
   - Swap out LangGraph for a different framework
   - Replace OpenAI with another LLM provider
   - Adapter doesn't care about internals

4. **Production Ready**
   - FastAPI provides automatic docs
   - Easy to add auth, rate limiting
   - Can monitor service separately

### ❌ Without This Pattern

The old way mixed everything together:
- OpenAI API calls in the adapter
- Tool execution in the adapter
- Hard to test, hard to deploy
- Can't reuse the agent for other clients

## Learn More

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [LangChain Agents](https://python.langchain.com/docs/modules/agents/)
- [OpenAI API Reference](https://platform.openai.com/docs/api-reference)
