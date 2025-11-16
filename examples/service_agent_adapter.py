"""Service-based agent adapter example.

This adapter demonstrates the CORRECT architecture - calling a real agent service
via HTTP instead of mixing business logic into the adapter.

Architecture:
    Agent Service (FastAPI) → HTTP → Adapter → DeepEval Turn

The service handles:
- OpenAI API calls
- Tool execution
- Business logic

The adapter handles:
- HTTP communication
- Format conversion (Service Response → DeepEval Turn)
- Error handling

To use this adapter:
1. Start the agent service:
   python examples/services/run.py

2. Run simulations with this adapter:
   llm-evals-starter simulate --agent examples/service_agent_adapter.py
"""

import os
import httpx
from typing import List, Optional
from deepeval.test_case import Turn


class ServiceAgentAdapter:
    """Adapter that calls the OpenAI agent service via HTTP.

    This demonstrates proper separation of concerns:
    - Service: Handles OpenAI API, tools, business logic
    - Adapter: Handles HTTP calls and format conversion
    """

    def __init__(
        self,
        service_url: str = "http://localhost:5555",
        timeout: float = 30.0
    ):
        """Initialize the service agent adapter.

        Args:
            service_url: Base URL of the agent service (default: http://localhost:5555)
            timeout: Request timeout in seconds (default: 30.0)
        """
        self.service_url = service_url.rstrip("/")
        self.timeout = timeout
        self.client = httpx.AsyncClient(timeout=timeout)

    async def __call__(
        self,
        input: str,
        turns: List[Turn],
        thread_id: str
    ) -> Turn:
        """Call the agent service and return the response.

        Args:
            input: Current user message
            turns: Conversation history
            thread_id: Unique conversation ID

        Returns:
            Turn: The agent's response
        """
        try:
            # Convert conversation history to service format
            history = [
                {"role": turn.role, "content": turn.content}
                for turn in turns
            ]

            # Prepare request payload
            payload = {
                "message": input,
                "conversation_id": thread_id,
                "history": history
            }

            # Call the service
            response = await self.client.post(
                f"{self.service_url}/chat",
                json=payload
            )

            # Check for errors
            if response.status_code != 200:
                error_detail = response.json().get("detail", "Unknown error")
                return Turn(
                    role="assistant",
                    content=f"Service error ({response.status_code}): {error_detail}"
                )

            # Parse response
            data = response.json()
            message = data.get("message", "")

            return Turn(role="assistant", content=message)

        except httpx.ConnectError:
            return Turn(
                role="assistant",
                content=(
                    "Cannot connect to agent service. Please make sure it's running:\n"
                    f"python examples/services/run.py"
                )
            )
        except httpx.TimeoutException:
            return Turn(
                role="assistant",
                content=f"Agent service timeout after {self.timeout} seconds"
            )
        except Exception as e:
            return Turn(
                role="assistant",
                content=f"Adapter error: {str(e)}"
            )

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit - cleanup HTTP client."""
        await self.client.aclose()


# Simple callback export for CLI usage
async def agent_callback(input: str, turns: List[Turn], thread_id: str) -> Turn:
    """Agent callback that uses the service adapter.

    This is the function that gets loaded when you use:
        llm-evals-starter simulate --agent examples/service_agent_adapter.py

    The service must be running for this to work:
        python examples/services/run.py
    """
    adapter = ServiceAgentAdapter()
    try:
        return await adapter(input, turns, thread_id)
    finally:
        await adapter.client.aclose()


# Output adapter for structured logging (optional)
def service_output_adapter(raw_response: dict) -> dict:
    """Convert service API response to LoggedTurn format.

    This would be used for structured logging if needed.

    Args:
        raw_response: Raw response from the service API

    Returns:
        Dictionary in LoggedTurn format
    """
    from src.core.types import LoggedTurn, LoggedToolCall

    # Extract tool calls if present
    tool_calls = []
    for tc in raw_response.get("tool_calls", []):
        tool_calls.append(
            LoggedToolCall(
                name=tc.get("name", "unknown"),
                arguments=tc.get("arguments", {}),
                result=tc.get("result")
            )
        )

    return LoggedTurn(
        role="assistant",
        content=raw_response.get("message", ""),
        tool_calls=tool_calls,
        meta=raw_response.get("metadata", {})
    )


if __name__ == "__main__":
    import asyncio

    async def test_service_adapter():
        """Test the service adapter.

        Make sure the service is running first:
            python examples/services/run.py
        """
        print("Testing Service Agent Adapter\n")
        print("Make sure the service is running at http://localhost:5555")
        print("=" * 60)

        adapter = ServiceAgentAdapter()

        # Test conversation
        conversation = [
            "Hello! Can you help me?",
            "What are your pricing plans?",
            "Tell me more about the Pro plan",
            "How do I authenticate with your API?",
        ]

        turns: List[Turn] = []
        thread_id = "test-thread-789"

        try:
            for user_input in conversation:
                print(f"\n👤 User: {user_input}")

                # Get agent response
                agent_turn = await adapter(user_input, turns, thread_id)
                print(f"🤖 Agent: {agent_turn.content}")

                # Add to history
                turns.append(Turn(role="user", content=user_input))
                turns.append(agent_turn)

            print("\n" + "=" * 60)
            print("✅ Test complete!")

        finally:
            await adapter.client.aclose()

    # Run test
    asyncio.run(test_service_adapter())
