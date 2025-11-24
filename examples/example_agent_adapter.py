"""Example agent adapter for the evaluation framework.

This adapter demonstrates how to integrate your agent service with the
evaluation framework. It calls the agent service via HTTP and converts
the response to Turn objects for the eval pipeline.

Architecture:
    Your Agent Service (HTTP API) → Adapter → DeepEval Turn Objects

To use this adapter:
1. Start your agent service:
   python examples/example_agent/run.py

2. Run simulations with this adapter:
   llm-evals-starter simulate --agent examples/example_agent_adapter.py
"""

import httpx
from typing import List
from deepeval.test_case import Turn


async def agent_callback(input: str, turns: List[Turn], thread_id: str) -> Turn:
    """Agent callback that calls the service and returns a Turn.

    This is the main function that the evaluation framework calls.
    It demonstrates the minimal adapter pattern:
    1. Convert eval framework data (turns) to service format
    2. Call your agent service via HTTP
    3. Convert service response back to eval format (Turn)

    Args:
        input: Current user message
        turns: Conversation history from the eval framework
        thread_id: Unique conversation ID

    Returns:
        Turn: The agent's response in eval framework format
    """
    # Convert conversation history to service format
    history = [{"role": turn.role, "content": turn.content} for turn in turns]

    # Prepare request for the agent service
    payload = {"message": input, "conversation_id": thread_id, "history": history}

    # Call the agent service
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            response = await client.post("http://localhost:5555/chat", json=payload)

            if response.status_code != 200:
                error_detail = response.json().get("detail", "Unknown error")
                return Turn(
                    role="assistant",
                    content=f"Service error ({response.status_code}): {error_detail}",
                )

            # Parse response
            data = response.json()
            message = data.get("message", "")

            # Convert service response to Turn
            return Turn(role="assistant", content=message)

        except httpx.ConnectError:
            return Turn(
                role="assistant",
                content=(
                    "Cannot connect to agent service. Please make sure it's running:\n"
                    "python examples/example_agent/run.py"
                ),
            )
        except httpx.TimeoutException:
            return Turn(role="assistant", content="Agent service timeout after 30 seconds")
        except Exception as e:
            return Turn(role="assistant", content=f"Adapter error: {str(e)}")


# Optional: Test the adapter directly
if __name__ == "__main__":
    import asyncio

    async def test_adapter():
        """Test the adapter with a simple conversation."""
        print("Testing Example Agent Adapter\n")
        print("Make sure the service is running at http://localhost:5555")
        print("=" * 60)

        # Test conversation
        conversation = [
            "Hello! Can you help me?",
            "What are your pricing plans?",
            "Tell me more about the Pro plan",
        ]

        turns: List[Turn] = []
        thread_id = "test-thread-123"

        for user_input in conversation:
            print(f"\n👤 User: {user_input}")

            # Get agent response
            agent_turn = await agent_callback(user_input, turns, thread_id)
            print(f"🤖 Agent: {agent_turn.content[:200]}...")

            # Add to history
            turns.append(Turn(role="user", content=user_input))
            turns.append(agent_turn)

        print("\n" + "=" * 60)
        print("✅ Test complete!")

    # Run test
    asyncio.run(test_adapter())
