"""OpenAI agent adapter example.

This example shows how to connect a real OpenAI-based chatbot to the
simulation system. It demonstrates:
1. Implementing the AgentCallback protocol
2. Converting conversation history to OpenAI format
3. Handling API responses
4. Optional: Implementing tool/function calling
5. Optional: Implementing AgentOutputAdapter for structured logging
"""

import os
from typing import List, Dict, Any, Optional
from deepeval.test_case import Turn
from src.core.types import LoggedTurn, LoggedToolCall

# Note: Install openai package separately
# pip install openai
try:
    from openai import AsyncOpenAI
except ImportError:
    print("OpenAI package not installed. Run: pip install openai")
    raise


class OpenAIAgentAdapter:
    """Adapter for OpenAI-based chatbot."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4o",
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        enable_functions: bool = False,
    ):
        """Initialize OpenAI agent adapter.

        Args:
            api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
            model: Model to use (gpt-4, gpt-3.5-turbo, etc.)
            system_prompt: Optional system prompt for the agent
            temperature: Sampling temperature
            enable_functions: Whether to enable function/tool calling
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OpenAI API key required. Set OPENAI_API_KEY env var or pass api_key parameter."
            )

        self.client = AsyncOpenAI(api_key=self.api_key)
        self.model = model
        self.system_prompt = system_prompt or "You are a helpful AI assistant."
        self.temperature = temperature
        self.enable_functions = enable_functions

        # Example function definitions (if enable_functions=True)
        self.functions = [
            {
                "name": "get_pricing_info",
                "description": "Get pricing information for different plans",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "plan_name": {
                            "type": "string",
                            "description": "The name of the pricing plan (basic, pro, enterprise)",
                        }
                    },
                    "required": ["plan_name"],
                },
            },
            {
                "name": "search_documentation",
                "description": "Search product documentation for specific topics",
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type": "string", "description": "The search query"}},
                    "required": ["query"],
                },
            },
        ]

    async def __call__(self, input: str, turns: List[Turn], thread_id: str) -> Turn:
        """Agent callback that calls OpenAI API.

        Args:
            input: Current user message
            turns: Conversation history
            thread_id: Unique conversation ID

        Returns:
            Turn: Assistant's response
        """
        # Convert conversation history to OpenAI format
        messages = [{"role": "system", "content": self.system_prompt}]

        # Add conversation history
        for turn in turns:
            messages.append({"role": turn.role, "content": turn.content})

        # Add current user input
        messages.append({"role": "user", "content": input})

        # Call OpenAI API
        try:
            if self.enable_functions:
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    functions=self.functions,
                    function_call="auto",
                )
            else:
                response = await self.client.chat.completions.create(
                    model=self.model, messages=messages, temperature=self.temperature
                )

            # Extract response
            message = response.choices[0].message

            # Handle function calls if present
            if hasattr(message, "function_call") and message.function_call:
                # In a real implementation, you would execute the function here
                # For this example, we'll just return a response indicating the function call
                function_name = message.function_call.name
                function_args = message.function_call.arguments

                # Simulate function execution
                function_result = self._execute_function(function_name, function_args)

                # Make another API call with the function result
                messages.append(
                    {
                        "role": "assistant",
                        "content": None,
                        "function_call": {"name": function_name, "arguments": function_args},
                    }
                )
                messages.append(
                    {"role": "function", "name": function_name, "content": str(function_result)}
                )

                # Get final response
                final_response = await self.client.chat.completions.create(
                    model=self.model, messages=messages, temperature=self.temperature
                )

                content = final_response.choices[0].message.content
            else:
                content = message.content

            return Turn(role="assistant", content=content or "")

        except Exception as e:
            # Handle errors gracefully
            error_message = f"I apologize, but I encountered an error: {str(e)}"
            return Turn(role="assistant", content=error_message)

    def _execute_function(self, function_name: str, arguments: str) -> Dict[str, Any]:
        """Simulate function execution.

        In a real implementation, this would call actual backend functions.
        """
        import json

        args = json.loads(arguments)

        if function_name == "get_pricing_info":
            plan = args.get("plan_name", "basic").lower()
            pricing = {
                "basic": {"price": "$10/month", "features": ["Feature A", "Feature B"]},
                "pro": {"price": "$25/month", "features": ["Feature A", "Feature B", "Feature C"]},
                "enterprise": {"price": "Custom", "features": ["All features", "Priority support"]},
            }
            return pricing.get(plan, {"error": "Plan not found"})

        elif function_name == "search_documentation":
            query = args.get("query", "")
            return {
                "results": [
                    f"Documentation result 1 for '{query}'",
                    f"Documentation result 2 for '{query}'",
                ]
            }

        return {"error": "Unknown function"}


# Simple callback export for CLI usage
async def agent_callback(input: str, turns: List[Turn], thread_id: str) -> Turn:
    """Simple agent callback that uses OpenAI adapter.

    This is the function that gets loaded when you use:
        llm-evals-starter simulate --agent examples/openai_agent_adapter.py

    You can customize the agent by modifying the adapter parameters below.
    """
    adapter = OpenAIAgentAdapter(
        model="gpt-4o",
        system_prompt="You are a helpful AI assistant.",
        temperature=0.7,
        enable_functions=False
    )
    return await adapter(input, turns, thread_id)


# Output adapter for structured logging
def openai_output_adapter(raw_response: Dict[str, Any]) -> LoggedTurn:
    """Convert OpenAI API response to LoggedTurn format.

    This would be used if you want structured logging with tool calls.

    Args:
        raw_response: Raw response from OpenAI API

    Returns:
        LoggedTurn: Normalized turn
    """
    # Extract tool calls if present
    tool_calls = []
    if "function_call" in raw_response:
        import json

        tool_calls.append(
            LoggedToolCall(
                name=raw_response["function_call"]["name"],
                arguments=json.loads(raw_response["function_call"]["arguments"]),
                result=None,  # Would be populated after function execution
            )
        )

    return LoggedTurn(
        role="assistant",
        content=raw_response.get("content", ""),
        tool_calls=tool_calls,
        meta={
            "model": raw_response.get("model"),
            "finish_reason": raw_response.get("finish_reason"),
            "usage": raw_response.get("usage", {}),
        },
    )


# Example usage
if __name__ == "__main__":
    import asyncio
    from dotenv import load_dotenv

    load_dotenv()

    async def test_openai_adapter():
        """Test the OpenAI adapter."""
        print("Testing OpenAI Agent Adapter\n")

        # Create adapter
        adapter = OpenAIAgentAdapter(
            model="gpt-3.5-turbo",  # Use cheaper model for testing
            system_prompt="You are a helpful product support assistant.",
            enable_functions=False,  # Set to True to test function calling
        )

        # Simulate a conversation
        conversation = [
            "Hello! Can you help me?",
            "What are your pricing plans?",
            "What's included in the Pro plan?",
            "Thanks!",
        ]

        turns: List[Turn] = []
        thread_id = "test-thread-456"

        for user_input in conversation:
            print(f"User: {user_input}")

            # Get agent response
            agent_turn = await adapter(user_input, turns, thread_id)
            print(f"Agent: {agent_turn.content}\n")

            # Add to history
            turns.append(Turn(role="user", content=user_input))
            turns.append(agent_turn)

        print("Test complete!")

    # Run test
    # Uncomment to test:
    # asyncio.run(test_openai_adapter())
    print(
        """
OpenAI Agent Adapter Example

To test this adapter:
1. Set OPENAI_API_KEY in your .env file
2. Install openai: pip install openai
3. Uncomment the last line in this file
4. Run: python examples/openai_agent_adapter.py

To use in simulations:
1. Import: from examples.openai_agent_adapter import OpenAIAgentAdapter
2. Create: adapter = OpenAIAgentAdapter()
3. Pass to SimulationRunner: runner = SimulationRunner(agent_callback=adapter)
    """
    )
