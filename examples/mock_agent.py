"""Standalone mock agent for quickstart demo.

This agent runs directly without requiring a separate service.
It provides a simple conversational AI using OpenAI's API directly.

Usage:
    This file is used by the `llm-evals-starter quickstart` command.
    It can also be used with:
    llm-evals-starter simulate --agent examples/example_agent.py
"""

import os
from typing import List
from openai import AsyncOpenAI
from deepeval.test_case import Turn

# Initialize OpenAI client
client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Simple in-memory conversation storage
conversations = {}


async def agent_callback(input: str, turns: List[Turn], thread_id: str) -> Turn:
    """Simple conversational agent using OpenAI directly.

    This is a standalone agent that doesn't require a separate service.
    It uses OpenAI's API to generate responses based on conversation history.

    Args:
        input: Current user message
        turns: Conversation history from the eval framework
        thread_id: Unique conversation ID

    Returns:
        Turn: The agent's response in eval framework format
    """
    # Build conversation history for OpenAI
    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful customer support assistant for a SaaS product. "
                "You help users understand pricing, features, and technical details. "
                "Be concise, friendly, and informative. "
                "When asked about pricing, mention three tiers: "
                "Basic ($10/month), Pro ($25/month), and Enterprise (custom pricing). "
                "When asked about technical details, provide clear explanations about "
                "APIs, integrations, and documentation."
            ),
        }
    ]

    # Add conversation history
    for turn in turns:
        role = "user" if turn.role == "user" else "assistant"
        messages.append({"role": role, "content": turn.content})

    # Add current user input
    messages.append({"role": "user", "content": input})

    try:
        # Get response from OpenAI
        response = await client.chat.completions.create(
            model="gpt-4o-mini",  # Using mini for faster/cheaper quickstart
            messages=messages,
            temperature=0.7,
            max_tokens=500,
        )

        # Extract the response
        agent_response = response.choices[0].message.content

        # Return as Turn object
        return Turn(role="assistant", content=agent_response)

    except Exception as e:
        # Handle errors gracefully
        return Turn(
            role="assistant",
            content=f"I apologize, but I encountered an error: {str(e)}. Please try again.",
        )


# Optional: Test the agent directly
if __name__ == "__main__":
    import asyncio

    async def test():
        """Test the agent with a simple conversation."""
        print("Testing Example Agent\n" + "=" * 50)

        # Test conversation
        turns = []
        test_inputs = [
            "Hello! Can you help me?",
            "What are your pricing plans?",
            "Tell me more about the Pro plan",
        ]

        for user_input in test_inputs:
            print(f"\n👤 User: {user_input}")

            # Get agent response
            response = await agent_callback(user_input, turns, "test-thread-123")
            print(f"🤖 Agent: {response.content}")

            # Add to history
            turns.append(Turn(role="user", content=user_input))
            turns.append(response)

        print("\n" + "=" * 50)
        print("✅ Test complete!")

    # Run test
    asyncio.run(test())
