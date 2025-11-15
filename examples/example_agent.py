"""Example agent adapter for testing the simulation system.

This example shows how to implement the AgentCallback protocol without
connecting to a real agent API. Perfect for:
- Testing the simulation system end-to-end
- Understanding the callback interface
- Quick demos and prototyping
"""

from typing import List
from deepeval.test_case import Turn
import random


async def example_agent_callback(
    input: str,
    turns: List[Turn],
    thread_id: str
) -> Turn:
    """Simple example agent that responds with canned responses.

    This demonstrates the minimal implementation of AgentCallback.

    Args:
        input: Current user message
        turns: Full conversation history
        thread_id: Unique conversation identifier

    Returns:
        Turn: The agent's response
    """
    # Example 1: Simple echo response
    if "hello" in input.lower() or "hi" in input.lower():
        return Turn(
            role="assistant",
            content="Hello! How can I help you today?"
        )

    # Example 2: Context-aware response using conversation history
    turn_count = len(turns)
    if turn_count > 8:
        return Turn(
            role="assistant",
            content="We've been chatting for a while. Is there anything else I can help with?"
        )

    # Example 3: Keyword-based responses
    keywords_responses = {
        "help": "I'm here to help! What do you need assistance with?",
        "thanks": "You're welcome! Happy to help.",
        "bye": "Goodbye! Have a great day!",
        "price": "Our pricing varies by plan. Basic starts at $10/month.",
        "feature": "We offer many features including X, Y, and Z.",
        "support": "Our support team is available 24/7 via email and chat.",
    }

    for keyword, response in keywords_responses.items():
        if keyword in input.lower():
            return Turn(role="assistant", content=response)

    # Example 4: Default responses with variety
    default_responses = [
        f"I understand you mentioned: '{input}'. Let me help with that.",
        "That's interesting. Can you tell me more about what you need?",
        "I see. Let me provide some information about that.",
        "Great question! Here's what I can tell you...",
        "I'd be happy to help with that. Here's what you should know..."
    ]

    return Turn(
        role="assistant",
        content=random.choice(default_responses)
    )


# Alias for CLI usage
agent_callback = example_agent_callback


async def stateful_example_agent_callback(
    input: str,
    turns: List[Turn],
    thread_id: str
) -> Turn:
    """More sophisticated example agent that maintains conversation context.

    This example shows how to use the turns history to build context-aware
    responses, similar to a real agent with memory.

    Args:
        input: Current user message
        turns: Full conversation history
        thread_id: Unique conversation identifier

    Returns:
        Turn: The agent's response
    """
    # Track what user has asked about
    topics_discussed = set()
    for turn in turns:
        if turn.role == "user":
            content_lower = turn.content.lower()
            if "price" in content_lower or "cost" in content_lower:
                topics_discussed.add("pricing")
            if "feature" in content_lower:
                topics_discussed.add("features")
            if "support" in content_lower:
                topics_discussed.add("support")

    # Provide contextual responses based on history
    current_lower = input.lower()

    if "price" in current_lower or "cost" in current_lower:
        if "pricing" in topics_discussed:
            return Turn(
                role="assistant",
                content="As I mentioned earlier, our Basic plan is $10/month. "
                        "Would you like to know about our other tiers?"
            )
        else:
            return Turn(
                role="assistant",
                content="Our pricing has three tiers: Basic ($10/mo), Pro ($25/mo), "
                        "and Enterprise (custom). Which interests you?"
            )

    if "feature" in current_lower:
        features = [
            "Real-time collaboration",
            "Advanced analytics",
            "API access",
            "Custom integrations",
            "Priority support"
        ]
        selected_features = random.sample(features, 3)
        return Turn(
            role="assistant",
            content=f"Key features include: {', '.join(selected_features)}. "
                    "Would you like details on any of these?"
        )

    # Check if user is repeating themselves
    user_turns = [t.content for t in turns if t.role == "user"]
    if input in user_turns:
        return Turn(
            role="assistant",
            content="I notice you've asked about this before. Can I clarify anything "
                    "from my previous response?"
        )

    # Default response
    return Turn(
        role="assistant",
        content=f"I understand you're asking about: {input}. "
                "Let me provide you with relevant information. "
                f"(Conversation turn #{len(turns) + 1})"
    )


# Example of how to use these callbacks with the simulation system
if __name__ == "__main__":
    import asyncio
    from deepeval.test_case import Turn

    async def test_example_agent():
        """Test the example agent callbacks."""
        print("Testing example_agent_callback:\n")

        # Simulate a conversation
        conversation = [
            "Hello!",
            "What are your features?",
            "How much does it cost?",
            "Thanks for the help!",
            "Bye!"
        ]

        turns: List[Turn] = []
        thread_id = "test-thread-123"

        for user_input in conversation:
            print(f"User: {user_input}")

            # Get agent response
            agent_turn = await example_agent_callback(user_input, turns, thread_id)
            print(f"Agent: {agent_turn.content}\n")

            # Add to history
            turns.append(Turn(role="user", content=user_input))
            turns.append(agent_turn)

        print("\n" + "="*50 + "\n")
        print("Testing stateful_example_agent_callback:\n")

        # Test stateful agent
        turns = []
        for user_input in conversation:
            print(f"User: {user_input}")

            agent_turn = await stateful_example_agent_callback(user_input, turns, thread_id)
            print(f"Agent: {agent_turn.content}\n")

            turns.append(Turn(role="user", content=user_input))
            turns.append(agent_turn)

    # Run the test
    asyncio.run(test_example_agent())
