"""Standalone mock agent for quickstart demo.

This agent runs directly without requiring a separate service.
It provides mock responses without any external API calls.

Usage:
    This file is used by the `agent-eval-toolkit quickstart` command.
    It can also be used with:
    agent-eval-toolkit simulate --agent examples/mock_agent.py
"""

import random
from typing import List
from deepeval.test_case import Turn

# Simple in-memory conversation storage
conversations = {}


async def agent_callback(input: str, turns: List[Turn], thread_id: str) -> Turn:
    """Mock conversational agent with predefined responses.

    This is a standalone agent that doesn't require any external API calls.
    It uses keyword matching and predefined responses to simulate conversation.

    Args:
        input: Current user message
        turns: Conversation history from the eval framework
        thread_id: Unique conversation ID

    Returns:
        Turn: The agent's response in eval framework format
    """

    # Store conversation for context (not currently used, but available)
    if thread_id not in conversations:
        conversations[thread_id] = []
    conversations[thread_id].append(input)

    # Convert input to lowercase for matching
    input_lower = input.lower()

    # Count conversation turns for variety
    turn_count = len(turns)

    # Greeting responses
    if any(word in input_lower for word in ['hello', 'hi', 'hey', 'greetings', 'good morning', 'good afternoon']):
        responses = [
            "Hello! Welcome to our support. How can I assist you today?",
            "Hi there! I'm here to help you with any questions about our product.",
            "Greetings! What can I help you with today?",
            "Hello! I'd be happy to help you with our services.",
        ]
        response = random.choice(responses)

    # Pricing-related queries
    elif any(word in input_lower for word in ['price', 'pricing', 'cost', 'expensive', 'cheap', 'plan', 'tier', 'subscription']):
        # Check if asking about specific plans
        if 'pro' in input_lower or 'professional' in input_lower:
            response = (
                "Our Pro plan is $25/month and includes advanced analytics, "
                "priority support, up to 10 team members, and API access with "
                "higher rate limits. It's perfect for growing teams."
            )
        elif 'enterprise' in input_lower:
            response = (
                "Our Enterprise plan offers custom pricing based on your needs. "
                "It includes unlimited team members, dedicated support, custom "
                "integrations, SLA guarantees, and advanced security features. "
                "Please contact our sales team for a personalized quote."
            )
        elif 'basic' in input_lower or 'starter' in input_lower:
            response = (
                "Our Basic plan is $10/month and includes core features, "
                "email support, up to 3 team members, and standard integrations. "
                "It's ideal for individuals and small teams just getting started."
            )
        else:
            response = (
                "We offer three pricing tiers to meet different needs:\n"
                "• Basic ($10/month) - Perfect for individuals\n"
                "• Pro ($25/month) - Great for growing teams\n"
                "• Enterprise (custom) - For large organizations\n"
                "Each plan includes a 14-day free trial. Which one interests you?"
            )

    # Feature-related queries
    elif any(word in input_lower for word in ['feature', 'capability', 'function', 'what can', 'do you have', 'support']):
        if 'integration' in input_lower or 'api' in input_lower:
            response = (
                "We support integrations with popular tools like Slack, GitHub, "
                "Jira, and Zapier. Our RESTful API allows you to build custom "
                "integrations. Pro and Enterprise plans include webhook support "
                "and higher API rate limits."
            )
        elif 'security' in input_lower:
            response = (
                "Security is our top priority. We offer end-to-end encryption, "
                "SOC 2 compliance, regular security audits, and SSO/SAML for "
                "Enterprise customers. All data is encrypted at rest and in transit."
            )
        else:
            response = (
                "Our platform includes real-time analytics, team collaboration tools, "
                "customizable dashboards, automated reporting, and extensive API access. "
                "We also offer mobile apps for iOS and Android. What specific feature "
                "are you most interested in?"
            )

    # Documentation/help queries
    elif any(word in input_lower for word in ['documentation', 'docs', 'guide', 'tutorial', 'how to', 'help']):
        response = (
            "You can find comprehensive documentation at docs.ourproduct.com. "
            "We have getting started guides, API references, video tutorials, "
            "and a community forum. For immediate assistance, I can help answer "
            "specific questions you might have."
        )

    # Support/problem queries
    elif any(word in input_lower for word in ['problem', 'issue', 'error', 'broken', 'not working', 'bug']):
        response = (
            "I'm sorry to hear you're experiencing issues. Could you please describe "
            "the problem in more detail? Include any error messages you're seeing, "
            "and I'll help troubleshoot. For urgent issues, Pro and Enterprise "
            "customers have access to priority support."
        )

    # Trial/demo queries
    elif any(word in input_lower for word in ['trial', 'demo', 'test', 'try', 'free']):
        response = (
            "Yes! We offer a 14-day free trial for all plans with no credit card "
            "required. You can also schedule a personalized demo with our team. "
            "The trial includes full access to all features of your chosen plan."
        )

    # Contact/sales queries
    elif any(word in input_lower for word in ['contact', 'sales', 'speak', 'call', 'email', 'reach']):
        response = (
            "You can reach our team at support@ourproduct.com or call 1-800-EXAMPLE. "
            "For sales inquiries, email sales@ourproduct.com. Our support hours are "
            "Monday-Friday 9 AM to 6 PM EST, with 24/7 support for Enterprise customers."
        )

    # Thank you / closing
    elif any(word in input_lower for word in ['thank', 'thanks', 'appreciate', 'helpful']):
        responses = [
            "You're very welcome! Is there anything else I can help you with?",
            "Happy to help! Feel free to ask if you have any other questions.",
            "Glad I could assist! Don't hesitate to reach out if you need more help.",
            "My pleasure! Let me know if there's anything else you'd like to know.",
        ]
        response = random.choice(responses)

    # Goodbye
    elif any(word in input_lower for word in ['bye', 'goodbye', 'see you', 'farewell', 'exit']):
        responses = [
            "Goodbye! Have a great day and feel free to come back anytime.",
            "Thank you for chatting with us. Have a wonderful day!",
            "Bye! Don't hesitate to reach out if you need help in the future.",
            "Take care! We're always here if you need assistance.",
        ]
        response = random.choice(responses)

    # Questions about specific use cases
    elif any(word in input_lower for word in ['use case', 'example', 'scenario', 'how do i']):
        response = (
            "Our product is versatile and used across many industries. Common use cases "
            "include project management, team collaboration, data analytics, and workflow "
            "automation. I can provide specific examples for your industry if you tell me "
            "more about your needs."
        )

    # Performance/speed queries
    elif any(word in input_lower for word in ['fast', 'slow', 'performance', 'speed', 'latency']):
        response = (
            "Our platform is built for speed with average response times under 200ms. "
            "We use global CDN for fast content delivery and have 99.9% uptime SLA "
            "for Enterprise customers. Real-time features update within milliseconds."
        )

    # Comparison queries
    elif any(word in input_lower for word in ['compare', 'versus', 'vs', 'better', 'difference', 'competitor']):
        response = (
            "While I respect all solutions in the market, our key differentiators include "
            "superior ease of use, competitive pricing, extensive integrations, and "
            "industry-leading customer support. We'd be happy to provide a detailed "
            "comparison based on your specific requirements."
        )

    # Default contextual responses based on conversation length
    else:
        if turn_count == 0:
            # First turn, be welcoming
            response = (
                "I'd be happy to help you learn more about our product. We offer "
                "comprehensive solutions for teams of all sizes. What would you like "
                "to know about our features, pricing, or capabilities?"
            )
        elif turn_count < 3:
            # Early in conversation, be informative
            responses = [
                "That's an interesting question. Let me provide you with more details about our solution.",
                "I can definitely help with that. Our platform offers various capabilities to address different needs.",
                "Great question! Our product is designed to be flexible and scalable for various use cases.",
            ]
            response = random.choice(responses)
        else:
            # Later in conversation, vary responses
            responses = [
                "I understand your interest. Could you tell me more about your specific requirements?",
                "That's a good point. Based on what you've mentioned, I think our Pro plan might be suitable for you.",
                "I can see how that would be important for your use case. Let me explain how we handle that.",
                "Certainly! Our solution addresses that through several key features I can describe.",
            ]
            response = random.choice(responses)

    # Return as Turn object
    return Turn(role="assistant", content=response)


# Optional: Test the agent directly
if __name__ == "__main__":
    import asyncio

    async def test():
        """Test the agent with a simple conversation."""
        print("Testing Mock Agent (No API Calls)\n" + "=" * 50)

        # Test conversation
        turns = []
        test_inputs = [
            "Hello! Can you help me?",
            "What are your pricing plans?",
            "Tell me more about the Pro plan",
            "Do you have API access?",
            "Thank you for the information!",
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
        print("✅ Test complete! (No API calls were made)")

    # Run test
    asyncio.run(test())