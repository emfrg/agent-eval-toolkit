"""LangChain agent implementation using OpenAI models.

This module creates and configures a LangChain agent that can use tools
to answer user queries.
"""

import os
from typing import List, Dict, Any, Optional
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from .tools import TOOLS
from .models import Message, ToolCall


class OpenAIAgent:
    """Wrapper for LangChain OpenAI agent with tools."""

    def __init__(
        self,
        model: str = "gpt-4o",
        temperature: float = 0.7,
        api_key: Optional[str] = None,
        system_prompt: Optional[str] = None
    ):
        """Initialize the OpenAI agent.

        Args:
            model: OpenAI model to use (default: gpt-4o)
            temperature: Sampling temperature (default: 0.7)
            api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
            system_prompt: Custom system prompt for the agent
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OpenAI API key required. Set OPENAI_API_KEY environment variable "
                "or pass api_key parameter."
            )

        # Initialize the ChatOpenAI model
        self.llm = ChatOpenAI(
            model=model,
            temperature=temperature,
            api_key=self.api_key
        )

        # Default system prompt
        self.default_system_prompt = (
            "You are a helpful AI assistant with access to tools that can help you "
            "answer questions about pricing and documentation. When users ask about "
            "pricing or features, use the get_pricing_info tool. When users ask about "
            "how to use the platform or need documentation, use the search_documentation tool. "
            "Always be concise and helpful in your responses."
        )

        # Create the agent with tools
        self.agent = create_agent(
            self.llm,
            TOOLS,
            system_prompt=system_prompt or self.default_system_prompt
        )

    async def chat(
        self,
        message: str,
        history: List[Message] = None,
        conversation_id: str = "default"
    ) -> Dict[str, Any]:
        """Send a message to the agent and get a response.

        Args:
            message: The user's message
            history: Previous conversation history
            conversation_id: Unique identifier for this conversation

        Returns:
            Dictionary containing the response message, tool calls, and metadata
        """
        # Convert history to LangChain message format
        messages = []

        if history:
            for msg in history:
                messages.append({
                    "role": msg.role,
                    "content": msg.content
                })

        # Add current user message
        messages.append({
            "role": "user",
            "content": message
        })

        # Invoke the agent
        try:
            # Stream the agent's response
            tool_calls_made = []
            final_message = ""

            # Run the agent
            result = await self.agent.ainvoke({
                "messages": messages
            })

            # Extract the response
            if "messages" in result and len(result["messages"]) > 0:
                last_message = result["messages"][-1]

                # Get the final response content
                if hasattr(last_message, "content"):
                    final_message = last_message.content
                elif isinstance(last_message, dict):
                    final_message = last_message.get("content", "")

                # Extract tool calls if any
                if hasattr(last_message, "tool_calls"):
                    for tool_call in last_message.tool_calls:
                        tool_calls_made.append(ToolCall(
                            name=tool_call.get("name", "unknown"),
                            arguments=tool_call.get("args", {}),
                            result=None  # Result is embedded in response
                        ))

            return {
                "message": final_message,
                "conversation_id": conversation_id,
                "tool_calls": [tc.model_dump() for tc in tool_calls_made],
                "metadata": {
                    "model": self.llm.model_name,
                    "temperature": self.llm.temperature,
                }
            }

        except Exception as e:
            # Handle errors gracefully
            return {
                "message": f"I apologize, but I encountered an error: {str(e)}",
                "conversation_id": conversation_id,
                "tool_calls": [],
                "metadata": {
                    "error": str(e),
                    "model": self.llm.model_name
                }
            }


# Singleton instance of the agent
_agent_instance: Optional[OpenAIAgent] = None


def get_agent() -> OpenAIAgent:
    """Get or create the global agent instance.

    Returns:
        The global OpenAIAgent instance
    """
    global _agent_instance
    if _agent_instance is None:
        _agent_instance = OpenAIAgent()
    return _agent_instance


def reset_agent():
    """Reset the global agent instance.

    Useful for testing or when configuration changes.
    """
    global _agent_instance
    _agent_instance = None
