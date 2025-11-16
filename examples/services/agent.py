"""LangGraph agent implementation using OpenAI models.

This module creates and configures a LangGraph agent that can use tools
to answer user queries.
"""

import os
from typing import List, Dict, Any, Optional, Literal
from typing_extensions import TypedDict, Annotated
import operator

from langchain_openai import ChatOpenAI
from langchain_core.messages import AnyMessage, SystemMessage, ToolMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END

from .tools import TOOLS
from .models import Message, ToolCall


# Define the state for the agent
class MessagesState(TypedDict):
    """State for the agent graph."""
    messages: Annotated[List[AnyMessage], operator.add]


class OpenAIAgent:
    """Wrapper for LangGraph OpenAI agent with tools."""

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

        self.system_prompt = system_prompt or self.default_system_prompt

        # Build tools lookup
        self.tools = TOOLS
        self.tools_by_name = {tool.name: tool for tool in TOOLS}

        # Bind tools to the model
        self.model_with_tools = self.llm.bind_tools(TOOLS)

        # Build the agent graph
        self.agent = self._build_graph()

    def _build_graph(self) -> StateGraph:
        """Build the LangGraph agent graph."""

        # Define the LLM call node
        def llm_call(state: MessagesState) -> Dict[str, Any]:
            """Call the LLM with tools."""
            messages = [SystemMessage(content=self.system_prompt)] + state["messages"]
            response = self.model_with_tools.invoke(messages)
            return {"messages": [response]}

        # Define the tool execution node
        def tool_node(state: MessagesState) -> Dict[str, Any]:
            """Execute tools based on the last message's tool calls."""
            last_message = state["messages"][-1]
            tool_messages = []

            for tool_call in last_message.tool_calls:
                tool = self.tools_by_name[tool_call["name"]]
                observation = tool.invoke(tool_call["args"])
                tool_messages.append(
                    ToolMessage(
                        content=str(observation),
                        tool_call_id=tool_call["id"]
                    )
                )

            return {"messages": tool_messages}

        # Define routing logic
        def should_continue(state: MessagesState) -> Literal["tool_node", END]:
            """Determine whether to continue to tools or end."""
            last_message = state["messages"][-1]
            if hasattr(last_message, "tool_calls") and last_message.tool_calls:
                return "tool_node"
            return END

        # Build the graph
        graph_builder = StateGraph(MessagesState)
        graph_builder.add_node("llm_call", llm_call)
        graph_builder.add_node("tool_node", tool_node)
        graph_builder.add_edge(START, "llm_call")
        graph_builder.add_conditional_edges("llm_call", should_continue, ["tool_node", END])
        graph_builder.add_edge("tool_node", "llm_call")

        return graph_builder.compile()

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
        try:
            # Convert history to LangChain message format
            messages = []

            if history:
                for msg in history:
                    if msg.role == "user":
                        messages.append(HumanMessage(content=msg.content))
                    elif msg.role == "assistant":
                        messages.append(AIMessage(content=msg.content))

            # Add current user message
            messages.append(HumanMessage(content=message))

            # Invoke the agent graph
            result = await self.agent.ainvoke({"messages": messages})

            # Extract the final response and all tool calls with results
            final_message = ""
            tool_calls_made = []

            # First pass: Build map of tool_call_id -> result from ToolMessages
            tool_results = {}
            for msg in result["messages"]:
                if isinstance(msg, ToolMessage):
                    tool_results[msg.tool_call_id] = msg.content

            # Second pass: Extract tool calls from AIMessages and match with results
            for msg in result["messages"]:
                if isinstance(msg, AIMessage):
                    # Always update final_message (last one wins)
                    final_message = msg.content

                    # Collect tool calls with their results
                    if hasattr(msg, "tool_calls") and msg.tool_calls:
                        for tool_call in msg.tool_calls:
                            tool_calls_made.append(ToolCall(
                                name=tool_call.get("name", "unknown"),
                                arguments=tool_call.get("args", {}),
                                result=tool_results.get(tool_call["id"])
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
