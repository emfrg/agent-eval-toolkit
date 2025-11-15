"""Core types and protocols for agent simulation and evaluation."""

from dataclasses import dataclass, field
from typing import Protocol, List, Dict, Any, Optional, Literal
from deepeval.test_case import Turn

Role = Literal["user", "assistant", "tool"]


@dataclass
class LoggedToolCall:
    """Represents a tool/function call made by the agent.

    Attributes:
        name: Name of the tool/function called
        arguments: Arguments passed to the tool (as dict)
        result: Optional result returned by the tool
    """
    name: str
    arguments: Dict[str, Any]
    result: Optional[Dict[str, Any]] = None


@dataclass
class LoggedTurn:
    """Standardized turn format for logging conversations.

    This is the target schema that all agent outputs should be normalized to.
    Users implement AgentOutputAdapter to convert their API responses to this format.

    Attributes:
        role: Who is speaking (user, assistant, or tool)
        content: The text content of the message
        tool_calls: List of tool calls made in this turn (if any)
        meta: Additional metadata (timestamps, IDs, scores, etc.)
    """
    role: Role
    content: str
    tool_calls: List[LoggedToolCall] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "role": self.role,
            "content": self.content,
            "tool_calls": [
                {
                    "name": tc.name,
                    "arguments": tc.arguments,
                    "result": tc.result
                }
                for tc in self.tool_calls
            ],
            "meta": self.meta
        }


class AgentCallback(Protocol):
    """Protocol for agent callback function.

    Users implement this to connect their agent API to the simulation system.
    This follows DeepEval's ConversationSimulator callback pattern.

    The callback should:
    1. Take the user input and conversation history
    2. Call your agent/chatbot API
    3. Return the agent's response as a Turn object

    Example:
        async def my_agent_callback(
            input: str,
            turns: List[Turn],
            thread_id: str
        ) -> Turn:
            # Convert turns to your agent's format
            history = [{"role": t.role, "content": t.content} for t in turns]

            # Call your agent
            response = await my_agent_api.chat(input, history=history)

            # Return as Turn
            return Turn(role="assistant", content=response)
    """

    async def __call__(
        self,
        input: str,
        turns: List[Turn],
        thread_id: str
    ) -> Turn:
        """Call the agent and return its response.

        Args:
            input: The current user message
            turns: Full conversation history as DeepEval Turn objects
            thread_id: Unique identifier for this conversation thread

        Returns:
            Turn: The agent's response
        """
        ...


class AgentOutputAdapter(Protocol):
    """Protocol for adapting agent-specific output to LoggedTurn format.

    Since every agent API returns different response structures, users implement
    this adapter to normalize their agent's output into the standard LoggedTurn
    schema used for logging and evaluation.

    Example:
        def my_output_adapter(raw_response: Dict[str, Any]) -> LoggedTurn:
            # Extract tool calls if present
            tool_calls = []
            if "function_calls" in raw_response:
                for fc in raw_response["function_calls"]:
                    tool_calls.append(LoggedToolCall(
                        name=fc["name"],
                        arguments=fc["arguments"],
                        result=fc.get("result")
                    ))

            # Create normalized turn
            return LoggedTurn(
                role="assistant",
                content=raw_response["message"],
                tool_calls=tool_calls,
                meta={"model": raw_response.get("model"), "timestamp": ...}
            )
    """

    def __call__(self, raw_agent_response: Any) -> LoggedTurn:
        """Convert raw agent response to normalized LoggedTurn.

        Args:
            raw_agent_response: The raw response from your agent API

        Returns:
            LoggedTurn: Normalized turn in standard format
        """
        ...


@dataclass
class PersonaConfig:
    """Configuration for a simulated user persona.

    Attributes:
        name: Unique identifier for this persona
        scenario: The scenario/task the user is trying to accomplish
        expected_outcome: What success looks like for this persona
        user_description: Description of the user's characteristics
        conversations: Number of conversations to simulate for this persona
        max_turns: Maximum number of turns per conversation (safety limit)
        context: Optional additional context for the simulation
        chatbot_role: Optional role description for the chatbot
    """
    name: str
    scenario: str
    expected_outcome: str
    user_description: str
    conversations: int = 5
    max_turns: int = 15
    context: Optional[str] = None
    chatbot_role: Optional[str] = None
