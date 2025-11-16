"""Pydantic models for the OpenAI agent service API.

These models define the request and response schemas for the FastAPI endpoints.
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class Message(BaseModel):
    """A single message in a conversation."""

    role: str = Field(..., description="Role of the message sender (user, assistant, tool)")
    content: str = Field(..., description="Content of the message")


class ToolCall(BaseModel):
    """Information about a tool/function call made by the agent."""

    name: str = Field(..., description="Name of the tool that was called")
    arguments: Dict[str, Any] = Field(default_factory=dict, description="Arguments passed to the tool")
    result: Optional[Any] = Field(None, description="Result returned by the tool")


class ChatRequest(BaseModel):
    """Request model for the chat endpoint."""

    message: str = Field(..., description="The user's message", min_length=1)
    conversation_id: str = Field(default="default", description="Unique identifier for the conversation thread")
    history: List[Message] = Field(
        default_factory=list,
        description="Previous messages in the conversation for context"
    )
    system_prompt: Optional[str] = Field(
        None,
        description="Optional custom system prompt for this conversation"
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "message": "What are your pricing plans?",
                    "conversation_id": "user-123-session-456",
                    "history": [
                        {"role": "user", "content": "Hello!"},
                        {"role": "assistant", "content": "Hi! How can I help you today?"}
                    ]
                }
            ]
        }
    }


class ChatResponse(BaseModel):
    """Response model for the chat endpoint."""

    message: str = Field(..., description="The agent's response message")
    conversation_id: str = Field(..., description="The conversation ID this response belongs to")
    tool_calls: List[ToolCall] = Field(
        default_factory=list,
        description="List of tools called during response generation"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata about the response (model, tokens, etc.)"
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "message": "We offer three pricing plans: Basic ($10/month), Pro ($25/month), and Enterprise (custom pricing).",
                    "conversation_id": "user-123-session-456",
                    "tool_calls": [
                        {
                            "name": "get_pricing_info",
                            "arguments": {"plan_name": "basic"},
                            "result": {"price": "$10/month", "features": ["Feature A", "Feature B"]}
                        }
                    ],
                    "metadata": {
                        "model": "gpt-4o",
                        "tokens_used": 245
                    }
                }
            ]
        }
    }


class HealthResponse(BaseModel):
    """Response model for the health check endpoint."""

    status: str = Field(..., description="Service status")
    version: str = Field(..., description="Service version")
    agent_configured: bool = Field(..., description="Whether the agent is properly configured")
