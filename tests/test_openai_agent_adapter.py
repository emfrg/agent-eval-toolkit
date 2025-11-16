"""Tests for OpenAI agent adapter.

This test suite validates the OpenAI agent adapter implementation including:
- Basic conversation flow
- Tool/function calling
- Parallel tool calling
- Error handling
- Response formatting

Run with: pytest tests/test_openai_agent_adapter.py -v
"""

import sys
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch
import pytest
import json

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from deepeval.test_case import Turn
from examples.openai_agent_adapter import OpenAIAgentAdapter, agent_callback


class TestOpenAIAgentAdapter:
    """Test suite for OpenAIAgentAdapter class."""

    @pytest.fixture
    def mock_openai_response(self):
        """Create a mock OpenAI API response."""
        mock_response = Mock()
        mock_message = Mock()
        mock_message.content = "Hello! How can I help you today?"
        mock_message.tool_calls = None
        mock_response.choices = [Mock(message=mock_message)]
        return mock_response

    @pytest.fixture
    def mock_tool_call_response(self):
        """Create a mock OpenAI API response with tool calls."""
        # First response - tool call
        tool_call = Mock()
        tool_call.id = "call_123"
        tool_call.function.name = "get_pricing_info"
        tool_call.function.arguments = '{"plan_name": "pro"}'

        mock_message = Mock()
        mock_message.content = None
        mock_message.tool_calls = [tool_call]

        mock_response = Mock()
        mock_response.choices = [Mock(message=mock_message)]
        return mock_response

    @pytest.fixture
    def mock_final_response(self):
        """Create a mock final response after tool execution."""
        mock_message = Mock()
        mock_message.content = "The Pro plan costs $25/month and includes Feature A, Feature B, and Feature C."
        mock_message.tool_calls = None

        mock_response = Mock()
        mock_response.choices = [Mock(message=mock_message)]
        return mock_response

    @pytest.mark.asyncio
    async def test_basic_conversation(self, mock_openai_response):
        """Test basic conversation without tool calling."""
        with patch("examples.openai_agent_adapter.AsyncOpenAI") as mock_client_class:
            # Setup mock
            mock_client = AsyncMock()
            mock_client.chat.completions.create = AsyncMock(return_value=mock_openai_response)
            mock_client_class.return_value = mock_client

            # Create adapter
            adapter = OpenAIAgentAdapter(
                api_key="test-key",
                model="gpt-4o",
                enable_functions=False
            )

            # Test conversation
            turns = []
            response = await adapter(
                input="Hello!",
                turns=turns,
                thread_id="test-123"
            )

            # Assertions
            assert response.role == "assistant"
            assert response.content == "Hello! How can I help you today?"
            assert isinstance(response, Turn)

            # Verify API was called correctly
            mock_client.chat.completions.create.assert_called_once()
            call_args = mock_client.chat.completions.create.call_args
            assert call_args.kwargs["model"] == "gpt-4o"
            assert "tools" not in call_args.kwargs  # No tools when enable_functions=False

    @pytest.mark.asyncio
    async def test_conversation_with_history(self, mock_openai_response):
        """Test conversation with existing history."""
        with patch("examples.openai_agent_adapter.AsyncOpenAI") as mock_client_class:
            # Setup mock
            mock_client = AsyncMock()
            mock_client.chat.completions.create = AsyncMock(return_value=mock_openai_response)
            mock_client_class.return_value = mock_client

            # Create adapter
            adapter = OpenAIAgentAdapter(api_key="test-key")

            # Create conversation history
            turns = [
                Turn(role="user", content="What's your name?"),
                Turn(role="assistant", content="I'm a helpful assistant.")
            ]

            # Test with history
            response = await adapter(
                input="What can you do?",
                turns=turns,
                thread_id="test-456"
            )

            # Verify history was included in API call
            call_args = mock_client.chat.completions.create.call_args
            messages = call_args.kwargs["messages"]

            # Should have: system + 2 history turns + current input = 4 messages
            assert len(messages) == 4
            assert messages[0]["role"] == "system"
            assert messages[1]["role"] == "user"
            assert messages[1]["content"] == "What's your name?"
            assert messages[2]["role"] == "assistant"
            assert messages[2]["content"] == "I'm a helpful assistant."
            assert messages[3]["role"] == "user"
            assert messages[3]["content"] == "What can you do?"

    @pytest.mark.asyncio
    async def test_tool_calling_single(self, mock_tool_call_response, mock_final_response):
        """Test single tool calling flow."""
        with patch("examples.openai_agent_adapter.AsyncOpenAI") as mock_client_class:
            # Setup mock to return tool call first, then final response
            mock_client = AsyncMock()
            mock_client.chat.completions.create = AsyncMock(
                side_effect=[mock_tool_call_response, mock_final_response]
            )
            mock_client_class.return_value = mock_client

            # Create adapter with tools enabled
            adapter = OpenAIAgentAdapter(
                api_key="test-key",
                enable_functions=True
            )

            # Test tool calling
            response = await adapter(
                input="What's the pricing for Pro plan?",
                turns=[],
                thread_id="test-789"
            )

            # Assertions
            assert response.role == "assistant"
            assert "Pro plan costs $25/month" in response.content

            # Verify two API calls were made (initial + after tool execution)
            assert mock_client.chat.completions.create.call_count == 2

            # Verify tool result was sent correctly
            second_call_args = mock_client.chat.completions.create.call_args_list[1]
            messages = second_call_args.kwargs["messages"]

            # Should include the tool result message
            tool_result_msg = [m for m in messages if m.get("role") == "tool"][0]
            assert tool_result_msg["tool_call_id"] == "call_123"
            assert tool_result_msg["name"] == "get_pricing_info"
            assert "25/month" in tool_result_msg["content"]

    @pytest.mark.asyncio
    async def test_parallel_tool_calling(self):
        """Test parallel tool calling (new API feature)."""
        with patch("examples.openai_agent_adapter.AsyncOpenAI") as mock_client_class:
            # Create mock with multiple tool calls
            tool_call_1 = Mock()
            tool_call_1.id = "call_1"
            tool_call_1.function.name = "get_pricing_info"
            tool_call_1.function.arguments = '{"plan_name": "basic"}'

            tool_call_2 = Mock()
            tool_call_2.id = "call_2"
            tool_call_2.function.name = "search_documentation"
            tool_call_2.function.arguments = '{"query": "features"}'

            mock_message = Mock()
            mock_message.content = None
            mock_message.tool_calls = [tool_call_1, tool_call_2]

            first_response = Mock()
            first_response.choices = [Mock(message=mock_message)]

            # Final response after both tools
            final_message = Mock()
            final_message.content = "Here's the information you requested."
            final_message.tool_calls = None

            final_response = Mock()
            final_response.choices = [Mock(message=final_message)]

            # Setup mock client
            mock_client = AsyncMock()
            mock_client.chat.completions.create = AsyncMock(
                side_effect=[first_response, final_response]
            )
            mock_client_class.return_value = mock_client

            # Create adapter
            adapter = OpenAIAgentAdapter(
                api_key="test-key",
                enable_functions=True
            )

            # Test
            response = await adapter(
                input="Give me basic plan pricing and feature docs",
                turns=[],
                thread_id="test-parallel"
            )

            # Verify both tools were executed
            second_call_args = mock_client.chat.completions.create.call_args_list[1]
            messages = second_call_args.kwargs["messages"]

            tool_results = [m for m in messages if m.get("role") == "tool"]
            assert len(tool_results) == 2
            assert tool_results[0]["tool_call_id"] == "call_1"
            assert tool_results[1]["tool_call_id"] == "call_2"

    @pytest.mark.asyncio
    async def test_error_handling(self):
        """Test error handling when API call fails."""
        with patch("examples.openai_agent_adapter.AsyncOpenAI") as mock_client_class:
            # Setup mock to raise an exception
            mock_client = AsyncMock()
            mock_client.chat.completions.create = AsyncMock(
                side_effect=Exception("API Error: Rate limit exceeded")
            )
            mock_client_class.return_value = mock_client

            # Create adapter
            adapter = OpenAIAgentAdapter(api_key="test-key")

            # Test error handling
            response = await adapter(
                input="Hello",
                turns=[],
                thread_id="test-error"
            )

            # Should return error message as Turn
            assert response.role == "assistant"
            assert "error" in response.content.lower()
            assert "Rate limit exceeded" in response.content

    @pytest.mark.asyncio
    async def test_api_key_validation(self):
        """Test that API key is required."""
        with patch.dict("os.environ", {}, clear=True):
            # Should raise ValueError when no API key
            with pytest.raises(ValueError, match="OpenAI API key required"):
                OpenAIAgentAdapter(api_key=None)

    @pytest.mark.asyncio
    async def test_custom_system_prompt(self, mock_openai_response):
        """Test custom system prompt is used."""
        with patch("examples.openai_agent_adapter.AsyncOpenAI") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.chat.completions.create = AsyncMock(return_value=mock_openai_response)
            mock_client_class.return_value = mock_client

            custom_prompt = "You are a pricing specialist."
            adapter = OpenAIAgentAdapter(
                api_key="test-key",
                system_prompt=custom_prompt
            )

            await adapter(input="Hello", turns=[], thread_id="test")

            # Verify system prompt was used
            call_args = mock_client.chat.completions.create.call_args
            messages = call_args.kwargs["messages"]
            assert messages[0]["role"] == "system"
            assert messages[0]["content"] == custom_prompt

    @pytest.mark.asyncio
    async def test_agent_callback_function(self, mock_openai_response):
        """Test the standalone agent_callback function."""
        with patch("examples.openai_agent_adapter.AsyncOpenAI") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.chat.completions.create = AsyncMock(return_value=mock_openai_response)
            mock_client_class.return_value = mock_client

            # Test the exported function
            response = await agent_callback(
                input="Test message",
                turns=[],
                thread_id="test-callback"
            )

            assert response.role == "assistant"
            assert isinstance(response, Turn)


class TestToolExecution:
    """Test suite for tool execution logic."""

    def test_execute_pricing_function(self):
        """Test pricing info function execution."""
        adapter = OpenAIAgentAdapter(api_key="test-key")

        # Test each plan
        result = adapter._execute_function(
            "get_pricing_info",
            '{"plan_name": "basic"}'
        )
        assert result["price"] == "$10/month"
        assert "Feature A" in result["features"]

        result = adapter._execute_function(
            "get_pricing_info",
            '{"plan_name": "pro"}'
        )
        assert result["price"] == "$25/month"

        result = adapter._execute_function(
            "get_pricing_info",
            '{"plan_name": "enterprise"}'
        )
        assert result["price"] == "Custom"

    def test_execute_search_function(self):
        """Test documentation search function execution."""
        adapter = OpenAIAgentAdapter(api_key="test-key")

        result = adapter._execute_function(
            "search_documentation",
            '{"query": "authentication"}'
        )
        assert "results" in result
        assert len(result["results"]) > 0
        assert "authentication" in result["results"][0]

    def test_execute_unknown_function(self):
        """Test handling of unknown function."""
        adapter = OpenAIAgentAdapter(api_key="test-key")

        result = adapter._execute_function(
            "unknown_function",
            '{"param": "value"}'
        )
        assert "error" in result
        assert result["error"] == "Unknown function"


if __name__ == "__main__":
    # Allow running directly with: python tests/test_openai_agent_adapter.py
    pytest.main([__file__, "-v"])
