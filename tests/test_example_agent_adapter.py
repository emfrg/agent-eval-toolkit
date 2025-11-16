"""Tests for the example agent adapter.

This test suite validates the adapter that connects the evaluation framework
to the agent service via HTTP including:
- Service connectivity
- Turn conversion (Service Response → DeepEval Turn)
- History formatting
- Error handling (timeout, connection errors)
- Mock service responses

Run with: pytest tests/test_example_agent_adapter.py -v
"""

import pytest
from unittest.mock import AsyncMock, patch, Mock
from deepeval.test_case import Turn
import httpx


# Import the adapter
from examples.example_agent_adapter import agent_callback


class TestAgentCallback:
    """Test the agent_callback function."""

    @pytest.mark.asyncio
    async def test_basic_conversation(self):
        """Test basic conversation with service."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "message": "Hello! How can I help you?",
            "conversation_id": "test-123",
            "tool_calls": [],
            "metadata": {"model": "gpt-4o"}
        }

        with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_response

            # Call adapter
            result = await agent_callback(
                input="Hello",
                turns=[],
                thread_id="test-123"
            )

            # Verify result is a Turn
            assert isinstance(result, Turn)
            assert result.role == "assistant"
            assert result.content == "Hello! How can I help you?"

            # Verify service was called correctly
            mock_post.assert_called_once()
            call_args = mock_post.call_args
            assert "http://localhost:5555/chat" in str(call_args)

    @pytest.mark.asyncio
    async def test_conversation_with_history(self):
        """Test adapter converts history to service format."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "message": "Sure, I can help with that.",
            "conversation_id": "test-456",
            "tool_calls": [],
            "metadata": {}
        }

        with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_response

            # Create history
            turns = [
                Turn(role="user", content="Hello"),
                Turn(role="assistant", content="Hi there!")
            ]

            result = await agent_callback(
                input="Can you help me?",
                turns=turns,
                thread_id="test-456"
            )

            # Verify history was included in request
            call_args = mock_post.call_args
            payload = call_args.kwargs["json"]

            assert "history" in payload
            assert len(payload["history"]) == 2
            assert payload["history"][0]["role"] == "user"
            assert payload["history"][0]["content"] == "Hello"
            assert payload["history"][1]["role"] == "assistant"
            assert payload["history"][1]["content"] == "Hi there!"

    @pytest.mark.asyncio
    async def test_service_error_handling(self):
        """Test adapter handles service errors gracefully."""
        mock_response = Mock()
        mock_response.status_code = 500
        mock_response.json.return_value = {
            "detail": "Internal server error"
        }

        with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_response

            result = await agent_callback(
                input="Hello",
                turns=[],
                thread_id="test-error"
            )

            # Should return error message as Turn
            assert isinstance(result, Turn)
            assert result.role == "assistant"
            assert "Service error" in result.content
            assert "500" in result.content

    @pytest.mark.asyncio
    async def test_connection_error(self):
        """Test adapter handles connection errors."""
        with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
            mock_post.side_effect = httpx.ConnectError("Connection refused")

            result = await agent_callback(
                input="Hello",
                turns=[],
                thread_id="test-connect-error"
            )

            # Should return helpful error message
            assert isinstance(result, Turn)
            assert result.role == "assistant"
            assert "Cannot connect" in result.content
            assert "python examples/example_agent/run.py" in result.content

    @pytest.mark.asyncio
    async def test_timeout_error(self):
        """Test adapter handles timeout errors."""
        with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
            mock_post.side_effect = httpx.TimeoutException("Request timeout")

            result = await agent_callback(
                input="Hello",
                turns=[],
                thread_id="test-timeout"
            )

            # Should return timeout message
            assert isinstance(result, Turn)
            assert result.role == "assistant"
            assert "timeout" in result.content.lower()

    @pytest.mark.asyncio
    async def test_unexpected_error(self):
        """Test adapter handles unexpected errors."""
        with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
            mock_post.side_effect = Exception("Unexpected error")

            result = await agent_callback(
                input="Hello",
                turns=[],
                thread_id="test-unexpected"
            )

            # Should return error message
            assert isinstance(result, Turn)
            assert result.role == "assistant"
            assert "Adapter error" in result.content

    @pytest.mark.asyncio
    async def test_payload_format(self):
        """Test adapter sends correct payload format."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "message": "Response",
            "conversation_id": "test-789",
            "tool_calls": [],
            "metadata": {}
        }

        with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_response

            await agent_callback(
                input="Test message",
                turns=[],
                thread_id="test-789"
            )

            # Verify payload structure
            call_args = mock_post.call_args
            payload = call_args.kwargs["json"]

            assert "message" in payload
            assert payload["message"] == "Test message"
            assert "conversation_id" in payload
            assert payload["conversation_id"] == "test-789"
            assert "history" in payload
            assert isinstance(payload["history"], list)

    @pytest.mark.asyncio
    async def test_turn_conversion(self):
        """Test adapter correctly converts service response to Turn."""
        service_responses = [
            {
                "message": "Short answer",
                "conversation_id": "test",
                "tool_calls": [],
                "metadata": {}
            },
            {
                "message": "Long answer with multiple sentences. This is a detailed response.",
                "conversation_id": "test",
                "tool_calls": [
                    {"name": "tool1", "arguments": {}, "result": None}
                ],
                "metadata": {"model": "gpt-4o"}
            }
        ]

        for service_response in service_responses:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = service_response

            with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
                mock_post.return_value = mock_response

                result = await agent_callback(
                    input="Test",
                    turns=[],
                    thread_id="test"
                )

                # Verify Turn structure
                assert isinstance(result, Turn)
                assert result.role == "assistant"
                assert result.content == service_response["message"]
                assert isinstance(result.content, str)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
