"""Tests for the example agent FastAPI service.

This test suite validates the example agent service API including:
- Health check endpoint
- Chat endpoint with/without history
- Tool calling integration
- Error handling
- Response format

Run with: pytest tests/test_example_agent_service.py -v
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, patch


@pytest.fixture
def client():
    """Create a test client for the FastAPI app."""
    from examples.example_agent.agent_service import app
    return TestClient(app)


@pytest.fixture
def mock_agent():
    """Mock the agent instance."""
    mock = AsyncMock()
    return mock


class TestHealthEndpoints:
    """Test health check endpoints."""

    def test_root_endpoint(self, client):
        """Test root endpoint returns health status."""
        response = client.get("/")
        assert response.status_code == 200

        data = response.json()
        assert data["status"] == "healthy"
        assert data["version"] == "1.0.0"
        assert "agent_configured" in data

    @patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"})
    def test_health_check_configured(self, client):
        """Test health check with API key configured."""
        response = client.get("/health")
        assert response.status_code == 200

        data = response.json()
        assert data["status"] == "healthy"
        assert data["agent_configured"] is True

    @patch.dict("os.environ", {}, clear=True)
    def test_health_check_not_configured(self, client):
        """Test health check fails when API key not set."""
        # Import here to ensure env vars are cleared
        from examples.example_agent.agent_service import app
        test_client = TestClient(app)

        response = test_client.get("/health")
        assert response.status_code == 503
        assert "OPENAI_API_KEY not set" in response.json()["detail"]


class TestChatEndpoint:
    """Test the main chat endpoint."""

    @patch("examples.example_agent.agent_service.get_agent")
    def test_chat_basic(self, mock_get_agent, client):
        """Test basic chat request."""
        # Setup mock agent
        mock_agent = AsyncMock()
        mock_agent.chat.return_value = {
            "message": "Hello! How can I help you?",
            "conversation_id": "test-123",
            "tool_calls": [],
            "metadata": {"model": "gpt-4o", "temperature": 0.7}
        }
        mock_get_agent.return_value = mock_agent

        # Make request
        response = client.post("/chat", json={
            "message": "Hello",
            "conversation_id": "test-123",
            "history": []
        })

        assert response.status_code == 200
        data = response.json()

        assert data["message"] == "Hello! How can I help you?"
        assert data["conversation_id"] == "test-123"
        assert data["tool_calls"] == []
        assert "metadata" in data

    @patch("examples.example_agent.agent_service.get_agent")
    def test_chat_with_history(self, mock_get_agent, client):
        """Test chat request with conversation history."""
        mock_agent = AsyncMock()
        mock_agent.chat.return_value = {
            "message": "I can help with pricing questions.",
            "conversation_id": "test-456",
            "tool_calls": [],
            "metadata": {"model": "gpt-4o", "temperature": 0.7}
        }
        mock_get_agent.return_value = mock_agent

        # Request with history
        response = client.post("/chat", json={
            "message": "What can you do?",
            "conversation_id": "test-456",
            "history": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi!"}
            ]
        })

        assert response.status_code == 200

        # Verify history was passed to agent
        call_args = mock_agent.chat.call_args
        assert call_args.kwargs["history"] is not None
        assert len(call_args.kwargs["history"]) == 2

    @patch("examples.example_agent.agent_service.get_agent")
    def test_chat_with_tool_calls(self, mock_get_agent, client):
        """Test chat response includes tool calls."""
        mock_agent = AsyncMock()
        mock_agent.chat.return_value = {
            "message": "The Pro plan costs $25/month.",
            "conversation_id": "test-789",
            "tool_calls": [
                {
                    "name": "get_pricing_info",
                    "arguments": {"plan_name": "pro"},
                    "result": {"price": "$25/month", "features": ["Feature A", "Feature B"]}
                }
            ],
            "metadata": {"model": "gpt-4o", "temperature": 0.7}
        }
        mock_get_agent.return_value = mock_agent

        response = client.post("/chat", json={
            "message": "What's the Pro plan pricing?",
            "conversation_id": "test-789",
            "history": []
        })

        assert response.status_code == 200
        data = response.json()

        assert data["message"] == "The Pro plan costs $25/month."
        assert len(data["tool_calls"]) == 1
        assert data["tool_calls"][0]["name"] == "get_pricing_info"
        assert data["tool_calls"][0]["arguments"]["plan_name"] == "pro"
        assert data["tool_calls"][0]["result"] is not None

    @patch("examples.example_agent.agent_service.get_agent")
    def test_chat_agent_error(self, mock_get_agent, client):
        """Test error handling when agent raises exception."""
        mock_agent = AsyncMock()
        mock_agent.chat.side_effect = Exception("Agent processing error")
        mock_get_agent.return_value = mock_agent

        response = client.post("/chat", json={
            "message": "Hello",
            "conversation_id": "test-error",
            "history": []
        })

        assert response.status_code == 500
        assert "Agent error" in response.json()["detail"]

    @patch("examples.example_agent.agent_service.get_agent")
    def test_chat_configuration_error(self, mock_get_agent, client):
        """Test error handling for configuration errors."""
        mock_agent = AsyncMock()
        mock_agent.chat.side_effect = ValueError("API key not configured")
        mock_get_agent.return_value = mock_agent

        response = client.post("/chat", json={
            "message": "Hello",
            "conversation_id": "test-config-error",
            "history": []
        })

        assert response.status_code == 503
        assert "Agent configuration error" in response.json()["detail"]

    def test_chat_invalid_request(self, client):
        """Test validation of request body."""
        # Missing required field 'message'
        response = client.post("/chat", json={
            "conversation_id": "test-invalid"
        })

        assert response.status_code == 422  # Validation error


class TestResetEndpoint:
    """Test agent reset endpoint."""

    @patch("examples.example_agent.agent_service.reset_agent")
    def test_reset(self, mock_reset, client):
        """Test reset endpoint."""
        response = client.post("/reset")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"

        # Verify reset was called
        mock_reset.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
