"""FastAPI service for the example agent.

This service exposes a LangGraph agent over HTTP endpoints, allowing
clients to interact with the agent via REST API calls.

Run with:
    python examples/example_agent/run.py
"""

import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

from .models import ChatRequest, ChatResponse, HealthResponse
from .agent import get_agent, reset_agent

# Load environment variables
load_dotenv()

# Create FastAPI app
app = FastAPI(
    title="Example Agent Service",
    description="LangGraph-powered agent with tool calling capabilities",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify allowed origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", response_model=HealthResponse)
async def root():
    """Root endpoint with service information."""
    api_key_set = bool(os.getenv("OPENAI_API_KEY"))

    return HealthResponse(
        status="healthy",
        version="1.0.0",
        agent_configured=api_key_set
    )


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint.

    Returns:
        HealthResponse with service status
    """
    api_key_set = bool(os.getenv("OPENAI_API_KEY"))

    if not api_key_set:
        raise HTTPException(
            status_code=503,
            detail="Service not configured: OPENAI_API_KEY not set"
        )

    return HealthResponse(
        status="healthy",
        version="1.0.0",
        agent_configured=True
    )


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Chat endpoint for interacting with the agent.

    Args:
        request: ChatRequest containing the message and conversation history

    Returns:
        ChatResponse with the agent's response

    Raises:
        HTTPException: If the agent is not configured or encounters an error
    """
    try:
        # Get the agent instance
        agent = get_agent()

        # Call the agent with the message and history
        result = await agent.chat(
            message=request.message,
            history=request.history,
            conversation_id=request.conversation_id
        )

        return ChatResponse(**result)

    except ValueError as e:
        # Configuration error (e.g., missing API key)
        raise HTTPException(
            status_code=503,
            detail=f"Agent configuration error: {str(e)}"
        )
    except Exception as e:
        # Other errors
        raise HTTPException(
            status_code=500,
            detail=f"Agent error: {str(e)}"
        )


@app.post("/reset")
async def reset():
    """Reset the agent instance.

    Useful for testing or when configuration changes.

    Returns:
        Dictionary with reset status
    """
    reset_agent()
    return {"status": "success", "message": "Agent instance reset"}
