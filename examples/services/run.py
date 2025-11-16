#!/usr/bin/env python
"""Launcher script for the OpenAI agent service.

This script runs the FastAPI service with sensible defaults:
- Default port: 5555
- Auto-reload enabled
- Can override with environment variables

Usage:
    python examples/services/run.py

    PORT=3000 python examples/services/run.py
    HOST=localhost python examples/services/run.py
"""

import os
import sys
from pathlib import Path

# Add project root to Python path so imports work
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Get port and host from environment or use defaults
PORT = int(os.getenv("PORT", "5555"))
HOST = os.getenv("HOST", "0.0.0.0")

# Run uvicorn
if __name__ == "__main__":
    import uvicorn

    print(f"""
╔══════════════════════════════════════════════════════════════════╗
║              OpenAI Agent Service                                 ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                   ║
║  Starting service...                                              ║
║                                                                   ║
║  📝 API Documentation:                                            ║
║     http://localhost:{PORT}/docs{' ' * (44 - len(str(PORT)))}║
║                                                                   ║
║  🔍 Health Check:                                                 ║
║     http://localhost:{PORT}/health{' ' * (42 - len(str(PORT)))}║
║                                                                   ║
║  💬 Chat Endpoint:                                                ║
║     POST http://localhost:{PORT}/chat{' ' * (44 - len(str(PORT)))}║
║                                                                   ║
║  ⚙️  Configuration:                                               ║
║     Make sure OPENAI_API_KEY is set in your .env file            ║
║                                                                   ║
╚══════════════════════════════════════════════════════════════════╝
""")

    uvicorn.run(
        "examples.services.openai_agent_service:app",
        host=HOST,
        port=PORT,
        reload=True,
        reload_dirs=[str(project_root)]  # Watch project root for changes
    )
