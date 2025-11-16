"""Tools for the OpenAI agent.

These tools are decorated with LangChain's @tool decorator and can be
called by the agent to perform specific actions.
"""

from typing import Dict, Any
from langchain.tools import tool


@tool
def get_pricing_info(plan_name: str) -> Dict[str, Any]:
    """Get pricing information for different subscription plans.

    Args:
        plan_name: The name of the pricing plan (basic, pro, or enterprise)

    Returns:
        Dictionary containing price and features for the requested plan
    """
    # In a real implementation, this would query a database or external API
    pricing_data = {
        "basic": {
            "price": "$10/month",
            "features": [
                "Feature A: Basic analytics",
                "Feature B: Standard support",
                "Feature C: 5 GB storage"
            ],
            "users": "1 user",
            "description": "Perfect for individuals getting started"
        },
        "pro": {
            "price": "$25/month",
            "features": [
                "Feature A: Advanced analytics",
                "Feature B: Priority support",
                "Feature C: 50 GB storage",
                "Feature D: API access",
                "Feature E: Custom integrations"
            ],
            "users": "Up to 5 users",
            "description": "Best for small teams and growing businesses"
        },
        "enterprise": {
            "price": "Custom pricing",
            "features": [
                "All Pro features",
                "Feature F: Dedicated account manager",
                "Feature G: SLA guarantees",
                "Feature H: Unlimited storage",
                "Feature I: Advanced security",
                "Feature J: Custom development"
            ],
            "users": "Unlimited users",
            "description": "Tailored solutions for large organizations"
        }
    }

    plan = plan_name.lower()
    if plan in pricing_data:
        return pricing_data[plan]
    else:
        return {
            "error": f"Plan '{plan_name}' not found",
            "available_plans": list(pricing_data.keys())
        }


@tool
def search_documentation(query: str) -> Dict[str, Any]:
    """Search product documentation for specific topics.

    Args:
        query: The search query to find relevant documentation

    Returns:
        Dictionary containing search results with title, content, and URL
    """
    # In a real implementation, this would:
    # 1. Query a vector database (e.g., Pinecone, Weaviate)
    # 2. Use semantic search to find relevant docs
    # 3. Return actual documentation content

    # For now, we'll return simulated results based on the query
    docs_database = {
        "authentication": [
            {
                "title": "Getting Started with Authentication",
                "content": "Our platform supports OAuth 2.0, API keys, and JWT tokens for authentication. Set up your credentials in the developer portal.",
                "url": "https://docs.example.com/auth/getting-started",
                "relevance": 0.95
            },
            {
                "title": "API Key Management",
                "content": "Learn how to create, rotate, and revoke API keys securely. Best practices for key storage and rotation policies.",
                "url": "https://docs.example.com/auth/api-keys",
                "relevance": 0.87
            }
        ],
        "api": [
            {
                "title": "API Reference",
                "content": "Complete REST API documentation with endpoints, parameters, and response formats. Includes rate limiting and error handling.",
                "url": "https://docs.example.com/api/reference",
                "relevance": 0.92
            },
            {
                "title": "API Best Practices",
                "content": "Tips for efficient API usage, including batch requests, caching strategies, and webhook setup.",
                "url": "https://docs.example.com/api/best-practices",
                "relevance": 0.85
            }
        ],
        "features": [
            {
                "title": "Feature Overview",
                "content": "Comprehensive guide to all platform features including analytics, integrations, and customization options.",
                "url": "https://docs.example.com/features/overview",
                "relevance": 0.88
            },
            {
                "title": "Advanced Features",
                "content": "Deep dive into advanced features like custom workflows, automation, and enterprise integrations.",
                "url": "https://docs.example.com/features/advanced",
                "relevance": 0.83
            }
        ]
    }

    # Simple keyword matching (in production, use vector search)
    query_lower = query.lower()
    results = []

    for category, docs in docs_database.items():
        if category in query_lower or any(category in query_lower for category in category.split()):
            results.extend(docs)

    # If no specific match, return general results
    if not results:
        results = [
            {
                "title": f"Search Results for '{query}'",
                "content": f"We found general information related to your query. Our documentation covers authentication, API usage, and feature guides.",
                "url": "https://docs.example.com/search?q=" + query,
                "relevance": 0.70
            }
        ]

    # Sort by relevance
    results.sort(key=lambda x: x.get("relevance", 0), reverse=True)

    return {
        "query": query,
        "results_count": len(results),
        "results": results[:3]  # Return top 3 results
    }


# List of all tools for easy import
TOOLS = [get_pricing_info, search_documentation]
