"""Base utilities for building custom metrics."""

import json
import re
from typing import Dict, Any
from deepeval.test_case import ConversationalTestCase


def format_conversation_for_judge(test_case: ConversationalTestCase) -> str:
    """Format a conversation as text for LLM judge input.

    Args:
        test_case: The conversation test case

    Returns:
        Formatted conversation string with role labels
    """
    lines = []
    for turn in test_case.turns:
        role = turn.role.upper()
        lines.append(f"{role}: {turn.content}")
    return "\n".join(lines)


def parse_json_from_llm(text: str) -> Dict[str, Any]:
    """Parse JSON from LLM response, handling common formatting issues.

    Args:
        text: Raw text response from LLM

    Returns:
        Parsed JSON dictionary

    Raises:
        ValueError: If JSON cannot be parsed
    """
    # Try direct JSON parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try to extract JSON from markdown code blocks
    json_pattern = r'```(?:json)?\s*(\{.*?\})\s*```'
    match = re.search(json_pattern, text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    # Try to find any JSON object in the text
    json_obj_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
    match = re.search(json_obj_pattern, text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    raise ValueError(f"Could not parse JSON from LLM response: {text[:200]}")


def extract_score_from_text(text: str) -> float:
    """Extract a numeric score from text response.

    Looks for patterns like "Score: 0.85" or "8.5/10".

    Args:
        text: Text potentially containing a score

    Returns:
        Extracted score (0.0 to 1.0)

    Raises:
        ValueError: If no score found
    """
    # Look for "score: X" or "score = X"
    score_pattern = r'score[:\s=]+(\d+\.?\d*)'
    match = re.search(score_pattern, text.lower())
    if match:
        score = float(match.group(1))
        # Normalize if > 1.0 (assume out of 10 or 100)
        if score > 1.0:
            if score <= 10:
                return score / 10.0
            elif score <= 100:
                return score / 100.0
        return score

    # Look for "X/10" or "X out of 10"
    ratio_pattern = r'(\d+\.?\d*)\s*(?:/|out of)\s*(\d+)'
    match = re.search(ratio_pattern, text.lower())
    if match:
        numerator = float(match.group(1))
        denominator = float(match.group(2))
        return numerator / denominator if denominator > 0 else 0.0

    raise ValueError(f"Could not extract score from text: {text[:200]}")


def create_judge_prompt(
    conversation: str,
    criteria: str,
    instruction: str = "Evaluate this conversation and provide a score from 0 to 1."
) -> str:
    """Create a standard judge prompt format.

    Args:
        conversation: Formatted conversation text
        criteria: Evaluation criteria
        instruction: Instructions for the judge

    Returns:
        Formatted prompt string
    """
    return f"""You are evaluating an AI assistant's conversation with a user.

{instruction}

{criteria}

Conversation:
{conversation}

Respond with a JSON object in this format:
{{"score": <number between 0 and 1>, "explanation": "<brief explanation>"}}
"""
