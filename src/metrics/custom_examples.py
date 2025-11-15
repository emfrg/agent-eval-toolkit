"""Example custom metrics for conversation evaluation.

This module provides two example metrics:
1. ConversationLengthMetric - Code-based metric (no LLM call)
2. OnboardingClarityMetric - LLM-based metric (custom rubric)

Use these as templates for creating your own custom metrics.
"""

import os
from typing import Optional

from deepeval.metrics.base_metric import BaseMetric
from deepeval.test_case import ConversationalTestCase, LLMTestCase
from deepeval.metrics.indicator import metric_progress_indicator

from .base import format_conversation_for_judge, parse_json_from_llm, create_judge_prompt


class ConversationLengthMetric(BaseMetric):
    """Code-based metric that checks if conversation length is within limits.

    This is an example of a deterministic, code-based metric that doesn't
    require an LLM call. It simply counts turns and compares to a threshold.

    Example:
        ```python
        from src.metrics import ConversationLengthMetric

        metric = ConversationLengthMetric(max_turns=12, min_turns=3)
        evaluator = JudgeEvaluator(custom_metrics=[metric])
        ```

    Args:
        max_turns: Maximum allowed turns (default: 15)
        min_turns: Minimum required turns (default: 1)
        name: Metric name (default: "Conversation Length")
    """

    def __init__(
        self,
        max_turns: int = 15,
        min_turns: int = 1,
        threshold: float = 0.5,
        name: str = "Conversation Length",
    ):
        self.max_turns = max_turns
        self.min_turns = min_turns
        self.threshold = threshold
        self.name = name

    def measure(self, test_case: ConversationalTestCase, _show_indicator: bool = True) -> float:
        """Measure conversation length.

        Args:
            test_case: The conversation to evaluate

        Returns:
            Score: 1.0 if within bounds, 0.5 if too short/long, 0.0 if way off
        """
        with metric_progress_indicator(self, _show_indicator=_show_indicator):
            num_turns = len(test_case.turns)

            # Calculate score based on length
            if self.min_turns <= num_turns <= self.max_turns:
                self.score = 1.0
                self.reason = (
                    f"✓ Conversation length ({num_turns} turns) is within "
                    f"optimal range ({self.min_turns}-{self.max_turns} turns)"
                )
            elif num_turns < self.min_turns:
                self.score = 0.5
                self.reason = (
                    f"⚠ Conversation too short ({num_turns} turns). "
                    f"Minimum: {self.min_turns} turns"
                )
            elif num_turns > self.max_turns:
                # Penalty increases with distance from max
                excess = num_turns - self.max_turns
                if excess <= 3:
                    self.score = 0.5
                    self.reason = (
                        f"⚠ Conversation slightly too long ({num_turns} turns). "
                        f"Maximum: {self.max_turns} turns"
                    )
                else:
                    self.score = 0.0
                    self.reason = (
                        f"✗ Conversation too long ({num_turns} turns, "
                        f"{excess} over limit). Maximum: {self.max_turns} turns"
                    )

            self.success = self.score >= self.threshold
            return self.score

    def is_successful(self) -> bool:
        """Check if metric passed threshold."""
        return self.success

    @property
    def __name__(self):
        return self.name


class OnboardingClarityMetric(BaseMetric):
    """LLM-based metric that evaluates onboarding clarity with a custom rubric.

    This is an example of an LLM-as-a-judge metric with a domain-specific
    rubric. It uses OpenAI to evaluate how clearly the agent guides new users.

    Example:
        ```python
        from src.metrics import OnboardingClarityMetric

        metric = OnboardingClarityMetric(
            model="gpt-4o",
            threshold=0.7
        )
        evaluator = JudgeEvaluator(custom_metrics=[metric])
        ```

    Args:
        model: LLM model to use for judging (default: gpt-4o)
        threshold: Pass/fail threshold (default: 0.7)
        name: Metric name (default: "Onboarding Clarity")
    """

    def __init__(
        self, model: str = "gpt-4o", threshold: float = 0.7, name: str = "Onboarding Clarity"
    ):
        self.model = model
        self.threshold = threshold
        self.name = name

        # Initialize OpenAI client
        try:
            from openai import OpenAI

            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError(
                    "OPENAI_API_KEY environment variable not set. "
                    "Required for OnboardingClarityMetric."
                )
            self.client = OpenAI(api_key=api_key)
        except ImportError:
            raise ImportError(
                "OpenAI package required for OnboardingClarityMetric. "
                "Install with: pip install openai"
            )

    def measure(self, test_case: ConversationalTestCase, _show_indicator: bool = True) -> float:
        """Evaluate onboarding clarity using LLM judge.

        Args:
            test_case: The conversation to evaluate

        Returns:
            Score from 0.0 to 1.0 based on onboarding clarity
        """
        with metric_progress_indicator(self, _show_indicator=_show_indicator):
            # Format conversation for judge
            conversation_text = format_conversation_for_judge(test_case)

            # Create judge prompt with custom criteria
            criteria = """
Evaluation Criteria for Onboarding Clarity:

1. **Step-by-Step Guidance (40%)**
   - Does the assistant break down complex tasks into clear steps?
   - Are instructions numbered or clearly sequenced?
   - Does the assistant check for understanding before proceeding?

2. **Jargon-Free Language (30%)**
   - Are technical terms explained or avoided?
   - Is language appropriate for a first-time user?
   - Are acronyms spelled out?

3. **Proactive Help (20%)**
   - Does the assistant anticipate common questions?
   - Are relevant resources or next steps offered?
   - Does the assistant provide context for actions?

4. **Patience and Encouragement (10%)**
   - Is the tone welcoming and supportive?
   - Does the assistant show patience with repeated questions?
   - Are positive reinforcements used?

Score from 0.0 (very confusing) to 1.0 (exceptionally clear onboarding).
"""

            prompt = create_judge_prompt(
                conversation=conversation_text,
                criteria=criteria,
                instruction="Evaluate how well this conversation guides a new user.",
            )

            # Call judge LLM
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are an expert evaluator of user onboarding experiences.",
                        },
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.0,
                )

                # Parse response
                result_text = response.choices[0].message.content
                result_data = parse_json_from_llm(result_text)

                self.score = float(result_data.get("score", 0.0))
                self.reason = result_data.get("explanation", "No explanation provided")

            except Exception as e:
                # Handle errors gracefully
                self.score = 0.0
                self.reason = f"Error during evaluation: {str(e)}"

            self.success = self.score >= self.threshold
            return self.score

    def is_successful(self) -> bool:
        """Check if metric passed threshold."""
        return self.success

    @property
    def __name__(self):
        return self.name


# Example: Creating a custom metric from scratch
#
# class MyCustomMetric(BaseMetric):
#     """Template for your own custom metric."""
#
#     def __init__(self, threshold: float = 0.5, name: str = "My Custom Metric"):
#         self.threshold = threshold
#         self.name = name
#
#     def measure(self, test_case: ConversationalTestCase, _show_indicator: bool = True) -> float:
#         """Implement your evaluation logic here."""
#         with metric_progress_indicator(self, _show_indicator=_show_indicator):
#             # Your evaluation logic
#             self.score = 0.0  # Calculate your score
#             self.reason = "Explanation of score"
#             self.success = self.score >= self.threshold
#             return self.score
#
#     def is_successful(self) -> bool:
#         return self.success
#
#     @property
#     def __name__(self):
#         return self.name
