"""Example of using custom metrics with the evaluation system.

This example demonstrates:
1. How to use built-in custom metrics (ConversationLengthMetric, OnboardingClarityMetric)
2. How to create your own custom metric from scratch
3. How to combine multiple custom metrics in one evaluation

Run this example:
    python examples/custom_metrics_example.py
"""

import asyncio
from pathlib import Path

# Import simulation and evaluation components
from src.simulation.config_loader import load_personas_from_yaml
from src.simulation.runner import SimulationRunner
from src.evaluation.judge import JudgeEvaluator
from src.evaluation.report import ReportGenerator

# Import example agent
from examples.example_agent import example_agent_callback

# Import built-in custom metrics
from src.metrics import ConversationLengthMetric, OnboardingClarityMetric


async def example_1_basic_custom_metrics():
    """Example 1: Using built-in custom metrics."""
    print("\n" + "="*60)
    print("Example 1: Using Built-in Custom Metrics")
    print("="*60 + "\n")

    # Load personas (using first 2 for quick demo)
    goldens, configs = load_personas_from_yaml("config/personas.yaml")
    goldens = goldens[:6]  # Take just 6 conversations for demo
    configs = configs[:2]   # From first 2 personas

    # Run simulations
    print("Running simulations...")
    runner = SimulationRunner(agent_callback=example_agent_callback)
    test_cases = await runner.run_simulations(goldens, configs)
    print(f"✓ Generated {len(test_cases)} conversations\n")

    # Create evaluator with custom metrics
    evaluator = JudgeEvaluator(
        threshold=0.7,
        use_default_metrics=True,  # Include standard metrics
        custom_metrics=[
            ConversationLengthMetric(
                max_turns=12,
                min_turns=3,
                threshold=0.5
            )
        ]
    )

    # Evaluate
    print("Evaluating with custom metrics...")
    results = evaluator.evaluate_conversations(test_cases, verbose=True)

    # Show summary
    summary = evaluator.get_summary(results)
    print(f"\nPass rate: {summary['pass_rate']*100:.1f}%")
    print(f"Metrics evaluated: {len(evaluator.metrics)}")


async def example_2_llm_judge_custom_metric():
    """Example 2: Using LLM-as-a-judge custom metric."""
    print("\n" + "="*60)
    print("Example 2: Using OnboardingClarityMetric (LLM Judge)")
    print("="*60 + "\n")
    print("Note: This requires OPENAI_API_KEY and will make API calls\n")

    # Load just one persona focused on onboarding
    goldens, configs = load_personas_from_yaml("config/personas.yaml")

    # Find the "confused_first_timer" persona
    confused_goldens = [
        g for g in goldens
        if g.additional_metadata.get('persona_name') == 'confused_first_timer'
    ][:3]  # Just 3 conversations

    if not confused_goldens:
        print("Skipping - no confused_first_timer persona found")
        return

    # Run simulations
    print("Running simulations...")
    runner = SimulationRunner(agent_callback=example_agent_callback)
    test_cases = await runner.simulate(
        conversational_goldens=confused_goldens,
        max_user_simulations=15
    )
    print(f"✓ Generated {len(test_cases)} conversations\n")

    # Create evaluator with OnboardingClarityMetric
    try:
        evaluator = JudgeEvaluator(
            threshold=0.6,
            use_default_metrics=False,  # Only use custom metrics
            custom_metrics=[
                OnboardingClarityMetric(
                    model="gpt-3.5-turbo",  # Use cheaper model for demo
                    threshold=0.6
                )
            ]
        )

        # Evaluate
        print("Evaluating onboarding clarity...")
        results = evaluator.evaluate_conversations(test_cases, verbose=True)

        # Show summary
        summary = evaluator.get_summary(results)
        print(f"\nOnboarding Clarity Results:")
        print(f"  Pass rate: {summary['pass_rate']*100:.1f}%")

    except ValueError as e:
        print(f"Skipping LLM judge example: {e}")
    except ImportError as e:
        print(f"Skipping LLM judge example: {e}")


def example_3_create_your_own_metric():
    """Example 3: Creating a custom metric from scratch."""
    print("\n" + "="*60)
    print("Example 3: Creating Your Own Custom Metric")
    print("="*60 + "\n")

    from deepeval.metrics.base_metric import BaseMetric
    from deepeval.test_case import ConversationalTestCase
    from deepeval.metrics.indicator import metric_progress_indicator

    class QuestionCountMetric(BaseMetric):
        """Custom metric that counts user questions."""

        def __init__(self, name: str = "Question Count", threshold: float = 0.5):
            self.name = name
            self.threshold = threshold

        def measure(
            self,
            test_case: ConversationalTestCase,
            _show_indicator: bool = True
        ) -> float:
            """Count questions asked by user."""
            with metric_progress_indicator(self, _show_indicator=_show_indicator):
                # Count messages with question marks from user
                user_questions = [
                    turn for turn in test_case.turns
                    if turn.role == "user" and "?" in turn.content
                ]

                question_count = len(user_questions)

                # Score based on whether user asked questions (engagement indicator)
                if question_count >= 2:
                    self.score = 1.0
                    self.reason = f"Good engagement: User asked {question_count} questions"
                elif question_count == 1:
                    self.score = 0.7
                    self.reason = f"Moderate engagement: User asked {question_count} question"
                else:
                    self.score = 0.3
                    self.reason = "Low engagement: User asked no questions"

                self.success = self.score >= self.threshold
                return self.score

        def is_successful(self) -> bool:
            return self.success

        @property
        def __name__(self):
            return self.name

    print("Created QuestionCountMetric:")
    print("  - Counts user questions (messages with '?')")
    print("  - Score 1.0 if ≥2 questions, 0.7 if 1 question, 0.3 if none")
    print("  - Indicates user engagement level\n")

    print("Usage:")
    print("""
    evaluator = JudgeEvaluator(
        custom_metrics=[
            QuestionCountMetric(threshold=0.5)
        ]
    )
    """)


async def example_4_multiple_custom_metrics():
    """Example 4: Using multiple custom metrics together."""
    print("\n" + "="*60)
    print("Example 4: Combining Multiple Custom Metrics")
    print("="*60 + "\n")

    # Load personas
    goldens, configs = load_personas_from_yaml("config/personas.yaml")
    goldens = goldens[:4]
    configs = configs[:1]

    # Run simulations
    print("Running simulations...")
    runner = SimulationRunner(agent_callback=example_agent_callback)
    test_cases = await runner.run_simulations(goldens, configs)
    print(f"✓ Generated {len(test_cases)} conversations\n")

    # Combine built-in and custom metrics
    evaluator = JudgeEvaluator(
        threshold=0.7,
        use_default_metrics=True,  # ConversationalGEval, Completeness, Relevancy
        custom_metrics=[
            ConversationLengthMetric(max_turns=15, threshold=0.5)
            # Add OnboardingClarityMetric if you have OPENAI_API_KEY:
            # OnboardingClarityMetric(model="gpt-3.5-turbo", threshold=0.6)
        ]
    )

    print(f"Total metrics: {len(evaluator.metrics)}")
    print("  Built-in:", len([m for m in evaluator.metrics if m not in (evaluator.custom_metrics or [])]))
    print("  Custom:", len(evaluator.custom_metrics))
    print()

    # Evaluate
    print("Evaluating...")
    results = evaluator.evaluate_conversations(test_cases, verbose=True)

    # Generate report
    report_gen = ReportGenerator(output_dir="reports/custom_metrics_demo")
    json_path, md_path = report_gen.generate_reports(
        evaluation_results=results,
        summary=evaluator.get_summary(results),
        report_name="multi_metric_demo"
    )

    print(f"\n[Reports generated]")
    print(f"  JSON: {json_path}")
    print(f"  Markdown: {md_path}")


async def main():
    """Run all examples."""
    print("\n" + "="*60)
    print("CUSTOM METRICS EXAMPLES")
    print("="*60)

    # Example 1: Basic usage with ConversationLengthMetric
    await example_1_basic_custom_metrics()

    # Example 2: LLM-based metric (requires API key)
    # Uncomment if you have OPENAI_API_KEY set:
    # await example_2_llm_judge_custom_metric()

    # Example 3: Creating your own metric (code demo only)
    example_3_create_your_own_metric()

    # Example 4: Multiple metrics together
    await example_4_multiple_custom_metrics()

    print("\n" + "="*60)
    print("ALL EXAMPLES COMPLETE!")
    print("="*60 + "\n")
    print("Next steps:")
    print("  1. Check reports/custom_metrics_demo/ for generated reports")
    print("  2. Try creating your own custom metric using the templates")
    print("  3. See src/metrics/custom_examples.py for implementation details")


if __name__ == "__main__":
    asyncio.run(main())
