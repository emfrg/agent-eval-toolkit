"""LLM-as-a-judge evaluation using DeepEval metrics."""

import os
from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import datetime

from deepeval.test_case import ConversationalTestCase
from deepeval.metrics import (
    ConversationalGEval,
    ConversationCompletenessMetric,
    TurnRelevancyMetric,
)
from deepeval import evaluate
from deepeval.evaluate import DisplayConfig
from rich.console import Console
from rich.table import Table

console = Console()


class JudgeEvaluator:
    """Evaluates conversations using LLM-as-a-judge metrics.

    This class wraps DeepEval's conversation evaluation metrics and provides:
    - Customizable judge prompts
    - Multiple evaluation criteria
    - Batch evaluation
    - Results aggregation

    Example:
        ```python
        evaluator = JudgeEvaluator(
            custom_criteria="The assistant should be helpful and professional"
        )

        results = await evaluator.evaluate_conversations(test_cases)
        print(evaluator.get_summary(results))
        ```
    """

    def __init__(
        self,
        custom_criteria: Optional[str] = None,
        custom_prompt_path: Optional[str | Path] = None,
        judge_model: Optional[str] = None,
        threshold: float = 0.7,
        use_default_metrics: bool = True,
        custom_metrics: Optional[List] = None,
    ):
        """Initialize the judge evaluator.

        Args:
            custom_criteria: Custom evaluation criteria (overrides prompt file)
            custom_prompt_path: Path to markdown file with judge prompt
            judge_model: LLM model to use for judging (default: from env or gpt-4o)
            threshold: Pass/fail threshold for metrics (0.0 to 1.0)
            use_default_metrics: Include default metrics (completeness, relevancy)
            custom_metrics: List of custom metric instances to include in evaluation

        Example with custom metrics:
            ```python
            from src.metrics import ConversationLengthMetric, OnboardingClarityMetric

            evaluator = JudgeEvaluator(
                threshold=0.7,
                custom_metrics=[
                    ConversationLengthMetric(max_turns=12),
                    OnboardingClarityMetric(model="gpt-4o")
                ]
            )
            ```
        """
        self.threshold = threshold
        self.judge_model = judge_model or os.getenv("JUDGE_MODEL", "gpt-4o")
        self.use_default_metrics = use_default_metrics
        self.custom_metrics = custom_metrics or []

        # Load custom criteria
        if custom_criteria:
            self.criteria = custom_criteria
        elif custom_prompt_path:
            self.criteria = self._load_prompt(custom_prompt_path)
        else:
            self.criteria = self._get_default_criteria()

        # Initialize metrics
        self.metrics = self._create_metrics()

    def _load_prompt(self, prompt_path: str | Path) -> str:
        """Load judge prompt from markdown file."""
        prompt_path = Path(prompt_path)

        if not prompt_path.exists():
            console.print(
                f"[yellow]Warning:[/yellow] Judge prompt file not found: {prompt_path}. "
                "Using default criteria."
            )
            return self._get_default_criteria()

        with open(prompt_path, "r") as f:
            content = f.read().strip()

        console.print(f"[green]✓[/green] Loaded custom judge prompt from {prompt_path}")
        return content

    def _get_default_criteria(self) -> str:
        """Get default evaluation criteria."""
        return """
        Evaluate the conversation based on the following criteria:

        1. **Helpfulness**: The assistant provides useful, relevant information that addresses the user's needs
        2. **Clarity**: Responses are clear, well-structured, and easy to understand
        3. **Accuracy**: Information provided is correct and not misleading
        4. **Professionalism**: The assistant maintains a professional, friendly tone
        5. **Efficiency**: The assistant resolves queries without unnecessary back-and-forth

        Provide a score from 0 to 1, where 1 is excellent and 0 is poor.
        """

    def _create_metrics(self) -> List:
        """Create evaluation metrics based on configuration."""
        metrics = []

        # Custom G-Eval metric with user-defined criteria
        custom_metric = ConversationalGEval(
            name="Overall Quality",
            criteria=self.criteria,
            threshold=self.threshold,
            model=self.judge_model,
        )
        metrics.append(custom_metric)

        # Add default metrics if requested
        if self.use_default_metrics:
            # Evaluate if the chatbot satisfied user needs
            completeness = ConversationCompletenessMetric(
                threshold=self.threshold, model=self.judge_model
            )
            metrics.append(completeness)

            # Evaluate if responses were relevant throughout conversation
            relevancy = TurnRelevancyMetric(threshold=self.threshold, model=self.judge_model)
            metrics.append(relevancy)

        # Add any custom metrics provided by user
        if self.custom_metrics:
            metrics.extend(self.custom_metrics)
            metric_names = [
                m.name if hasattr(m, "name") else m.__class__.__name__ for m in self.custom_metrics
            ]
            console.print(
                f"[green]✓[/green] Added {len(self.custom_metrics)} custom metric(s): "
                f"{', '.join(metric_names)}"
            )

        return metrics

    def evaluate_conversations(
        self, test_cases: List[ConversationalTestCase], verbose: bool = True
    ) -> List[Dict[str, Any]]:
        """Evaluate a list of conversations.

        Args:
            test_cases: List of ConversationalTestCase objects to evaluate
            verbose: Print progress and results

        Returns:
            List of evaluation results with scores and metadata
        """
        if verbose:
            console.print(f"\n[bold blue]Evaluating {len(test_cases)} conversations...[/bold blue]")
            console.print(f"Judge model: {self.judge_model}")
            console.print(f"Threshold: {self.threshold}")
            console.print(f"Metrics: {len(self.metrics)}\n")

        # Run DeepEval evaluation
        display_config = DisplayConfig(
            verbose_mode=False,
            print_results=False
        )

        try:
            evaluation_result = evaluate(
                test_cases=test_cases,
                metrics=self.metrics,
                display_config=display_config
            )
        except Exception as e:
            error_name = type(e).__name__
            error_msg = str(e)
            if "ConfidentApiError" in error_name or "Invalid API key" in error_msg:
                console.print(
                    "\n[red]✗ Error:[/red] DeepEval tried to upload results to Confident AI cloud but failed."
                )
                console.print(
                    "[yellow]Solution:[/yellow] Remove or comment out CONFIDENT_API_KEY from your .env file."
                )
                console.print(
                    "[dim]Confident AI is an optional paid service. Evaluations run locally without it.[/dim]"
                )
                raise RuntimeError(
                    "Confident AI API key is invalid. Remove CONFIDENT_API_KEY from .env to run locally."
                ) from e
            else:
                raise

        # Process and structure results
        structured_results = []

        # Extract scores from evaluation_result.test_results
        for i, (test_case, test_result) in enumerate(zip(test_cases, evaluation_result.test_results)):
            result = {
                "test_case_index": i,
                "scenario": test_case.scenario,
                "expected_outcome": test_case.expected_outcome,
                "num_turns": len(test_case.turns),
                "metadata": getattr(test_case, "additional_metadata", {}),
                "metric_scores": {},
                "passed": True,
                "evaluated_at": datetime.now().isoformat(),
            }

            # Extract metric scores from test_result.metrics_data
            if test_result.metrics_data:
                for metric_data in test_result.metrics_data:
                    result["metric_scores"][metric_data.name] = {
                        "score": metric_data.score if metric_data.score is not None else 0.0,
                        "threshold": metric_data.threshold,
                        "passed": metric_data.success if metric_data.success is not None else False,
                    }

                    # Overall pass/fail
                    if metric_data.score is not None and metric_data.score < self.threshold:
                        result["passed"] = False

            structured_results.append(result)

        if verbose:
            self._print_results_table(structured_results)

        return structured_results

    def _print_results_table(self, results: List[Dict[str, Any]]):
        """Print evaluation results in a nice table."""
        table = Table(title="Evaluation Results")

        table.add_column("Index", style="cyan")
        table.add_column("Scenario", style="magenta")
        table.add_column("Turns", justify="right")
        table.add_column("Overall Quality", justify="right")
        table.add_column("Completeness", justify="right")
        table.add_column("Relevancy", justify="right")
        table.add_column("Passed", justify="center")

        for result in results:
            metrics = result["metric_scores"]

            # Use the actual metric names that DeepEval returns
            overall_score = metrics.get("Overall Quality [Conversational GEval]", {}).get("score")
            if overall_score is None:
                overall_score = metrics.get("Overall Quality", {}).get("score")

            completeness_score = metrics.get("Conversation Completeness", {}).get("score")
            if completeness_score is None:
                completeness_score = metrics.get("ConversationCompletenessMetric", {}).get("score")

            relevancy_score = metrics.get("Turn Relevancy", {}).get("score")
            if relevancy_score is None:
                relevancy_score = metrics.get("TurnRelevancyMetric", {}).get("score")

            # Truncate scenario for display
            scenario = (
                result["scenario"][:40] + "..."
                if len(result["scenario"]) > 40
                else result["scenario"]
            )

            table.add_row(
                str(result["test_case_index"]),
                scenario,
                str(result["num_turns"]),
                f"{overall_score:.2f}" if overall_score is not None else "N/A",
                f"{completeness_score:.2f}" if completeness_score is not None else "N/A",
                f"{relevancy_score:.2f}" if relevancy_score is not None else "N/A",
                "✓" if result["passed"] else "✗",
            )

        console.print("\n")
        console.print(table)
        console.print("\n")

    def get_summary(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Get summary statistics from evaluation results.

        Args:
            results: List of evaluation results

        Returns:
            Dict with summary statistics
        """
        if not results:
            return {"error": "No results to summarize"}

        total = len(results)
        passed = sum(1 for r in results if r["passed"])
        pass_rate = passed / total if total > 0 else 0

        # Calculate average scores per metric
        metric_averages = {}
        for metric_name in results[0]["metric_scores"].keys():
            scores = [
                r["metric_scores"][metric_name]["score"]
                for r in results
                if r["metric_scores"][metric_name]["score"] is not None
            ]
            if scores:
                metric_averages[metric_name] = {
                    "average": sum(scores) / len(scores),
                    "min": min(scores),
                    "max": max(scores),
                    "count": len(scores),
                }

        # Group by persona if metadata available
        persona_stats = {}
        for result in results:
            persona = result["metadata"].get("persona_name", "unknown")
            if persona not in persona_stats:
                persona_stats[persona] = {"total": 0, "passed": 0, "scores": []}

            persona_stats[persona]["total"] += 1
            if result["passed"]:
                persona_stats[persona]["passed"] += 1

            # Collect overall quality score
            overall_score = result["metric_scores"].get("Overall Quality [Conversational GEval]", {}).get("score")
            if overall_score is not None:
                persona_stats[persona]["scores"].append(overall_score)

        # Calculate averages per persona
        for persona, stats in persona_stats.items():
            # Always calculate pass_rate
            stats["pass_rate"] = stats["passed"] / stats["total"] if stats["total"] > 0 else 0

            # Calculate average score if available
            if stats["scores"]:
                stats["average_score"] = sum(stats["scores"]) / len(stats["scores"])
            del stats["scores"]  # Remove raw scores from summary

        return {
            "total_conversations": total,
            "passed": passed,
            "failed": total - passed,
            "pass_rate": pass_rate,
            "metric_averages": metric_averages,
            "persona_statistics": persona_stats,
            "threshold": self.threshold,
            "judge_model": self.judge_model,
            "evaluated_at": datetime.now().isoformat(),
        }
