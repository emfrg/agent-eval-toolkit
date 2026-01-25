"""Standalone test script for evaluation step.

This script allows testing the evaluation pipeline independently
without re-running the full simulation. Useful for debugging and iteration.

Usage:
    python tests/test_eval.py
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
from deepeval.test_case import ConversationalTestCase, Turn
from rich.console import Console

from src.core.logging import ConversationLogger
from src.evaluation.judge import JudgeEvaluator
from src.evaluation.report import ReportGenerator

load_dotenv()
console = Console()


def main():
    """Run standalone evaluation test."""
    console.print("[bold cyan]Standalone Evaluation Test[/bold cyan]\n")

    # Load conversations from logs
    logs_dir = project_root / "quickstart_output" / "logs"
    console.print(f"[cyan]Loading conversations from:[/cyan] {logs_dir}")

    if not logs_dir.exists():
        console.print(f"[red]Error: Logs directory not found: {logs_dir}[/red]")
        console.print("Run 'agent-eval-toolkit quickstart' first to generate logs")
        return 1

    logger = ConversationLogger(output_dir=str(logs_dir))

    # Find most recent JSONL file
    jsonl_files = list(logs_dir.glob("*.jsonl"))
    if not jsonl_files:
        console.print(f"[red]No JSONL files found in {logs_dir}[/red]")
        console.print("Run 'agent-eval-toolkit quickstart' first to generate logs")
        return 1

    latest_file = max(jsonl_files, key=lambda p: p.stat().st_mtime)
    console.print(f"[green]✓[/green] Using: {latest_file.name}")

    conversations = logger.load_from_jsonl(latest_file)
    console.print(f"[green]✓[/green] Loaded {len(conversations)} conversations")

    # Take only first 3 for quick testing
    test_conversations = conversations[:3]
    console.print(
        f"[yellow]Testing with {len(test_conversations)} conversations for speed[/yellow]\n"
    )

    # Convert to test cases
    test_cases = []
    for conv in test_conversations:
        turns = [Turn(role=turn["role"], content=turn["content"]) for turn in conv["turns"]]

        test_case = ConversationalTestCase(
            scenario=conv["scenario"],
            expected_outcome=conv["expected_outcome"],
            turns=turns,
        )

        if "metadata" in conv:
            test_case.additional_metadata = conv["metadata"]

        test_cases.append(test_case)

    # Initialize evaluator
    judge_prompt_path = project_root / "config" / "judge_prompt.md"
    evaluator = JudgeEvaluator(
        custom_prompt_path=judge_prompt_path if judge_prompt_path.exists() else None,
        threshold=0.7,
    )

    # Run evaluation
    console.print("[bold blue]Running evaluation...[/bold blue]\n")
    try:
        results = evaluator.evaluate_conversations(test_cases, verbose=True)
    except RuntimeError as e:
        if "Confident AI" in str(e):
            console.print("\n[red]Evaluation failed due to Confident AI configuration issue.[/red]")
            console.print("See error message above for solution.")
            return 1
        raise

    # Get summary
    summary = evaluator.get_summary(results)

    console.print(f"\n[bold green]✨ Evaluation complete![/bold green]")
    console.print(f"Pass rate: {summary['pass_rate']*100:.1f}%")
    console.print(f"Passed: {summary['passed']} | Failed: {summary['failed']}")

    # Optionally generate reports
    reports_dir = project_root / "quickstart_output" / "test_reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    report_gen = ReportGenerator(output_dir=str(reports_dir))
    json_path, md_path = report_gen.generate_reports(evaluation_results=results, summary=summary)

    console.print(f"\n[green]✓[/green] Test reports saved to: {reports_dir}/")

    return 0


if __name__ == "__main__":
    sys.exit(main())
