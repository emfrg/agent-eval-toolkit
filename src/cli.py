"""Command-line interface for AI Agent Conversation Simulator."""

import asyncio
import gc
import time
import importlib.util
import inspect
from pathlib import Path
from typing import Optional, Callable
import sys

import typer
from rich.console import Console
from rich.panel import Panel
from dotenv import load_dotenv

from src.simulation.config_loader import load_personas_from_yaml, validate_persona_config
from src.simulation.runner import SimulationRunner
from src.core.logging import ConversationLogger
from src.evaluation.judge import JudgeEvaluator
from src.evaluation.report import ReportGenerator

# Load environment variables
load_dotenv()

app = typer.Typer(
    name="agent-eval-toolkit",
    help="AI Agent Conversation Simulator & Evaluator - Built on DeepEval",
    add_completion=False,
)
console = Console()


def load_agent_callback(agent_path: str) -> Callable:
    """Dynamically load agent callback from a Python file.

    Args:
        agent_path: Path to Python file, optionally with function name
                   (e.g., "my_agent.py" or "my_agent.py:custom_callback")

    Returns:
        The agent callback function

    Raises:
        typer.Exit: If the agent file or function cannot be loaded
    """
    if ":" in agent_path:
        file_path, function_name = agent_path.split(":", 1)
    else:
        file_path = agent_path
        function_name = "agent_callback"

    file_path = Path(file_path)

    if not file_path.exists():
        console.print(f"[red]Error:[/red] Agent file not found: {file_path}")
        raise typer.Exit(1)

    try:
        spec = importlib.util.spec_from_file_location("custom_agent", file_path)
        if spec is None or spec.loader is None:
            console.print(f"[red]Error:[/red] Could not load module from {file_path}")
            raise typer.Exit(1)

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        if not hasattr(module, function_name):
            console.print(f"[red]Error:[/red] Function '{function_name}' not found in {file_path}")
            console.print(
                f"[yellow]Hint:[/yellow] Make sure your file exports a function named '{function_name}'"
            )
            raise typer.Exit(1)

        callback = getattr(module, function_name)

        if not callable(callback):
            console.print(f"[red]Error:[/red] '{function_name}' is not a function")
            raise typer.Exit(1)

        if not inspect.iscoroutinefunction(callback):
            console.print(f"[red]Error:[/red] '{function_name}' must be an async function")
            console.print(
                f"[yellow]Hint:[/yellow] Use 'async def {function_name}(...)' in your agent file"
            )
            raise typer.Exit(1)

        console.print(f"[green]✓[/green] Loaded agent callback: {function_name} from {file_path}")
        return callback

    except Exception as e:
        console.print(f"[red]Error loading agent:[/red] {e}")
        import traceback

        console.print(traceback.format_exc())
        raise typer.Exit(1)


@app.command()
def simulate(
    agent: str = typer.Option(
        ...,
        "--agent",
        "-a",
        help="Path to your agent file (e.g., my_agent.py or examples/example_agent.py)",
    ),
    config: str = typer.Option(
        "config/personas.yaml", "--config", "-c", help="Path to personas YAML configuration file"
    ),
    output_dir: str = typer.Option(
        "logs", "--output", "-o", help="Directory to save conversation logs"
    ),
    format: str = typer.Option(
        "jsonl", "--format", "-f", help="Output format: json, jsonl, or both"
    ),
    max_concurrent: Optional[int] = typer.Option(
        None, "--max-concurrent", "-m", help="Maximum concurrent simulations"
    ),
):
    """Run conversation simulations with your agent.

    Example:
        agent-eval-toolkit simulate --agent examples/example_agent.py
        agent-eval-toolkit simulate --agent my_agent.py
    """
    console.print(
        Panel.fit(
            "[bold blue]AI Agent Conversation Simulator[/bold blue]\n" "Running simulations...",
            border_style="blue",
        )
    )

    try:
        # Validate configuration
        console.print(f"\n[cyan]Validating configuration:[/cyan] {config}")
        validation = validate_persona_config(config)

        if not validation["valid"]:
            console.print(f"[red]✗ Configuration error:[/red] {validation['error']}")
            raise typer.Exit(1)

        console.print(f"[green]✓[/green] Configuration valid!")
        console.print(f"  Personas: {validation['total_personas']}")
        console.print(f"  Total conversations: {validation['total_conversations']}")

        # Load personas
        console.print(f"\n[cyan]Loading personas...[/cyan]")
        goldens, persona_configs = load_personas_from_yaml(config)
        console.print(f"[green]✓[/green] Loaded {len(persona_configs)} personas")

        # Load agent
        console.print(f"\n[cyan]Loading agent...[/cyan]")
        agent_callback = load_agent_callback(agent)

        # Run simulations
        runner = SimulationRunner(agent_callback=agent_callback, max_concurrent=max_concurrent)

        async def run():
            result = await runner.run_simulations(goldens, persona_configs)
            await asyncio.sleep(0.1)  # Allow httpx cleanup
            return result

        test_cases = asyncio.run(run())

        # Display summary
        summary = runner.get_simulation_summary(test_cases)
        console.print(f"\n[bold green]Simulation Summary:[/bold green]")
        console.print(f"  Total conversations: {summary['total_conversations']}")
        console.print(f"  Total turns: {summary['total_turns']}")
        console.print(f"  Average turns/conversation: {summary['avg_turns_per_conversation']:.1f}")

        # Save logs
        logger = ConversationLogger(output_dir=output_dir)

        if format in ["json", "both"]:
            json_path = logger.save_as_json(test_cases)
            console.print(f"\n[green]✓[/green] Saved JSON: {json_path}")

        if format in ["jsonl", "both"]:
            jsonl_path = logger.save_as_jsonl(test_cases)
            console.print(f"[green]✓[/green] Saved JSONL: {jsonl_path}")

        # Clean up event loop resources
        gc.collect()

        console.print(f"\n[bold green]✨ Simulation complete![/bold green]")

    except FileNotFoundError as e:
        console.print(f"\n[red]✗ Error:[/red] {e}")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"\n[red]✗ Unexpected error:[/red] {e}")
        import traceback

        console.print(traceback.format_exc())
        raise typer.Exit(1)


@app.command()
def evaluate(
    logs_dir: str = typer.Option(
        "logs", "--logs-dir", "-l", help="Directory containing conversation logs (JSONL)"
    ),
    output_dir: str = typer.Option(
        "reports", "--output", "-o", help="Directory to save evaluation reports"
    ),
    judge_prompt: Optional[str] = typer.Option(
        "config/judge_prompt.md",
        "--prompt",
        "-p",
        help="Path to custom judge prompt (markdown file)",
    ),
    threshold: float = typer.Option(
        0.7, "--threshold", "-t", help="Pass/fail threshold (0.0 to 1.0)"
    ),
):
    """Evaluate conversations using LLM-as-a-judge.

    This command:
    1. Loads conversations from logs directory
    2. Evaluates using DeepEval metrics
    3. Generates JSON and Markdown reports

    Example:
        agent-eval-toolkit evaluate --logs-dir logs --threshold 0.7
    """
    console.print(
        Panel.fit(
            "[bold blue]LLM-as-a-Judge Evaluation[/bold blue]\n" "Evaluating conversations...",
            border_style="blue",
        )
    )

    try:
        # Load conversations from logs
        console.print(f"\n[cyan]Loading conversations from:[/cyan] {logs_dir}")
        logger = ConversationLogger(output_dir=logs_dir)

        # Find the most recent JSONL file
        jsonl_files = list(Path(logs_dir).glob("*.jsonl"))
        if not jsonl_files:
            console.print(f"[red]✗ No JSONL files found in {logs_dir}[/red]")
            console.print("Run 'simulate' command first to generate conversation logs")
            raise typer.Exit(1)

        # Use most recent file
        latest_file = max(jsonl_files, key=lambda p: p.stat().st_mtime)
        console.print(f"[green]✓[/green] Using: {latest_file.name}")

        conversations = logger.load_from_jsonl(latest_file)

        # Convert to test cases for evaluation
        from deepeval.test_case import ConversationalTestCase, Turn

        test_cases = []
        for conv in conversations:
            turns = [Turn(role=turn["role"], content=turn["content"]) for turn in conv["turns"]]

            test_case = ConversationalTestCase(
                scenario=conv["scenario"], expected_outcome=conv["expected_outcome"], turns=turns
            )

            # Add metadata if present
            if "metadata" in conv:
                test_case.additional_metadata = conv["metadata"]

            test_cases.append(test_case)

        console.print(f"[green]✓[/green] Loaded {len(test_cases)} conversations for evaluation")

        # Initialize evaluator
        evaluator = JudgeEvaluator(
            custom_prompt_path=judge_prompt if Path(judge_prompt).exists() else None,
            threshold=threshold,
        )

        # Run evaluation
        results = evaluator.evaluate_conversations(test_cases, verbose=True)

        # Get summary
        summary = evaluator.get_summary(results)

        # Generate reports
        report_gen = ReportGenerator(output_dir=output_dir)
        json_path, md_path = report_gen.generate_reports(
            evaluation_results=results, summary=summary
        )

        console.print(f"\n[bold green]✨ Evaluation complete![/bold green]")
        console.print(f"\nPass rate: {summary['pass_rate']*100:.1f}%")
        console.print(f"Passed: {summary['passed']} | Failed: {summary['failed']}")

    except FileNotFoundError as e:
        console.print(f"\n[red]✗ Error:[/red] {e}")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"\n[red]✗ Unexpected error:[/red] {e}")
        import traceback

        console.print(traceback.format_exc())
        raise typer.Exit(1)


@app.command()
def quickstart(
    output_dir: str = typer.Option(
        "quickstart_output", "--output", "-o", help="Directory for all output (logs and reports)"
    ),
):
    """Run a complete end-to-end demo.

    This command:
    1. Runs simulations with example agent
    2. Evaluates conversations
    3. Generates reports

    Perfect for testing the system without setting up a real agent!

    Example:
        agent-eval-toolkit quickstart
    """
    console.print(
        Panel.fit(
            "[bold magenta]Quickstart Demo[/bold magenta]\n"
            "Running complete simulation + evaluation pipeline",
            border_style="magenta",
        )
    )

    try:
        # Create output directories
        logs_dir = Path(output_dir) / "logs"
        reports_dir = Path(output_dir) / "reports"
        logs_dir.mkdir(parents=True, exist_ok=True)
        reports_dir.mkdir(parents=True, exist_ok=True)

        # Check if personas.yaml exists, otherwise guide user
        personas_path = Path("config/personas.yaml")
        if not personas_path.exists():
            console.print(f"\n[yellow]No personas configuration found at {personas_path}[/yellow]")
            console.print("Please create config/personas.yaml first or use:")
            console.print("  agent-eval-toolkit validate-config")
            raise typer.Exit(1)

        # Step 1: Simulate
        console.print("\n[bold cyan]Step 1: Running simulations...[/bold cyan]")

        goldens, persona_configs = load_personas_from_yaml(str(personas_path))
        console.print(f"[green]✓[/green] Loaded {len(persona_configs)} personas")

        mock_agent_path = Path("examples/mock_agent.py")
        agent_callback = load_agent_callback(str(mock_agent_path))

        runner = SimulationRunner(agent_callback=agent_callback)

        async def run_sims():
            result = await runner.run_simulations(goldens, persona_configs)
            await asyncio.sleep(0.1)  # Allow httpx cleanup
            return result

        test_cases = asyncio.run(run_sims())
        console.print(f"[green]✓[/green] Generated {len(test_cases)} conversations")

        logger = ConversationLogger(output_dir=str(logs_dir))
        jsonl_path = logger.save_as_jsonl(test_cases)
        console.print(f"[green]✓[/green] Saved logs: {jsonl_path}")

        # Clean up event loop resources from simulation
        gc.collect()
        time.sleep(0.5)

        # Step 2: Evaluate
        console.print("\n[bold cyan]Step 2: Evaluating conversations...[/bold cyan]")

        judge_prompt_path = Path("config/judge_prompt.md")
        evaluator = JudgeEvaluator(
            custom_prompt_path=judge_prompt_path if judge_prompt_path.exists() else None,
            threshold=0.7,
        )

        results = evaluator.evaluate_conversations(test_cases, verbose=True)
        summary = evaluator.get_summary(results)

        report_gen = ReportGenerator(output_dir=str(reports_dir))
        json_path, md_path = report_gen.generate_reports(
            evaluation_results=results, summary=summary
        )

        console.print(f"\n[bold green]✨ Evaluation complete![/bold green]")
        console.print(f"Pass rate: {summary['pass_rate']*100:.1f}%")
        console.print(f"Passed: {summary['passed']} | Failed: {summary['failed']}")

        # Final cleanup
        gc.collect()

        console.print(f"\n[bold green]✨ Quickstart complete![/bold green]")
        console.print(f"\nAll outputs saved to: {output_dir}/")
        console.print(f"  Logs: {logs_dir}/")
        console.print(f"  Reports: {reports_dir}/")

    except typer.Exit:
        raise
    except Exception as e:
        console.print(f"\n[red]✗ Unexpected error:[/red] {e}")
        import traceback

        console.print(traceback.format_exc())
        raise typer.Exit(1)


@app.command()
def validate_config(
    config: str = typer.Argument(
        "config/personas.yaml", help="Path to personas YAML configuration"
    ),
):
    """Validate persona configuration file.

    Example:
        agent-eval-toolkit validate-config config/personas.yaml
    """
    console.print(f"[cyan]Validating configuration:[/cyan] {config}")

    validation = validate_persona_config(config)

    if validation["valid"]:
        console.print(f"\n[bold green]✓ Configuration is valid![/bold green]\n")
        console.print(f"Total personas: {validation['total_personas']}")
        console.print(f"Total conversations: {validation['total_conversations']}\n")

        console.print("[bold]Persona breakdown:[/bold]")
        for persona_name, details in validation["personas"].items():
            console.print(f"\n  [cyan]{persona_name}[/cyan]")
            console.print(f"    Conversations: {details['conversations']}")
            console.print(f"    Max turns: {details['max_turns']}")
            console.print(f"    Scenario: {details['scenario'][:60]}...")
    else:
        console.print(f"\n[bold red]✗ Configuration is invalid![/bold red]")
        console.print(f"Error: {validation['error']}")
        raise typer.Exit(1)


@app.command()
def version():
    """Show version information."""
    from src import __version__

    console.print(f"AI Agent Conversation Simulator v{__version__}")
    console.print("Built on DeepEval")


if __name__ == "__main__":
    app()
