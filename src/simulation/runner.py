"""Conversation simulation runner using DeepEval."""

import asyncio
from pathlib import Path
from typing import List, Optional, Callable
from datetime import datetime
import os

from deepeval.simulator import ConversationSimulator
from deepeval.dataset import ConversationalGolden
from deepeval.test_case import ConversationalTestCase, Turn
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

from src.core.types import AgentCallback, PersonaConfig

console = Console()


class SimulationRunner:
    """Runs conversation simulations using DeepEval's ConversationSimulator.

    This class wraps DeepEval's simulator and provides additional features:
    - Progress tracking and logging
    - Batch simulation management
    - Integration with persona configurations
    - Graceful error handling

    Example:
        ```python
        async def my_agent(input, turns, thread_id):
            return Turn(role="assistant", content=f"Response to: {input}")

        runner = SimulationRunner(agent_callback=my_agent)

        goldens, configs = load_personas_from_yaml("personas.yaml")
        test_cases = await runner.run_simulations(goldens, configs)
        ```
    """

    def __init__(
        self,
        agent_callback: AgentCallback,
        max_concurrent: Optional[int] = None,
        model: str = "gpt-4o",
    ):
        """Initialize the simulation runner.

        Args:
            agent_callback: The callback function that calls your agent
            max_concurrent: Maximum concurrent simulations (default: from env or 10)
            model: LLM model to use for simulating users (default: gpt-4o)
        """
        self.agent_callback = agent_callback
        self.max_concurrent = max_concurrent or int(os.getenv("MAX_CONCURRENT_SIMULATIONS", "10"))
        self.model = model

        # Initialize DeepEval simulator
        self.simulator = ConversationSimulator(
            model_callback=self.agent_callback, max_concurrent=self.max_concurrent
        )

    async def _check_agent_health(self) -> tuple[bool, str]:
        """Test the agent with a simple message before running simulations.

        Returns:
            Tuple of (is_healthy, message)
        """
        try:
            test_response = await self.agent_callback(
                "Hello, this is a health check.",
                [],  # empty turns
                "health-check-test"
            )

            # Check for common error patterns that indicate connectivity issues
            response_text = test_response.content.lower()
            error_patterns = [
                "cannot connect",
                "connection refused",
                "service error",
                "timeout",
                "not running",
                "connect error",
            ]

            for pattern in error_patterns:
                if pattern in response_text:
                    return False, test_response.content

            return True, "Agent responded successfully"

        except Exception as e:
            return False, str(e)

    async def run_simulations(
        self,
        goldens: List[ConversationalGolden],
        persona_configs: List[PersonaConfig],
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> List[ConversationalTestCase]:
        """Run simulations for all personas.

        Args:
            goldens: List of ConversationalGolden objects (from YAML loader)
            persona_configs: List of PersonaConfig objects for metadata
            progress_callback: Optional callback for progress updates

        Returns:
            List of ConversationalTestCase objects with full conversation history

        Raises:
            RuntimeError: If agent health check fails
            Exception: If simulation fails
        """
        # Pre-flight health check
        console.print("[cyan]Checking agent connectivity...[/cyan]")
        is_healthy, health_message = await self._check_agent_health()

        if not is_healthy:
            console.print(f"\n[bold red]✗ Agent health check failed![/bold red]")
            console.print(f"[red]Error:[/red] {health_message}")
            console.print("\n[yellow]Hint:[/yellow] Make sure your agent service is running before simulations.")
            raise RuntimeError(f"Agent health check failed: {health_message}")

        console.print("[green]✓[/green] Agent is responding\n")

        console.print(f"[bold blue]Starting simulations...[/bold blue]")
        console.print(f"Total conversations to simulate: {len(goldens)}")
        console.print(f"Total personas: {len(persona_configs)}")
        console.print(f"Max concurrent: {self.max_concurrent}\n")

        # Create a mapping of persona names to their configs for max_turns lookup
        persona_map = {pc.name: pc for pc in persona_configs}

        # Run the simulations with progress tracking
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("[cyan]Simulating conversations...", total=len(goldens))

            # DeepEval's simulate method handles the concurrent execution
            test_cases = self.simulator.simulate(
                conversational_goldens=goldens,
                max_user_simulations=max(
                    pc.max_turns for pc in persona_configs
                ),  # Use max across all personas
            )

            # Update progress (DeepEval runs all at once)
            progress.update(task, completed=len(goldens))

            if progress_callback:
                progress_callback(len(goldens), len(goldens))

        console.print(
            f"\n[bold green]✓[/bold green] Successfully simulated {len(test_cases)} conversations\n"
        )

        # Add persona metadata to test cases
        for i, test_case in enumerate(test_cases):
            if i < len(goldens) and hasattr(goldens[i], "additional_metadata"):
                metadata = goldens[i].additional_metadata
                persona_name = metadata.get("persona_name", "unknown")

                # Enhance test case with persona info
                test_case.additional_metadata = {
                    **metadata,
                    "simulated_at": datetime.now().isoformat(),
                    "model": self.model,
                    "persona_config": (
                        persona_map.get(persona_name).__dict__
                        if persona_name in persona_map
                        else {}
                    ),
                }

        return test_cases

    async def run_single_conversation(
        self,
        scenario: str,
        expected_outcome: str,
        user_description: str,
        max_turns: int = 15,
        context: Optional[str] = None,
    ) -> ConversationalTestCase:
        """Run a single conversation simulation.

        Useful for testing or quick evaluations.

        Args:
            scenario: The scenario/task description
            expected_outcome: What success looks like
            user_description: Description of the simulated user
            max_turns: Maximum number of conversation turns
            context: Optional additional context

        Returns:
            ConversationalTestCase: The simulated conversation
        """
        golden = ConversationalGolden(
            scenario=scenario,
            expected_outcome=expected_outcome,
            user_description=user_description,
            context=[context] if context else None,
        )

        console.print(f"[cyan]Simulating single conversation:[/cyan]")
        console.print(f"  Scenario: {scenario}")
        console.print(f"  Max turns: {max_turns}\n")

        test_cases = self.simulator.simulate(
            conversational_goldens=[golden], max_user_simulations=max_turns
        )

        if test_cases:
            test_case = test_cases[0]
            console.print(
                f"[green]✓[/green] Conversation completed with {len(test_case.turns)} turns\n"
            )
            return test_case
        else:
            raise Exception("Simulation failed to produce a test case")

    def get_simulation_summary(self, test_cases: List[ConversationalTestCase]) -> dict:
        """Get summary statistics about the simulations.

        Args:
            test_cases: List of completed test cases

        Returns:
            Dict with summary stats
        """
        total_turns = sum(len(tc.turns) for tc in test_cases)
        avg_turns = total_turns / len(test_cases) if test_cases else 0

        # Group by persona
        persona_stats = {}
        for tc in test_cases:
            if hasattr(tc, "additional_metadata"):
                persona = tc.additional_metadata.get("persona_name", "unknown")
                if persona not in persona_stats:
                    persona_stats[persona] = {"count": 0, "total_turns": 0, "conversations": []}
                persona_stats[persona]["count"] += 1
                persona_stats[persona]["total_turns"] += len(tc.turns)
                persona_stats[persona]["conversations"].append(tc)

        # Calculate averages per persona
        for persona, stats in persona_stats.items():
            stats["avg_turns"] = stats["total_turns"] / stats["count"] if stats["count"] > 0 else 0

        return {
            "total_conversations": len(test_cases),
            "total_turns": total_turns,
            "avg_turns_per_conversation": avg_turns,
            "personas": persona_stats,
        }
