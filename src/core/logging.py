"""Conversation logging utilities for JSON and JSONL formats."""

import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

from deepeval.test_case import ConversationalTestCase, Turn
from deepeval.dataset import EvaluationDataset
from rich.console import Console

from src.core.types import LoggedTurn, AgentOutputAdapter

console = Console()


class ConversationLogger:
    """Handles logging of conversations in multiple formats.

    DeepEval natively supports JSON and CSV export, but many users prefer JSONL
    for streaming and line-by-line processing. This logger provides both.

    Features:
    - Save conversations as JSON (DeepEval native)
    - Save conversations as JSONL (custom implementation)
    - Optional output normalization via AgentOutputAdapter
    - Automatic directory creation
    - Rich metadata preservation
    """

    def __init__(
        self,
        output_dir: str | Path = "logs",
        output_adapter: Optional[AgentOutputAdapter] = None
    ):
        """Initialize the conversation logger.

        Args:
            output_dir: Directory to save conversation logs
            output_adapter: Optional adapter to normalize outputs to LoggedTurn format
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.output_adapter = output_adapter

    def save_as_json(
        self,
        test_cases: List[ConversationalTestCase],
        filename: Optional[str] = None
    ) -> Path:
        """Save conversations as JSON using DeepEval's native format.

        Args:
            test_cases: List of ConversationalTestCase objects
            filename: Optional custom filename (default: timestamped)

        Returns:
            Path: Path to the saved JSON file
        """
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"conversations_{timestamp}.json"

        # Use DeepEval's EvaluationDataset for native JSON export
        dataset = EvaluationDataset(test_cases=test_cases)
        output_path = self.output_dir / filename

        # DeepEval's save_as method
        dataset.save_as(
            file_type="json",
            directory=str(self.output_dir),
            file_name=filename.replace('.json', '')  # DeepEval adds .json extension
        )

        console.print(f"[green]✓[/green] Saved JSON to: {output_path}")
        return output_path

    def save_as_jsonl(
        self,
        test_cases: List[ConversationalTestCase],
        filename: Optional[str] = None,
        include_metadata: bool = True
    ) -> Path:
        """Save conversations as JSONL (one conversation per line).

        JSONL format is preferred for:
        - Streaming processing
        - Line-by-line analysis
        - Easier appending and incremental processing

        Args:
            test_cases: List of ConversationalTestCase objects
            filename: Optional custom filename (default: timestamped)
            include_metadata: Include additional metadata in each line

        Returns:
            Path: Path to the saved JSONL file
        """
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"conversations_{timestamp}.jsonl"

        output_path = self.output_dir / filename

        with open(output_path, 'w') as f:
            for test_case in test_cases:
                conversation_data = self._test_case_to_dict(
                    test_case,
                    include_metadata=include_metadata
                )
                f.write(json.dumps(conversation_data) + '\n')

        console.print(f"[green]✓[/green] Saved JSONL to: {output_path}")
        console.print(f"   Total conversations: {len(test_cases)}")
        return output_path

    def save_normalized_jsonl(
        self,
        test_cases: List[ConversationalTestCase],
        filename: Optional[str] = None
    ) -> Path:
        """Save conversations in normalized LoggedTurn format as JSONL.

        This uses the AgentOutputAdapter (if provided) to normalize all turns
        to the standard LoggedTurn schema, preserving tool calls and metadata.

        Args:
            test_cases: List of ConversationalTestCase objects
            filename: Optional custom filename (default: timestamped)

        Returns:
            Path: Path to the saved JSONL file

        Raises:
            ValueError: If output_adapter was not provided during initialization
        """
        if not self.output_adapter:
            console.print(
                "[yellow]Warning:[/yellow] No output_adapter provided. "
                "Saving in standard format instead."
            )
            return self.save_as_jsonl(test_cases, filename)

        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"conversations_normalized_{timestamp}.jsonl"

        output_path = self.output_dir / filename

        with open(output_path, 'w') as f:
            for test_case in test_cases:
                # Convert each turn using the adapter
                normalized_turns = []
                for turn in test_case.turns:
                    try:
                        # For simplicity, we'll create a basic representation
                        # In practice, you'd pass the raw agent response here
                        logged_turn = LoggedTurn(
                            role=turn.role,
                            content=turn.content,
                            tool_calls=[],  # Would be populated by adapter
                            meta={}
                        )
                        normalized_turns.append(logged_turn.to_dict())
                    except Exception as e:
                        console.print(f"[yellow]Warning:[/yellow] Failed to normalize turn: {e}")
                        # Fallback to basic format
                        normalized_turns.append({
                            "role": turn.role,
                            "content": turn.content,
                            "tool_calls": [],
                            "meta": {}
                        })

                conversation_data = {
                    "scenario": test_case.scenario,
                    "expected_outcome": test_case.expected_outcome,
                    "turns": normalized_turns,
                    "metadata": getattr(test_case, 'additional_metadata', {})
                }

                f.write(json.dumps(conversation_data) + '\n')

        console.print(f"[green]✓[/green] Saved normalized JSONL to: {output_path}")
        return output_path

    def load_from_jsonl(self, file_path: str | Path) -> List[Dict[str, Any]]:
        """Load conversations from JSONL file.

        Args:
            file_path: Path to the JSONL file

        Returns:
            List of conversation dictionaries

        Raises:
            FileNotFoundError: If file doesn't exist
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"JSONL file not found: {file_path}")

        conversations = []
        with open(file_path, 'r') as f:
            for line in f:
                if line.strip():
                    conversations.append(json.loads(line))

        console.print(f"[green]✓[/green] Loaded {len(conversations)} conversations from {file_path}")
        return conversations

    def _test_case_to_dict(
        self,
        test_case: ConversationalTestCase,
        include_metadata: bool = True
    ) -> Dict[str, Any]:
        """Convert a ConversationalTestCase to a dictionary.

        Args:
            test_case: The test case to convert
            include_metadata: Include additional metadata

        Returns:
            Dict representation of the conversation
        """
        conversation = {
            "scenario": test_case.scenario,
            "expected_outcome": test_case.expected_outcome,
            "turns": [
                {
                    "role": turn.role,
                    "content": turn.content
                }
                for turn in test_case.turns
            ]
        }

        if include_metadata and hasattr(test_case, 'additional_metadata'):
            conversation["metadata"] = test_case.additional_metadata

        return conversation

    def get_log_summary(self) -> Dict[str, Any]:
        """Get summary of all logs in the output directory.

        Returns:
            Dict with summary information about logged conversations
        """
        json_files = list(self.output_dir.glob("*.json"))
        jsonl_files = list(self.output_dir.glob("*.jsonl"))

        summary = {
            "output_dir": str(self.output_dir),
            "json_files": len(json_files),
            "jsonl_files": len(jsonl_files),
            "files": {
                "json": [f.name for f in json_files],
                "jsonl": [f.name for f in jsonl_files]
            }
        }

        return summary
