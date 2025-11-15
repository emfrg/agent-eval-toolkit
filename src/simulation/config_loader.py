"""Configuration loader for personas and scenarios."""

import yaml
from pathlib import Path
from typing import List, Dict, Any
from deepeval.dataset import ConversationalGolden
from src.core.types import PersonaConfig


def load_personas_from_yaml(yaml_path: str | Path) -> tuple[List[ConversationalGolden], List[PersonaConfig]]:
    """Load persona configurations from YAML file.

    DeepEval doesn't natively support YAML configuration, so this loader
    bridges the gap by converting YAML personas into ConversationalGolden objects.

    Args:
        yaml_path: Path to the YAML configuration file

    Returns:
        Tuple of (goldens, persona_configs):
            - goldens: List of ConversationalGolden objects for DeepEval
            - persona_configs: List of PersonaConfig objects with metadata

    Raises:
        FileNotFoundError: If YAML file doesn't exist
        ValueError: If YAML format is invalid

    Example YAML format:
        ```yaml
        personas:
          busy_banker:
            scenario: "User wants quick loan information"
            expected_outcome: "Clear answer in 5 messages"
            user_description: "Time-poor professional who expects concise answers"
            conversations: 10
            max_turns: 12
            context: "User is a returning customer"  # optional
            chatbot_role: "Helpful loan assistant"  # optional
        ```
    """
    yaml_path = Path(yaml_path)

    if not yaml_path.exists():
        raise FileNotFoundError(f"Persona configuration file not found: {yaml_path}")

    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)

    if not data or 'personas' not in data:
        raise ValueError("YAML file must contain a 'personas' key")

    goldens = []
    persona_configs = []

    for persona_name, persona_data in data['personas'].items():
        # Validate required fields
        required_fields = ['scenario', 'expected_outcome', 'user_description']
        missing_fields = [f for f in required_fields if f not in persona_data]
        if missing_fields:
            raise ValueError(
                f"Persona '{persona_name}' missing required fields: {missing_fields}"
            )

        # Create PersonaConfig for metadata tracking
        persona_config = PersonaConfig(
            name=persona_name,
            scenario=persona_data['scenario'],
            expected_outcome=persona_data['expected_outcome'],
            user_description=persona_data['user_description'],
            conversations=persona_data.get('conversations', 5),
            max_turns=persona_data.get('max_turns', 15),
            context=persona_data.get('context'),
            chatbot_role=persona_data.get('chatbot_role')
        )
        persona_configs.append(persona_config)

        # Create ConversationalGolden for each conversation this persona should have
        for i in range(persona_config.conversations):
            golden = ConversationalGolden(
                scenario=persona_config.scenario,
                expected_outcome=persona_config.expected_outcome,
                user_description=persona_config.user_description,
                context=[persona_config.context] if persona_config.context else None
            )
            # Add metadata to track which persona this belongs to
            golden.additional_metadata = {
                "persona_name": persona_name,
                "conversation_index": i,
                "max_turns": persona_config.max_turns
            }
            goldens.append(golden)

    return goldens, persona_configs


def load_judge_prompt(prompt_path: str | Path) -> str:
    """Load custom judge prompt from markdown file.

    Args:
        prompt_path: Path to the judge prompt markdown file

    Returns:
        str: The prompt text

    Raises:
        FileNotFoundError: If prompt file doesn't exist
    """
    prompt_path = Path(prompt_path)

    if not prompt_path.exists():
        raise FileNotFoundError(f"Judge prompt file not found: {prompt_path}")

    with open(prompt_path, 'r') as f:
        return f.read().strip()


def validate_persona_config(config_path: str | Path) -> Dict[str, Any]:
    """Validate persona configuration and return summary stats.

    Args:
        config_path: Path to the YAML configuration file

    Returns:
        Dict with validation results and stats

    Example:
        {
            "valid": True,
            "total_personas": 3,
            "total_conversations": 25,
            "personas": {
                "busy_banker": {"conversations": 10, "max_turns": 12},
                ...
            }
        }
    """
    try:
        goldens, persona_configs = load_personas_from_yaml(config_path)

        stats = {
            "valid": True,
            "total_personas": len(persona_configs),
            "total_conversations": len(goldens),
            "personas": {}
        }

        for pc in persona_configs:
            stats["personas"][pc.name] = {
                "conversations": pc.conversations,
                "max_turns": pc.max_turns,
                "scenario": pc.scenario
            }

        return stats

    except Exception as e:
        return {
            "valid": False,
            "error": str(e)
        }
