"""Red team configuration and deepteam object builders."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List

import yaml


@dataclass
class RedTeamConfig:
    """Configuration for a red team scan.

    Attributes:
        purpose: What the agent does and what it should refuse. The more
                 specific, the better the adversarial probes generated.
        vulnerabilities: List of vulnerability class names to test.
                         Use 'default' for the standard starter set.
        attacks: List of attack strategy names to apply.
                 Use 'default' for PromptInjection + Roleplay.
        attacks_per_type: Number of adversarial probes generated per
                          vulnerability type (maps to deepteam's
                          attacks_per_vulnerability_type).
        max_concurrency: Max parallel adversarial requests to the agent.
        output_dir: Directory to save scan results and reports.
        simulator_model: LLM used to generate adversarial probes.
        evaluation_model: LLM used to judge whether the agent was vulnerable.
    """

    purpose: str
    vulnerabilities: List[str] = field(default_factory=lambda: ["default"])
    attacks: List[str] = field(default_factory=lambda: ["default"])
    attacks_per_type: int = 5
    max_concurrency: int = 4
    output_dir: str = "reports/redteam"
    simulator_model: str = "gpt-4o-mini"
    evaluation_model: str = "gpt-4o"


def load_redteam_config(config_path: str | Path) -> RedTeamConfig:
    """Load red team configuration from a YAML file.

    Args:
        config_path: Path to the YAML config file

    Returns:
        RedTeamConfig populated from the file

    Raises:
        FileNotFoundError: If the config file does not exist
        ValueError: If old promptfoo-style keys are detected
        KeyError: If the required 'purpose' field is missing
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Red team config not found: {config_path}")

    with open(config_path, "r") as f:
        data = yaml.safe_load(f)

    # Detect and reject stale promptfoo-style keys
    _check_for_old_keys(data, config_path)

    return RedTeamConfig(
        purpose=data["purpose"],
        vulnerabilities=data.get("vulnerabilities", ["default"]),
        attacks=data.get("attacks", ["default"]),
        attacks_per_type=data.get("attacks_per_type", 5),
        max_concurrency=data.get("max_concurrency", 4),
        output_dir=data.get("output_dir", "reports/redteam"),
        simulator_model=data.get("simulator_model", "gpt-4o"),
        evaluation_model=data.get("evaluation_model", "gpt-4o"),
    )


def build_vulnerabilities(names: List[str]) -> List[Any]:
    """Instantiate deepteam vulnerability objects from a list of string names.

    Supports the 'default' sentinel, which expands to a curated starter set
    covering the most commonly relevant categories.

    Args:
        names: List of vulnerability names (e.g. ["Bias", "PIILeakage"])

    Returns:
        List of instantiated deepteam vulnerability objects

    Raises:
        ValueError: If an unknown vulnerability name is encountered
    """
    from deepteam.vulnerabilities import (
        Bias,
        BFLA,
        BOLA,
        ChildProtection,
        Competition,
        DebugAccess,
        Ethics,
        ExcessiveAgency,
        Fairness,
        GoalTheft,
        GraphicContent,
        IllegalActivity,
        IndirectInstruction,
        IntellectualProperty,
        Misinformation,
        PersonalSafety,
        PIILeakage,
        PromptLeakage,
        RBAC,
        ShellInjection,
        SQLInjection,
        SSRF,
        Toxicity,
    )

    _VULNERABILITY_MAP = {
        "Bias": Bias,
        "Toxicity": Toxicity,
        "PIILeakage": PIILeakage,
        "PromptLeakage": PromptLeakage,
        "SQLInjection": SQLInjection,
        "ShellInjection": ShellInjection,
        "Misinformation": Misinformation,
        "IllegalActivity": IllegalActivity,
        "GraphicContent": GraphicContent,
        "PersonalSafety": PersonalSafety,
        "ChildProtection": ChildProtection,
        "BOLA": BOLA,
        "BFLA": BFLA,
        "RBAC": RBAC,
        "DebugAccess": DebugAccess,
        "SSRF": SSRF,
        "Competition": Competition,
        "IntellectualProperty": IntellectualProperty,
        "ExcessiveAgency": ExcessiveAgency,
        "GoalTheft": GoalTheft,
        "IndirectInstruction": IndirectInstruction,
        "Ethics": Ethics,
        "Fairness": Fairness,
    }

    _DEFAULT_VULNERABILITIES = [
        Bias(),
        Toxicity(),
        PIILeakage(),
        PromptLeakage(),
        IllegalActivity(),
        Misinformation(),
        ExcessiveAgency(),
    ]

    instances = []
    for name in names:
        if name == "default":
            instances.extend(_DEFAULT_VULNERABILITIES)
        elif name in _VULNERABILITY_MAP:
            instances.append(_VULNERABILITY_MAP[name]())
        else:
            valid = sorted(_VULNERABILITY_MAP.keys()) + ["default"]
            raise ValueError(
                f"Unknown vulnerability: '{name}'. Valid options:\n  {', '.join(valid)}"
            )
    return instances


def build_attacks(names: List[str]) -> List[Any]:
    """Instantiate deepteam attack objects from a list of string names.

    Supports the 'default' sentinel, which expands to PromptInjection + Roleplay.

    Args:
        names: List of attack names (e.g. ["PromptInjection", "CrescendoJailbreaking"])

    Returns:
        List of instantiated deepteam attack objects

    Raises:
        ValueError: If an unknown attack name is encountered
    """
    from deepteam.attacks.single_turn import (
        AdversarialPoetry,
        AuthorityEscalation,
        Base64,
        EmotionalManipulation,
        GrayBox,
        Leetspeak,
        MathProblem,
        Multilingual,
        PromptInjection,
        PromptProbing,
        Roleplay,
        ROT13,
    )
    from deepteam.attacks.multi_turn import (
        BadLikertJudge,
        CrescendoJailbreaking,
        LinearJailbreaking,
        SequentialJailbreak,
        TreeJailbreaking,
    )

    _ATTACK_MAP = {
        # Single-turn
        "PromptInjection": PromptInjection,
        "Roleplay": Roleplay,
        "ROT13": ROT13,
        "Base64": Base64,
        "Leetspeak": Leetspeak,
        "Multilingual": Multilingual,
        "GrayBox": GrayBox,
        "MathProblem": MathProblem,
        "PromptProbing": PromptProbing,
        "AdversarialPoetry": AdversarialPoetry,
        "AuthorityEscalation": AuthorityEscalation,
        "EmotionalManipulation": EmotionalManipulation,
        # Multi-turn
        "CrescendoJailbreaking": CrescendoJailbreaking,
        "LinearJailbreaking": LinearJailbreaking,
        "TreeJailbreaking": TreeJailbreaking,
        "SequentialJailbreak": SequentialJailbreak,
        "BadLikertJudge": BadLikertJudge,
    }

    _DEFAULT_ATTACKS = [PromptInjection(), Roleplay()]

    instances = []
    for name in names:
        if name == "default":
            instances.extend(_DEFAULT_ATTACKS)
        elif name in _ATTACK_MAP:
            instances.append(_ATTACK_MAP[name]())
        else:
            valid = sorted(_ATTACK_MAP.keys()) + ["default"]
            raise ValueError(
                f"Unknown attack: '{name}'. Valid options:\n  {', '.join(valid)}"
            )
    return instances


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _check_for_old_keys(data: dict, config_path: Path) -> None:
    """Raise a clear error if stale promptfoo-style keys are present."""
    old_key_hints = {
        "plugins": "vulnerabilities",
        "strategies": "attacks",
        "num_tests": "attacks_per_type",
    }
    for old_key, new_key in old_key_hints.items():
        if old_key in data:
            raise ValueError(
                f"'{old_key}' is a promptfoo config key and is no longer supported.\n"
                f"Rename it to '{new_key}' in {config_path}.\n"
                f"See config/redteam_config.yaml for the current format."
            )
