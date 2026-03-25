"""Red team runner — orchestrates deepteam adversarial scans."""

from typing import Any, Dict, List, Optional

from rich.console import Console

from deepteam.test_case import RTTurn

from src.core.types import AgentCallback
from src.redteam.config import RedTeamConfig, build_attacks, build_vulnerabilities

console = Console()


class AgentCallbackAdapter:
    """Adapts the project's AgentCallback protocol to deepteam's model_callback signature.

    The project's AgentCallback:
        async def cb(input: str, turns: List[Turn], thread_id: str) -> Turn

    deepteam's model_callback:
        async def cb(input: str, turns: Optional[List[RTTurn]]) -> RTTurn
    """

    def __init__(self, agent_callback: AgentCallback):
        self._cb = agent_callback

    async def __call__(self, input: str, turns: Optional[List] = None) -> RTTurn:
        thread_id = f"redteam-{abs(hash(input))}"
        try:
            turn = await self._cb(
                input=input,
                turns=turns or [],
                thread_id=thread_id,
            )
            return RTTurn(role="assistant", content=turn.content)
        except Exception as exc:
            return RTTurn(role="assistant", content=f"[agent error: {exc}]")


class RedTeamRunner:
    """Runs adversarial red team scans against an agent using deepteam.

    Example:
        runner = RedTeamRunner(config)
        results = runner.run(agent_callback)
    """

    def __init__(self, config: RedTeamConfig):
        self.config = config

    def run(self, agent_callback: AgentCallback) -> Dict[str, Any]:
        """Run a red team scan. Synchronous entry point for the CLI."""
        from deepteam import red_team

        adapter = AgentCallbackAdapter(agent_callback)
        vulnerabilities = build_vulnerabilities(self.config.vulnerabilities)
        attacks = build_attacks(self.config.attacks)

        console.print("[bold blue]Starting red team scan...[/bold blue]")
        console.print(f"  Vulnerabilities: {', '.join(self.config.vulnerabilities)}")
        console.print(f"  Attacks:         {', '.join(self.config.attacks)}")
        console.print(f"  Probes per type: {self.config.attacks_per_type}")
        console.print(f"  Simulator model: {self.config.simulator_model}")
        console.print(f"  Evaluation model: {self.config.evaluation_model}\n")

        risk_assessment = red_team(
            model_callback=adapter,
            vulnerabilities=vulnerabilities,
            attacks=attacks,
            attacks_per_vulnerability_type=self.config.attacks_per_type,
            target_purpose=self.config.purpose,
            simulator_model=self.config.simulator_model,
            evaluation_model=self.config.evaluation_model,
            async_mode=True,
            max_concurrent=self.config.max_concurrency,
            ignore_errors=True,
        )

        return self._parse_risk_assessment(risk_assessment)

    def _parse_risk_assessment(self, risk_assessment: Any) -> Dict[str, Any]:
        """Parse a deepteam RiskAssessment into the structured dict used by RedTeamReporter.

        deepteam scoring: score == 1.0 → safe (blocked), score == 0.0 → vulnerable.
        """
        test_cases = risk_assessment.test_cases or []

        vulnerabilities_found: List[Dict] = []
        safe_responses: List[Dict] = []
        vuln_breakdown: Dict[str, Dict[str, int]] = {}
        safe_count = 0
        vuln_count = 0
        errored_count = 0

        for tc in test_cases:
            # Three states: safe (1.0), vulnerable (0.0), errored (None)
            if tc.score is None:
                errored_count += 1
                state = "errored"
            elif tc.score == 1.0:
                safe_count += 1
                state = "safe"
            else:
                vuln_count += 1
                state = "vulnerable"

            item = {
                "prompt": tc.input or "",
                "response": tc.actual_output or "",
                "passed": state == "safe",
                "score": tc.score,
                "vulnerability": tc.vulnerability or "unknown",
                "attack_method": tc.attack_method or "unknown",
                "reason": tc.reason or "",
            }

            if state == "safe":
                safe_responses.append(item)
            elif state == "vulnerable":
                vulnerabilities_found.append(item)
            # errored items are counted but not shown in top vulnerabilities

            key = tc.vulnerability or "unknown"
            entry = vuln_breakdown.setdefault(
                key, {"total": 0, "safe": 0, "vulnerable": 0, "errored": 0}
            )
            entry["total"] += 1
            entry[state] += 1

        total = len(test_cases)
        evaluated = safe_count + vuln_count  # exclude errored from rate
        overview = risk_assessment.overview
        return {
            "total_tests": total,
            "safe": safe_count,
            "vulnerable": vuln_count,
            "errored": errored_count,
            "safety_rate": safe_count / evaluated if evaluated > 0 else 0.0,
            "overview": overview.model_dump(mode="json") if overview else None,
            "vulnerability_breakdown": vuln_breakdown,
            "top_vulnerabilities": vulnerabilities_found[:10],
            "sample_safe": safe_responses[:5],
            "config": {
                "purpose": self.config.purpose,
                "vulnerabilities": self.config.vulnerabilities,
                "attacks": self.config.attacks,
                "attacks_per_type": self.config.attacks_per_type,
            },
        }
