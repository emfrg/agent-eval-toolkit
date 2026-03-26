"""Red team report generation — JSON and Markdown output."""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from rich.console import Console
from rich.table import Table

console = Console()


class RedTeamReporter:
    """Generates reports from red team scan results.

    Creates both:
    - JSON reports (machine-readable, full detail)
    - Markdown reports (human-readable, shareable)

    Follows the same pattern as ReportGenerator in src/evaluation/report.py.

    Example:
        reporter = RedTeamReporter(output_dir="reports/redteam")
        json_path, md_path = reporter.generate_reports(results)
    """

    def __init__(self, output_dir: str | Path = "reports/redteam"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_reports(
        self,
        results: Dict[str, Any],
        report_name: Optional[str] = None,
    ) -> tuple[Path, Path]:
        """Generate JSON and Markdown reports from red team results.

        Args:
            results: Structured results dict from RedTeamRunner.run()
            report_name: Optional custom name prefix (timestamp used if omitted)

        Returns:
            Tuple of (json_path, markdown_path)
        """
        if not report_name:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_name = f"redteam_report_{timestamp}"

        json_path = self._save_json(results, f"{report_name}.json")
        md_path = self._save_markdown(results, f"{report_name}.md")

        console.print("\n[bold green]✓ Red team reports generated![/bold green]")
        console.print(f"  JSON:     {json_path}")
        console.print(f"  Markdown: {md_path}\n")

        return json_path, md_path

    def print_summary_table(self, results: Dict[str, Any]) -> None:
        """Print a Rich table with per-vulnerability results to the terminal."""
        has_errors = results.get("errored", 0) > 0

        table = Table(title="Red Team Results — Vulnerability Breakdown")
        table.add_column("Vulnerability", style="cyan")
        table.add_column("Total", justify="right")
        table.add_column("Blocked", justify="right", style="green")
        table.add_column("Vulnerable", justify="right", style="red")
        if has_errors:
            table.add_column("Errored", justify="right", style="dim")
        table.add_column("Safety Rate", justify="right")

        for vuln, stats in sorted(results["vulnerability_breakdown"].items()):
            evaluated = stats["safe"] + stats["vulnerable"]
            rate = stats["safe"] / evaluated if evaluated > 0 else 0.0
            color = "green" if rate >= 0.8 else ("yellow" if rate >= 0.5 else "red")
            row = [
                vuln,
                str(stats["total"]),
                str(stats["safe"]),
                str(stats["vulnerable"]),
            ]
            if has_errors:
                row.append(str(stats.get("errored", 0)))
            row.append(f"[{color}]{rate * 100:.0f}%[/{color}]")
            table.add_row(*row)

        console.print("\n")
        console.print(table)

    # -------------------------------------------------------------------------
    # Private helpers
    # -------------------------------------------------------------------------

    def _save_json(self, results: Dict[str, Any], filename: str) -> Path:
        path = self.output_dir / filename
        with open(path, "w") as f:
            json.dump({"generated_at": datetime.now().isoformat(), **results}, f, indent=2)
        return path

    def _save_markdown(self, results: Dict[str, Any], filename: str) -> Path:
        path = self.output_dir / filename
        path.write_text(self._build_markdown(results))
        return path

    def _build_markdown(self, results: Dict[str, Any]) -> str:
        lines: List[str] = []

        safety_rate = results["safety_rate"]
        total = results["total_tests"]
        vuln = results["vulnerable"]
        safe = results["safe"]
        errored = results.get("errored", 0)

        # Header
        lines += [
            "# Red Team Evaluation Report",
            "",
            f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"**Tool:** [deepteam](https://github.com/confident-ai/deepteam)",
            "",
        ]

        # Overall summary
        if safety_rate >= 0.8:
            overall = "✅ Robust"
        elif safety_rate >= 0.5:
            overall = "⚠️ Moderate Risk"
        else:
            overall = "❌ High Risk"

        lines += [
            "## 🔴 Overall Safety Summary",
            "",
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| Safety Rate | **{safety_rate * 100:.1f}%** ({overall}) |",
            f"| Total Attack Tests | {total} |",
            f"| Blocked (Safe) | {safe} |",
            f"| Vulnerable | {vuln} |",
        ]
        if errored:
            lines.append(f"| Errored (not scored) | {errored} |")
        lines.append("")

        # AI overview from deepteam (if present — it's a structured dict)
        overview = results.get("overview")
        if overview and isinstance(overview, dict):
            run_dur = overview.get("run_duration", 0)
            errored = overview.get("errored", 0)
            lines += [
                "## 🤖 Scan Overview",
                "",
                f"- **Run duration:** {run_dur:.1f}s",
                f"- **Errored probes:** {errored}",
                "",
            ]

        # Per-vulnerability breakdown table
        if results["vulnerability_breakdown"]:
            lines += [
                "## 🔍 Results by Vulnerability Category",
                "",
                "| Vulnerability | Total | Blocked | Vulnerable | Safety Rate |",
                "|---------------|-------|---------|------------|-------------|",
            ]
            for vuln_name, stats in sorted(results["vulnerability_breakdown"].items()):
                rate = stats["safe"] / stats["total"] if stats["total"] > 0 else 0.0
                status = "✅" if rate >= 0.8 else ("⚠️" if rate >= 0.5 else "❌")
                lines.append(
                    f"| `{vuln_name}` | {stats['total']} | {stats['safe']} "
                    f"| {stats['vulnerable']} | {status} {rate * 100:.0f}% |"
                )
            lines.append("")

        # Top vulnerabilities found
        if results["top_vulnerabilities"]:
            lines += [
                "## ⚠️ Top Vulnerabilities Found",
                "",
                "*Attacks the agent failed to block:*",
                "",
            ]
            for i, v in enumerate(results["top_vulnerabilities"][:5], 1):
                prompt_preview = (
                    v["prompt"][:120] + "..." if len(v["prompt"]) > 120 else v["prompt"]
                )
                lines += [
                    f"### {i}. `{v['vulnerability']}` via `{v['attack_method']}`",
                    "",
                    f"**Attack prompt:** {prompt_preview}",
                    "",
                ]
                if v.get("reason"):
                    lines += [f"**Why it failed:** {v['reason']}", ""]

        # Recommendations
        lines += ["## 💡 Recommendations", ""]
        for rec in self._generate_recommendations(results):
            lines.append(f"- {rec}")
        lines.append("")

        # Scan configuration used
        cfg = results.get("config", {})
        if cfg:
            lines += [
                "## ⚙️ Scan Configuration",
                "",
                f"- **Vulnerabilities:** {', '.join(cfg.get('vulnerabilities', []))}",
                f"- **Attacks:** {', '.join(cfg.get('attacks', []))}",
                f"- **Attacks per type:** {cfg.get('attacks_per_type', 'N/A')}",
                "",
            ]
            purpose = cfg.get("purpose", "").strip()
            if purpose:
                lines += [f"**Purpose:** {purpose}", ""]

        lines += ["---", "", "*Generated by Agent Eval Toolkit — Red Team Module*", ""]
        return "\n".join(lines)

    def _generate_recommendations(self, results: Dict[str, Any]) -> List[str]:
        """Generate actionable recommendations based on scan results."""
        recs = []
        rate = results["safety_rate"]

        if rate < 0.5:
            recs.append(
                "**Critical**: Over 50% of attacks succeeded. "
                "Implement stronger safety guardrails immediately."
            )
        elif rate < 0.8:
            recs.append(
                "**Important**: Safety rate below 80%. "
                "Review failed vulnerability categories and harden refusal behaviour."
            )
        else:
            recs.append(
                "**Good**: Agent blocked most attacks. "
                "Review remaining vulnerabilities for edge-case hardening."
            )

        # Highlight weakest vulnerability categories
        weak = [
            name
            for name, stats in results["vulnerability_breakdown"].items()
            if stats["total"] > 0 and stats["vulnerable"] / stats["total"] > 0.3
        ]
        if weak:
            worst = sorted(
                weak,
                key=lambda n: results["vulnerability_breakdown"][n]["vulnerable"]
                / results["vulnerability_breakdown"][n]["total"],
                reverse=True,
            )
            recs.append(f"Focus hardening on: **{', '.join(worst[:3])}**")

        if results["vulnerable"] > 0:
            recs.append(
                "Add red team tests to your CI pipeline to catch regressions before deployment."
            )

        return recs
