"""Report generation for evaluation results."""

import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
from rich.console import Console

console = Console()


class ReportGenerator:
    """Generates human-readable reports from evaluation results.

    Creates both:
    - JSON reports (machine-readable, detailed)
    - Markdown reports (human-readable, perfect for sharing)

    Example:
        ```python
        generator = ReportGenerator(output_dir="reports")
        generator.generate_reports(
            evaluation_results=results,
            summary=summary
        )
        ```
    """

    def __init__(self, output_dir: str | Path = "reports"):
        """Initialize the report generator.

        Args:
            output_dir: Directory to save reports
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_reports(
        self,
        evaluation_results: List[Dict[str, Any]],
        summary: Dict[str, Any],
        report_name: Optional[str] = None
    ) -> tuple[Path, Path]:
        """Generate both JSON and Markdown reports.

        Args:
            evaluation_results: List of detailed evaluation results
            summary: Summary statistics from evaluator
            report_name: Optional custom report name

        Returns:
            Tuple of (json_path, markdown_path)
        """
        if not report_name:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_name = f"evaluation_report_{timestamp}"

        # Generate JSON report
        json_path = self.save_json_report(
            evaluation_results,
            summary,
            filename=f"{report_name}.json"
        )

        # Generate Markdown report
        markdown_path = self.save_markdown_report(
            evaluation_results,
            summary,
            filename=f"{report_name}.md"
        )

        console.print(f"\n[bold green]✓ Reports generated![/bold green]")
        console.print(f"  JSON: {json_path}")
        console.print(f"  Markdown: {markdown_path}\n")

        return json_path, markdown_path

    def save_json_report(
        self,
        evaluation_results: List[Dict[str, Any]],
        summary: Dict[str, Any],
        filename: str = "report_raw.json"
    ) -> Path:
        """Save detailed results as JSON.

        Args:
            evaluation_results: List of evaluation results
            summary: Summary statistics
            filename: Output filename

        Returns:
            Path to saved JSON file
        """
        output_path = self.output_dir / filename

        report_data = {
            "generated_at": datetime.now().isoformat(),
            "summary": summary,
            "detailed_results": evaluation_results
        }

        with open(output_path, 'w') as f:
            json.dump(report_data, f, indent=2)

        return output_path

    def save_markdown_report(
        self,
        evaluation_results: List[Dict[str, Any]],
        summary: Dict[str, Any],
        filename: str = "summary.md"
    ) -> Path:
        """Save human-readable Markdown report.

        Perfect for:
        - Sharing with team
        - Including in PRs
        - Posting on LinkedIn!

        Args:
            evaluation_results: List of evaluation results
            summary: Summary statistics
            filename: Output filename

        Returns:
            Path to saved Markdown file
        """
        output_path = self.output_dir / filename

        markdown_content = self._generate_markdown_content(
            evaluation_results,
            summary
        )

        with open(output_path, 'w') as f:
            f.write(markdown_content)

        return output_path

    def _generate_markdown_content(
        self,
        evaluation_results: List[Dict[str, Any]],
        summary: Dict[str, Any]
    ) -> str:
        """Generate the markdown content for the report."""
        lines = []

        # Header
        lines.append("# Agent Evaluation Report")
        lines.append("")
        lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"**Judge Model:** {summary.get('judge_model', 'N/A')}")
        lines.append(f"**Threshold:** {summary.get('threshold', 0.7)}")
        lines.append("")

        # Overall Summary
        lines.append("## 📊 Overall Summary")
        lines.append("")
        lines.append(f"- **Total Conversations:** {summary['total_conversations']}")
        lines.append(f"- **Passed:** {summary['passed']} ({summary['pass_rate']*100:.1f}%)")
        lines.append(f"- **Failed:** {summary['failed']}")
        lines.append("")

        # Metric Averages
        if summary.get('metric_averages'):
            lines.append("## 📈 Metric Averages")
            lines.append("")
            for metric_name, stats in summary['metric_averages'].items():
                lines.append(f"### {metric_name}")
                lines.append("")
                lines.append(f"- **Average Score:** {stats['average']:.3f}")
                lines.append(f"- **Min Score:** {stats['min']:.3f}")
                lines.append(f"- **Max Score:** {stats['max']:.3f}")
                lines.append(f"- **Conversations Evaluated:** {stats['count']}")
                lines.append("")

        # Persona-level Statistics
        if summary.get('persona_statistics'):
            lines.append("## 👥 Performance by Persona")
            lines.append("")

            for persona_name, stats in summary['persona_statistics'].items():
                lines.append(f"### {persona_name}")
                lines.append("")
                lines.append(f"- **Conversations:** {stats['total']}")
                lines.append(f"- **Pass Rate:** {stats.get('pass_rate', 0)*100:.1f}%")

                if 'average_score' in stats:
                    lines.append(f"- **Average Score:** {stats['average_score']:.3f}")

                # Determine strengths vs weaknesses
                if stats.get('pass_rate', 0) >= 0.8:
                    lines.append("- **Status:** ✅ Strong performance")
                elif stats.get('pass_rate', 0) >= 0.6:
                    lines.append("- **Status:** ⚠️ Moderate performance")
                else:
                    lines.append("- **Status:** ❌ Needs improvement")

                lines.append("")

                # Add insights based on results
                insights = self._generate_persona_insights(
                    persona_name,
                    stats,
                    evaluation_results
                )
                if insights:
                    lines.append("**Key Insights:**")
                    for insight in insights:
                        lines.append(f"- {insight}")
                    lines.append("")

        # Detailed Results (Top Failures & Successes)
        lines.append("## 🔍 Detailed Results")
        lines.append("")

        # Top 5 failures
        failures = [r for r in evaluation_results if not r['passed']]
        if failures:
            lines.append("### ❌ Top Failures")
            lines.append("")
            for i, result in enumerate(failures[:5], 1):
                scenario = result['scenario'][:60] + "..." if len(result['scenario']) > 60 else result['scenario']
                lines.append(f"{i}. **{scenario}**")
                lines.append(f"   - Turns: {result['num_turns']}")

                # Show which metrics failed
                failed_metrics = [
                    f"{name} ({scores['score']:.2f})"
                    for name, scores in result['metric_scores'].items()
                    if not scores['passed']
                ]
                if failed_metrics:
                    lines.append(f"   - Failed metrics: {', '.join(failed_metrics)}")
                lines.append("")

        # Top 5 successes
        successes = [r for r in evaluation_results if r['passed']]
        if successes:
            # Sort by average score
            successes_sorted = sorted(
                successes,
                key=lambda r: sum(
                    s['score'] for s in r['metric_scores'].values() if s['score'] is not None
                ) / len(r['metric_scores']),
                reverse=True
            )

            lines.append("### ✅ Top Successes")
            lines.append("")
            for i, result in enumerate(successes_sorted[:5], 1):
                scenario = result['scenario'][:60] + "..." if len(result['scenario']) > 60 else result['scenario']
                avg_score = sum(
                    s['score'] for s in result['metric_scores'].values() if s['score'] is not None
                ) / len(result['metric_scores'])

                lines.append(f"{i}. **{scenario}**")
                lines.append(f"   - Turns: {result['num_turns']}")
                lines.append(f"   - Average Score: {avg_score:.3f}")
                lines.append("")

        # Recommendations
        lines.append("## 💡 Recommendations")
        lines.append("")
        recommendations = self._generate_recommendations(summary, evaluation_results)
        for rec in recommendations:
            lines.append(f"- {rec}")
        lines.append("")

        # Footer
        lines.append("---")
        lines.append("")
        lines.append("*Generated by AI Agent Conversation Simulator & Evaluator*")
        lines.append("")

        return "\n".join(lines)

    def _generate_persona_insights(
        self,
        persona_name: str,
        stats: Dict[str, Any],
        all_results: List[Dict[str, Any]]
    ) -> List[str]:
        """Generate insights for a specific persona."""
        insights = []

        # Get results for this persona
        persona_results = [
            r for r in all_results
            if r['metadata'].get('persona_name') == persona_name
        ]

        if not persona_results:
            return insights

        # Check conversation length patterns
        avg_turns = sum(r['num_turns'] for r in persona_results) / len(persona_results)
        if avg_turns > 15:
            insights.append(f"Conversations tend to be lengthy (avg {avg_turns:.1f} turns)")
        elif avg_turns < 5:
            insights.append(f"Conversations are brief (avg {avg_turns:.1f} turns)")

        # Check metric patterns
        metric_scores = {}
        for result in persona_results:
            for metric_name, scores in result['metric_scores'].items():
                if metric_name not in metric_scores:
                    metric_scores[metric_name] = []
                if scores['score'] is not None:
                    metric_scores[metric_name].append(scores['score'])

        # Find strongest and weakest metrics
        metric_avgs = {
            name: sum(scores) / len(scores)
            for name, scores in metric_scores.items()
            if scores
        }

        if metric_avgs:
            strongest = max(metric_avgs.items(), key=lambda x: x[1])
            weakest = min(metric_avgs.items(), key=lambda x: x[1])

            if strongest[1] > 0.8:
                insights.append(f"Strong in {strongest[0]} ({strongest[1]:.2f})")
            if weakest[1] < 0.6:
                insights.append(f"Needs improvement in {weakest[0]} ({weakest[1]:.2f})")

        return insights

    def _generate_recommendations(
        self,
        summary: Dict[str, Any],
        results: List[Dict[str, Any]]
    ) -> List[str]:
        """Generate actionable recommendations based on results."""
        recommendations = []

        pass_rate = summary.get('pass_rate', 0)

        # Overall performance recommendations
        if pass_rate < 0.5:
            recommendations.append(
                "**Critical**: Less than 50% of conversations passed. "
                "Review agent prompts and behavior urgently."
            )
        elif pass_rate < 0.7:
            recommendations.append(
                "**Important**: Pass rate below 70%. Consider refining agent responses and adding guardrails."
            )
        else:
            recommendations.append(
                "**Good**: Strong overall performance. Focus on edge cases and persona-specific improvements."
            )

        # Metric-specific recommendations
        metric_avgs = summary.get('metric_averages', {})
        for metric_name, stats in metric_avgs.items():
            if stats['average'] < 0.6:
                recommendations.append(
                    f"Improve **{metric_name}** - currently averaging {stats['average']:.2f}"
                )

        # Persona-specific recommendations
        persona_stats = summary.get('persona_statistics', {})
        weak_personas = [
            name for name, stats in persona_stats.items()
            if stats.get('pass_rate', 0) < 0.5
        ]

        if weak_personas:
            recommendations.append(
                f"Focus testing on personas: {', '.join(weak_personas)}"
            )

        # Conversation length recommendation
        avg_turns_all = sum(r['num_turns'] for r in results) / len(results) if results else 0
        if avg_turns_all > 20:
            recommendations.append(
                "Conversations are lengthy. Consider improving efficiency and providing more direct answers."
            )

        return recommendations
