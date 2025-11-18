#!/usr/bin/env python
"""Minimal Evaluation Test Script

This script demonstrates the complete evaluation pipeline with minimal configuration.
It's designed to help you understand how all the pieces fit together.

What this script does:
1. Loads a minimal persona configuration (2 personas, 4 total conversations)
2. Runs simulations with the example agent service
3. Saves conversation logs
4. Evaluates conversations using LLM-as-a-judge
5. Generates JSON and Markdown reports

Prerequisites:
- Example agent service running at http://localhost:5555
  Start it with: python examples/example_agent/run.py

- OPENAI_API_KEY set in .env file

Usage:
    python test_minimal_eval.py

Output:
    - Logs saved to: minimal_test_output/logs/
    - Reports saved to: minimal_test_output/reports/
"""

import asyncio
import sys
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.simulation.config_loader import load_personas_from_yaml
from src.simulation.runner import SimulationRunner
from src.evaluation.judge import JudgeEvaluator
from src.evaluation.report import ReportGenerator
from src.core.logging import ConversationLogger

# Import the example agent adapter
from examples.example_agent_adapter import agent_callback


async def run_minimal_evaluation():
    """Run a complete minimal evaluation pipeline."""

    print("=" * 70)
    print("🧪 MINIMAL EVALUATION TEST")
    print("=" * 70)
    print()

    # Configuration
    config_path = "config/personas_minimal.yaml"
    output_base = Path("minimal_test_output")
    logs_dir = output_base / "logs"
    reports_dir = output_base / "reports"

    # Create output directories
    logs_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    # ========================================================================
    # STEP 1: Load Persona Configuration
    # ========================================================================
    print("📋 Step 1: Loading persona configuration...")
    print(f"   Config file: {config_path}")

    try:
        goldens, persona_configs = load_personas_from_yaml(config_path)
    except FileNotFoundError:
        print(f"❌ Error: Config file not found at {config_path}")
        print("   Make sure you're running from the project root directory.")
        return

    total_conversations = sum(p.conversations for p in persona_configs)
    print(f"   ✅ Loaded {len(persona_configs)} personas")
    print(f"   ✅ Total conversations to simulate: {total_conversations}")

    for config in persona_configs:
        print(f"      - {config.name}: {config.conversations} conversations, "
              f"max {config.max_turns} turns")
    print()

    # ========================================================================
    # STEP 2: Run Simulations
    # ========================================================================
    print("🎭 Step 2: Running simulations...")
    print("   This will simulate realistic user conversations with your agent.")
    print("   DeepEval will act as different user personas.")
    print()

    runner = SimulationRunner(
        agent_callback=agent_callback,
        max_concurrent=2  # Lower concurrency for minimal test
    )

    try:
        test_cases = await runner.run_simulations(goldens, persona_configs)
        print(f"   ✅ Completed {len(test_cases)} conversations")
    except Exception as e:
        print(f"   ❌ Simulation failed: {e}")
        print()
        print("   Common issues:")
        print("   1. Agent service not running → Start with: python examples/example_agent/run.py")
        print("   2. OPENAI_API_KEY not set → Check your .env file")
        return

    print()

    # ========================================================================
    # STEP 3: Save Conversation Logs
    # ========================================================================
    print("💾 Step 3: Saving conversation logs...")

    logger = ConversationLogger(output_dir=str(logs_dir))

    # Save in both formats for demonstration
    jsonl_path = logger.save_as_jsonl(test_cases)
    json_path = logger.save_as_json(test_cases)

    print(f"   ✅ Saved logs to:")
    print(f"      - JSONL: {jsonl_path}")
    print(f"      - JSON:  {json_path}")
    print()

    # ========================================================================
    # STEP 4: Evaluate Conversations
    # ========================================================================
    print("⚖️  Step 4: Evaluating conversations with LLM-as-a-judge...")
    print("   Using criteria from config/judge_prompt.md")
    print()

    evaluator = JudgeEvaluator(
        custom_prompt_path="config/judge_prompt.md",
        threshold=0.7,  # Pass threshold (0.0-1.0)
        use_default_metrics=True  # Use built-in metrics
    )

    # Run evaluation with verbose output
    results = evaluator.evaluate_conversations(test_cases, verbose=True)
    print()

    # ========================================================================
    # STEP 5: Generate Reports
    # ========================================================================
    print("📊 Step 5: Generating evaluation reports...")

    # Get summary statistics
    summary = evaluator.get_summary(results)

    # Generate reports
    report_gen = ReportGenerator(output_dir=str(reports_dir))
    json_path, md_path = report_gen.generate_reports(results, summary)

    print(f"   ✅ Generated reports:")
    print(f"      - JSON: {json_path}")
    print(f"      - MARKDOWN: {md_path}")
    print()

    # ========================================================================
    # STEP 6: Display Summary
    # ========================================================================
    print("=" * 70)
    print("📈 EVALUATION SUMMARY")
    print("=" * 70)
    print()
    print(f"Total Conversations:  {summary['total_conversations']}")
    print(f"Passed:               {summary['passed']} "
          f"({summary['pass_rate']*100:.1f}%)")
    print(f"Failed:               {summary['failed']} "
          f"({(1-summary['pass_rate'])*100:.1f}%)")
    print()

    # Get average score from summary (already calculated)
    # Try both possible metric names (DeepEval may use either)
    overall_avg = summary["metric_averages"].get("Overall Quality [Conversational GEval]", {}).get("average")
    if overall_avg is None:
        overall_avg = summary["metric_averages"].get("Overall Quality", {}).get("average", 0.0)
    print(f"Average Score:        {overall_avg:.3f}")
    print(f"Pass Threshold:       {summary['threshold']:.2f}")
    print()

    # Overall pass/fail based on whether all conversations passed
    if summary['failed'] == 0:
        print("✅ OVERALL: PASSED")
    else:
        print("❌ OVERALL: FAILED")

    print()
    print("=" * 70)
    print("🎉 Evaluation Complete!")
    print("=" * 70)
    print()
    print("Next steps:")
    print("1. Review the Markdown report for detailed insights")
    print(f"   → {md_path}")
    print()
    print("2. Examine conversation logs to see what your agent said")
    print(f"   → {logs_dir}")
    print()
    print("3. Adjust personas or judge criteria and re-run")
    print()
    print("4. Scale up to full evaluation with all personas:")
    print("   → llm-evals-starter simulate --agent examples/example_agent_adapter.py")
    print()


if __name__ == "__main__":
    # Check if running from correct directory
    if not Path("config").exists():
        print("❌ Error: Must run from project root directory")
        print("   Current directory:", Path.cwd())
        sys.exit(1)

    # Check if .env exists
    if not Path(".env").exists():
        print("⚠️  Warning: No .env file found")
        print("   Make sure OPENAI_API_KEY is set in your environment")
        print()

    # Run the evaluation
    try:
        asyncio.run(run_minimal_evaluation())
    except KeyboardInterrupt:
        print("\n\n⚠️  Evaluation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
