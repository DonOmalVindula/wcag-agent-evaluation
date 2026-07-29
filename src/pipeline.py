"""
Main Pipeline - Orchestrates the full dual-evaluation framework.

Runs:
1. Accessibility auditing on target websites
2. Agent task execution on the same websites
3. Correlation analysis between accessibility scores and agent performance
"""

import asyncio
import json
from pathlib import Path

from .accessibility.axe_auditor import audit_urls, save_results as save_audit_results
from .agent.task_runner import WebTask, run_task
from .analysis.correlation import (
    load_audit_results,
    load_agent_results,
    merge_datasets,
    run_correlation_analysis,
    feature_importance_analysis,
    generate_report,
)


async def run_full_pipeline(
    urls: list[str],
    tasks: list[WebTask],
    observation_modes: list[str] | None = None,
    output_dir: str = "data/processed",
) -> None:
    """
    Run the complete dual-evaluation pipeline.

    Args:
        urls: Websites to evaluate.
        tasks: Agent tasks to execute.
        observation_modes: Observation modes to test.
        output_dir: Directory for output files.
    """
    if observation_modes is None:
        observation_modes = ["accessibility_tree", "dom"]

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # === Step 1: Accessibility Auditing ===
    print("\n[1/3] Running accessibility audits...")
    audit_results = await audit_urls(urls)
    audit_file = output_path / "audit_results.json"
    save_audit_results(audit_results, audit_file)
    print(f"  Audited {len(audit_results)} sites")

    # === Step 2: Agent Task Execution ===
    print("\n[2/3] Running agent tasks...")
    agent_results = []
    for task in tasks:
        for mode in observation_modes:
            print(f"  Running: {task.description} [{mode}]")
            try:
                result = await run_task(task, observation_mode=mode)
                agent_results.append(result.to_dict())
                status = "OK" if result.success else result.status.value
                print(f"    -> {status} ({result.total_steps} steps, {result.duration:.1f}s)")
            except Exception as e:
                print(f"    -> ERROR: {e}")

    agent_file = output_path / "agent_results.json"
    agent_file.write_text(json.dumps(agent_results, indent=2))
    print(f"  Completed {len(agent_results)} task runs")

    # === Step 3: Correlation Analysis ===
    print("\n[3/3] Running correlation analysis...")
    audit_df = load_audit_results(audit_file)
    agent_df = load_agent_results(agent_file)
    merged = merge_datasets(audit_df, agent_df)

    if len(merged) < 3:
        print("  WARNING: Insufficient data for correlation analysis (need >= 3 matched sites)")
        print(f"  Audit sites: {len(audit_df)}, Agent sites: {agent_df['url'].nunique()}")
        print(f"  Matched: {len(merged)}")
        return

    correlations = run_correlation_analysis(merged)
    report = generate_report(correlations)
    print(report)

    # Save analysis
    analysis_output = {
        "correlations": [c.to_dict() for c in correlations],
        "merged_data": merged.to_dict(orient="records"),
    }
    analysis_file = output_path / "analysis_results.json"
    analysis_file.write_text(json.dumps(analysis_output, indent=2, default=str))
    print(f"\nAnalysis saved to {analysis_file}")
