"""
CLI script to run agent tasks.

Usage:
    python -m src.agent.run_agent --task navigation --mode accessibility_tree
    python -m src.agent.run_agent --url https://www.bbc.com --task-desc "Find the Sport section"
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

from .task_runner import TaskResult, WebTask, run_task, run_task_multi_mode
from .tasks import ALL_TASKS, NAVIGATION_TASKS, SEARCH_TASKS, INFO_TASKS


TASK_GROUPS = {
    "navigation": NAVIGATION_TASKS,
    "search": SEARCH_TASKS,
    "info": INFO_TASKS,
    "all": ALL_TASKS,
}


def print_result(result: TaskResult) -> None:
    """Print a human-readable task result."""
    status_icon = {
        "success": "[OK]",
        "failure": "[FAIL]",
        "error": "[ERR]",
        "timeout": "[TIME]",
    }
    icon = status_icon.get(result.status.value, "[??]")

    print(f"\n{'='*70}")
    print(f"{icon} {result.task_description}")
    print(f"  URL:   {result.url}")
    print(f"  Mode:  {result.observation_mode}")
    print(f"  Steps: {result.total_steps}")
    print(f"  Time:  {result.duration:.1f}s")
    if result.error_message:
        print(f"  Error: {result.error_message}")

    if result.steps:
        print(f"\n  Step log:")
        for s in result.steps:
            print(f"    Step {s.step_number}: {s.action} ({s.observation_tokens} tokens)")
            if s.reasoning:
                print(f"           Reasoning: {s.reasoning[:100]}...")


async def main() -> None:
    parser = argparse.ArgumentParser(description="Run web agent tasks")
    parser.add_argument("--task", choices=TASK_GROUPS.keys(), help="Task group to run")
    parser.add_argument("--url", help="Custom URL to test")
    parser.add_argument("--task-desc", help="Custom task description")
    parser.add_argument("--success-criteria", default="Task completed successfully", help="Success criteria")
    parser.add_argument("--mode", default="accessibility_tree",
                       choices=["accessibility_tree", "dom", "screenshot"],
                       help="Observation mode")
    parser.add_argument("--multi-mode", action="store_true", help="Run across all modes")
    parser.add_argument("--output", "-o", help="Output JSON file")
    parser.add_argument("--model", default="claude-sonnet-5", help="LLM model")
    args = parser.parse_args()

    tasks: list[WebTask] = []

    if args.url and args.task_desc:
        tasks.append(WebTask(
            url=args.url,
            description=args.task_desc,
            success_criteria=args.success_criteria,
        ))
    elif args.task:
        tasks = TASK_GROUPS[args.task]
    else:
        print("Provide either --task <group> or --url + --task-desc")
        sys.exit(1)

    print(f"Running {len(tasks)} task(s) in '{args.mode}' mode...")

    all_results: list[TaskResult] = []
    for task in tasks:
        if args.multi_mode:
            results = await run_task_multi_mode(task, llm_model=args.model)
        else:
            results = [await run_task(task, observation_mode=args.mode, llm_model=args.model)]
        all_results.extend(results)
        for r in results:
            print_result(r)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        data = [r.to_dict() for r in all_results]
        output_path.write_text(json.dumps(data, indent=2))
        print(f"\nResults saved to {output_path}")

    # Summary
    successes = sum(1 for r in all_results if r.success)
    print(f"\n{'='*70}")
    print(f"Summary: {successes}/{len(all_results)} tasks succeeded")


if __name__ == "__main__":
    asyncio.run(main())
