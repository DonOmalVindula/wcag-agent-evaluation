"""
Batch orchestrator for the agent task execution phase (Phase 3).

Runs the full evaluation matrix — sites x tasks x observation modes x
repetitions — over the stratified 200-site sample, with:

- Resumability: every completed run is appended to a JSONL file; on
  restart, already-completed (task_id, mode, rep) combinations are skipped.
- Bounded concurrency (default 3, matching the ethics statement in Ch.4).
- Real token accounting aggregated across runs for cost reporting.

Usage:
    python -m src.agent.batch_runner --dry-run --limit-sites 5
    python -m src.agent.batch_runner --modes accessibility_tree dom screenshot \
        --categories navigation search info --reps 3 \
        --out data/results/agent_runs.jsonl
"""

import argparse
import asyncio
import json
import time
from pathlib import Path

from .task_runner import TaskResult, TaskStatus, run_task
from .task_templates import generate_all_tasks

DEFAULT_SAMPLE = "data/processed/final_sample.csv"
DEFAULT_OUT = "data/results/agent_runs.jsonl"
ALL_MODES = ["accessibility_tree", "dom", "screenshot"]


def run_key(task_id: str, mode: str, rep: int) -> str:
    return f"{task_id}|{mode}|{rep}"


def load_completed(out_path: Path, include_errors: bool = False) -> set[str]:
    """
    Read the checkpoint file and return keys of completed runs.

    Runs that ended in status "error" (crashes, API failures — not task
    failures) are excluded by default so they are retried on resume; the
    analysis stage deduplicates by keeping the latest record per key.
    """
    completed = set()
    if out_path.exists():
        with open(out_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    if rec.get("status") == "error" and not include_errors:
                        continue
                    completed.add(run_key(rec["task_id"], rec["observation_mode"], rec["run_index"]))
                except (json.JSONDecodeError, KeyError):
                    continue
    return completed


def load_dotenv(path: Path) -> None:
    """Load KEY=VALUE lines from a .env file into os.environ (no override)."""
    import os

    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


async def robots_allows(url: str, cache: dict) -> bool:
    """
    Check whether the site's robots.txt permits fetching the seed URL for
    a generic user-agent ("*"). Missing/unreachable robots.txt counts as
    allowed (standard interpretation). Cached per origin.
    """
    import urllib.robotparser
    from urllib.parse import urlsplit

    import httpx

    parts = urlsplit(url)
    origin = f"{parts.scheme}://{parts.netloc}"
    if origin in cache:
        return cache[origin]
    allowed = True
    try:
        async with httpx.AsyncClient(follow_redirects=True) as client:
            resp = await client.get(origin + "/robots.txt", timeout=10.0)
        if resp.status_code < 400 and "text" in resp.headers.get("content-type", "text"):
            rp = urllib.robotparser.RobotFileParser()
            rp.parse(resp.text.splitlines())
            allowed = rp.can_fetch("*", url)
    except Exception:
        allowed = True
    cache[origin] = allowed
    return allowed


async def execute_matrix(args) -> None:
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    pairs = generate_all_tasks(
        args.sample,
        categories=args.categories,
        sectors=args.sectors,
        limit_sites=args.limit_sites,
    )

    # Ethics gate (thesis Ch.4): exclude sites whose robots.txt disallows
    # automated access to the seed URL; record exclusions for reporting.
    robots_cache: dict = {}
    site_urls = {site.domain: site.url for site, _ in pairs}
    allowed_flags = await asyncio.gather(
        *(robots_allows(u, robots_cache) for u in site_urls.values())
    )
    excluded = sorted(
        domain for (domain, _), ok in zip(site_urls.items(), allowed_flags) if not ok
    )
    if excluded:
        excl_path = out_path.parent / "excluded_robots.json"
        excl_path.write_text(json.dumps(excluded, indent=2))
        print(f"robots.txt exclusions: {len(excluded)} site(s) -> {excl_path}: {', '.join(excluded)}")
        pairs = [(s, t) for s, t in pairs if s.domain not in excluded]

    # Build the run matrix
    matrix = [
        (site, task, mode, rep)
        for site, task in pairs
        for mode in args.modes
        for rep in range(args.reps)
    ]

    completed = load_completed(out_path)
    todo = [
        (site, task, mode, rep)
        for site, task, mode, rep in matrix
        if run_key(task.task_id, mode, rep) not in completed
    ]
    if args.max_runs and len(todo) > args.max_runs:
        print(f"Chunked session: doing {args.max_runs} of {len(todo)} remaining runs, then exiting")
        todo = todo[:args.max_runs]

    n_sites = len({site.domain for site, _ in pairs})
    print(f"Matrix: {n_sites} sites x {len(args.categories or ['navigation','search','info'])} "
          f"task categories x {len(args.modes)} modes x {args.reps} reps "
          f"= {len(matrix)} runs ({len(completed)} already done, {len(todo)} to go)")
    if args.dry_run:
        print("DRY RUN: using mock LLM provider (no API calls, no cost)")

    provider = "mock" if args.dry_run else "anthropic"
    sem = asyncio.Semaphore(args.concurrency)
    write_lock = asyncio.Lock()
    stats = {"done": 0, "success": 0, "verified": 0, "error": 0,
             "input_tokens": 0, "output_tokens": 0, "api_error_streak": 0,
             "net_error_streak": 0}
    abort = asyncio.Event()
    net_ok = asyncio.Event()
    net_ok.set()
    t0 = time.time()

    NET_ERROR_MARKERS = ("ERR_NAME_NOT_RESOLVED", "ERR_INTERNET_DISCONNECTED",
                         "ERR_NETWORK_CHANGED", "ERR_CONNECTION_CLOSED",
                         "Page.goto: Timeout")  # timeout storms = degraded network;
                         # worst case a single dead site false-triggers one 30s probe pause

    async def wait_for_network():
        """Probe connectivity until it returns, then resume the batch."""
        import httpx
        print("\nNETWORK OUTAGE detected — pausing batch, probing every 30s...")
        while True:
            await asyncio.sleep(30)
            try:
                async with httpx.AsyncClient() as client:
                    await client.get("https://www.google.com/generate_204", timeout=10.0)
                break
            except Exception:
                continue
        stats["net_error_streak"] = 0
        print("Network restored — resuming batch.")
        net_ok.set()

    async def one_run(site, task, mode, rep):
        if abort.is_set():
            return
        await net_ok.wait()
        async with sem:
            if abort.is_set():
                return
            await net_ok.wait()
            try:
                result: TaskResult = await run_task(
                    task,
                    observation_mode=mode,
                    llm_provider=provider,
                    llm_model=args.model,
                    run_index=rep,
                )
            except Exception as e:
                # Never let one failure kill the batch — record it and move on
                result = TaskResult(
                    url=task.url, task_description=task.description,
                    observation_mode=mode, status=TaskStatus.ERROR,
                    task_id=task.task_id, category=task.category, run_index=rep,
                    error_message=f"Runner crashed: {e}",
                )
            rec = result.to_dict()
            rec["sector"] = site.sector
            rec["domain"] = site.domain
            rec["compliance_score"] = site.compliance_score
            rec["compliance_bin"] = site.compliance_bin
            # Circuit breaker: a run of consecutive API-level errors means a
            # systemic problem (auth, billing, outage) — abort instead of
            # churning through the whole matrix recording garbage.
            is_api_error = rec["status"] == "error" and (
                "API 4" in rec["error"] or "api.anthropic.com" in rec["error"]
            )
            is_net_error = rec["status"] == "error" and any(
                m in rec["error"] for m in NET_ERROR_MARKERS
            )
            async with write_lock:
                stats["api_error_streak"] = stats["api_error_streak"] + 1 if is_api_error else 0
                stats["net_error_streak"] = stats["net_error_streak"] + 1 if is_net_error else 0
                if stats["api_error_streak"] >= 15 and not abort.is_set():
                    abort.set()
                    print(f"\nFATAL: {stats['api_error_streak']} consecutive API errors — "
                          f"aborting batch. Last error: {rec['error'][:200]}")
                if stats["net_error_streak"] >= 9 and net_ok.is_set():
                    net_ok.clear()
                    asyncio.ensure_future(wait_for_network())
                with open(out_path, "a") as f:
                    f.write(json.dumps(rec) + "\n")
                stats["done"] += 1
                stats["success"] += 1 if rec["status"] == "success" else 0
                stats["verified"] += 1 if rec.get("verified_success") else 0
                stats["error"] += 1 if rec["status"] == "error" else 0
                stats["input_tokens"] += rec.get("input_tokens", 0)
                stats["output_tokens"] += rec.get("output_tokens", 0)
                elapsed = time.time() - t0
                rate = stats["done"] / elapsed if elapsed > 0 else 0
                eta_min = (len(todo) - stats["done"]) / rate / 60 if rate > 0 else float("inf")
                print(f"[{stats['done']}/{len(todo)}] {site.domain} | {task.category} | {mode} | rep{rep} "
                      f"-> {rec['status']} (verified={rec.get('verified_success')}) "
                      f"| {rec['total_steps']} steps {rec['duration_seconds']}s "
                      f"| ETA {eta_min:.0f}m")

    await asyncio.gather(*(one_run(*item) for item in todo))

    elapsed = time.time() - t0
    print("\n" + "=" * 70)
    print(f"Batch complete: {stats['done']} runs in {elapsed / 60:.1f} min")
    print(f"  Agent-reported success: {stats['success']}/{stats['done']}")
    print(f"  Keyword-verified success: {stats['verified']}/{stats['done']}")
    print(f"  Errors: {stats['error']}")
    print(f"  Tokens: {stats['input_tokens']:,} in / {stats['output_tokens']:,} out")
    print(f"  Results: {out_path} ({len(load_completed(out_path))} total records)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch agent task execution over the site sample")
    parser.add_argument("--sample", default=DEFAULT_SAMPLE, help="Sample CSV path")
    parser.add_argument("--out", default=DEFAULT_OUT, help="Output JSONL (checkpoint) path")
    parser.add_argument("--modes", nargs="+", default=ALL_MODES, choices=ALL_MODES)
    parser.add_argument("--categories", nargs="+", default=None,
                        choices=["navigation", "search", "info"],
                        help="Task categories (default: all three)")
    parser.add_argument("--sectors", nargs="+", default=None,
                        help="Restrict to specific sectors")
    parser.add_argument("--reps", type=int, default=3, help="Repetitions per task-mode combo")
    parser.add_argument("--limit-sites", type=int, default=None,
                        help="Limit total sites (balanced across sectors) — for pilots")
    parser.add_argument("--concurrency", type=int, default=3)
    parser.add_argument("--model", default="claude-sonnet-5")
    parser.add_argument("--dry-run", action="store_true",
                        help="Use mock LLM provider (validates plumbing, zero cost)")
    parser.add_argument("--max-runs", type=int, default=None,
                        help="Process at most N runs this session, then exit (chunked mode)")
    args = parser.parse_args()

    load_dotenv(Path(__file__).resolve().parents[2] / ".env")
    asyncio.run(execute_matrix(args))


if __name__ == "__main__":
    main()
