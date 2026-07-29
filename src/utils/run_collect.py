"""
CLI script to collect URLs, audit them, and produce a stratified sample.

Usage:
    # Step 1: Collect candidate URLs
    python -m src.utils.run_collect collect --output data/raw/candidates.json

    # Step 2: Audit all candidates (takes a while)
    python -m src.utils.run_collect audit --input data/raw/candidates.json --output data/raw/candidates_scored.json

    # Step 3: Stratified sample
    python -m src.utils.run_collect sample --input data/raw/candidates_scored.json --per-sector 40 --output data/processed/final_sample.json
"""

import argparse
import asyncio
import sys
from pathlib import Path

from .url_collector import (
    collect_candidates,
    load_candidates,
    save_candidates,
    save_candidates_csv,
    stratified_sample,
)


async def cmd_collect(args: argparse.Namespace) -> None:
    """Collect candidate URLs from Tranco + seed lists."""
    candidates = collect_candidates(tranco_top_n=args.tranco_top_n)
    save_candidates(candidates, args.output)
    save_candidates_csv(candidates, args.output.replace(".json", ".csv"))


async def cmd_audit(args: argparse.Namespace) -> None:
    """Run accessibility audits on all candidates to get compliance scores."""
    from ..accessibility.axe_auditor import audit_url

    candidates = load_candidates(args.input)
    total = len(candidates)
    print(f"Auditing {total} candidates...")

    for i, candidate in enumerate(candidates, 1):
        if candidate.compliance_score is not None and not args.force:
            print(f"  [{i}/{total}] {candidate.domain} — already scored ({candidate.compliance_score:.1%}), skipping")
            continue

        try:
            result = await audit_url(candidate.url, timeout=20000)
            candidate.compliance_score = result.compliance_score
            print(f"  [{i}/{total}] {candidate.domain} — {result.compliance_score:.1%} "
                  f"({result.total_rules_violated} violations, {result.total_rules_passed} passed)")
        except Exception as e:
            print(f"  [{i}/{total}] {candidate.domain} — FAILED: {e}")
            candidate.compliance_score = None

        # Save progress every 10 sites
        if i % 10 == 0:
            save_candidates(candidates, args.output)
            print(f"  ... progress saved ({i}/{total})")

    save_candidates(candidates, args.output)
    save_candidates_csv(candidates, args.output.replace(".json", ".csv"))

    scored = [c for c in candidates if c.compliance_score is not None]
    print(f"\nDone: {len(scored)}/{total} candidates scored")
    if scored:
        scores = [c.compliance_score for c in scored]
        print(f"  Min: {min(scores):.1%}, Max: {max(scores):.1%}, "
              f"Mean: {sum(scores)/len(scores):.1%}")


async def cmd_sample(args: argparse.Namespace) -> None:
    """Produce a stratified sample from scored candidates."""
    candidates = load_candidates(args.input)
    scored = [c for c in candidates if c.compliance_score is not None]
    print(f"Loaded {len(scored)} scored candidates (out of {len(candidates)} total)")

    sample = stratified_sample(scored, per_sector=args.per_sector)
    save_candidates(sample, args.output)
    save_candidates_csv(sample, args.output.replace(".json", ".csv"))

    print(f"\nFinal sample: {len(sample)} sites saved to {args.output}")


async def main() -> None:
    parser = argparse.ArgumentParser(description="URL collection and sampling pipeline")
    subparsers = parser.add_subparsers(dest="command")

    # Collect
    p_collect = subparsers.add_parser("collect", help="Collect candidate URLs")
    p_collect.add_argument("--tranco-top-n", type=int, default=5000, help="Tranco top N to scan")
    p_collect.add_argument("--output", "-o", default="data/raw/candidates.json")

    # Audit
    p_audit = subparsers.add_parser("audit", help="Audit candidates for accessibility")
    p_audit.add_argument("--input", "-i", required=True, help="Candidates JSON file")
    p_audit.add_argument("--output", "-o", required=True, help="Output scored JSON file")
    p_audit.add_argument("--force", action="store_true", help="Re-audit already-scored sites")

    # Sample
    p_sample = subparsers.add_parser("sample", help="Stratified sampling from scored candidates")
    p_sample.add_argument("--input", "-i", required=True, help="Scored candidates JSON")
    p_sample.add_argument("--output", "-o", required=True, help="Output sample JSON")
    p_sample.add_argument("--per-sector", type=int, default=40, help="Sites per sector (default: 40)")

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(1)

    if args.command == "collect":
        await cmd_collect(args)
    elif args.command == "audit":
        await cmd_audit(args)
    elif args.command == "sample":
        await cmd_sample(args)


if __name__ == "__main__":
    asyncio.run(main())
