"""
CLI script to run accessibility audits on one or more URLs.

Usage:
    python -m src.accessibility.run_audit https://www.bbc.com https://www.gov.uk
    python -m src.accessibility.run_audit --file configs/sample_sites.yaml --output data/raw/audit_results.json
"""

import argparse
import asyncio
import sys
from pathlib import Path

from .axe_auditor import AuditResult, audit_url, audit_urls, save_results


def print_summary(result: AuditResult) -> None:
    """Print a human-readable summary of an audit result."""
    print(f"\n{'='*70}")
    print(f"URL: {result.url}")
    print(f"Timestamp: {result.timestamp}")
    print(f"{'='*70}")
    print(f"  Rules passed:   {result.total_rules_passed}")
    print(f"  Rules violated: {result.total_rules_violated}")
    print(f"  Total issues:   {result.total_violations}")
    print(f"  Compliance:     {result.compliance_score:.1%}")
    print(f"  WAB score:      {result.wab_score():.4f}")

    by_impact = result.violations_by_impact()
    if by_impact:
        print(f"\n  Violations by impact:")
        for impact in ["critical", "serious", "moderate", "minor"]:
            if impact in by_impact:
                count = sum(v.count for v in by_impact[impact])
                print(f"    {impact:>10}: {count} instances across {len(by_impact[impact])} rules")

    by_level = result.violations_by_level()
    if by_level:
        print(f"\n  Violations by WCAG level:")
        for level in ["A", "AA", "AAA"]:
            if level in by_level:
                count = sum(v.count for v in by_level[level])
                print(f"    Level {level:>3}: {count} instances across {len(by_level[level])} rules")

    if result.violations:
        print(f"\n  Top violations:")
        sorted_v = sorted(result.violations, key=lambda v: v.count, reverse=True)
        for v in sorted_v[:5]:
            print(f"    [{v.impact}] {v.rule_id}: {v.help_text} ({v.count} instances)")


def load_urls_from_yaml(path: str) -> list[str]:
    """Load URLs from a sample_sites.yaml config file."""
    import yaml

    with open(path) as f:
        data = yaml.safe_load(f)

    urls = []
    for sector, sites in data.items():
        if isinstance(sites, list):
            for site in sites:
                if isinstance(site, dict) and "url" in site:
                    urls.append(site["url"])
    return urls


async def main() -> None:
    parser = argparse.ArgumentParser(description="Run WCAG accessibility audits")
    parser.add_argument("urls", nargs="*", help="URLs to audit")
    parser.add_argument("--file", "-f", help="YAML file with URLs to audit")
    parser.add_argument("--output", "-o", help="Output JSON file path")
    parser.add_argument("--concurrent", "-c", type=int, default=3, help="Max concurrent audits")
    args = parser.parse_args()

    urls: list[str] = list(args.urls)
    if args.file:
        urls.extend(load_urls_from_yaml(args.file))

    if not urls:
        print("No URLs provided. Pass URLs as arguments or use --file.")
        sys.exit(1)

    print(f"Auditing {len(urls)} URL(s)...")
    results = await audit_urls(urls, max_concurrent=args.concurrent)

    for result in results:
        print_summary(result)

    if args.output:
        save_results(results, args.output)

    print(f"\n{'='*70}")
    print(f"Completed: {len(results)}/{len(urls)} URLs audited successfully")


if __name__ == "__main__":
    asyncio.run(main())
