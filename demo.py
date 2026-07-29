"""
Interim Submission Demo Script
Demonstrates the dual-evaluation framework's core capabilities.

Usage:
    cd framework && source .venv/bin/activate
    python demo.py
"""

import asyncio
import csv
from pathlib import Path

from playwright.async_api import async_playwright
from src.accessibility.axe_auditor import audit_url
from src.agent.browser_env import extract_accessibility_tree


def separator(title: str) -> None:
    print(f"\n{'='*70}")
    print(f"  DEMO: {title}")
    print(f"{'='*70}\n")


def pause(msg: str = "Press Enter to continue to the next demo...") -> None:
    input(f"\n  >> {msg}")


async def demo_1_audit() -> None:
    """Demo 1: Live WCAG accessibility audit on contrasting sites."""
    separator("Module 1 — Live WCAG Accessibility Audit")
    print("Auditing two sites with different compliance levels...\n")

    sites = [
        ("https://www.gov.uk", "GOV.UK (government, expected high compliance)"),
        ("https://www.amazon.com", "Amazon (e-commerce, expected lower compliance)"),
    ]

    for url, desc in sites:
        print(f"  Auditing: {desc}")
        print(f"  URL: {url}")
        try:
            result = await audit_url(url, timeout=20000)
            print(f"  Compliance Score: {result.compliance_score:.1%}")
            print(f"  WAB Score:        {result.wab_score():.4f}")
            print(f"  Rules Passed:     {result.total_rules_passed}")
            print(f"  Rules Violated:   {result.total_rules_violated}")
            print(f"  Total Issues:     {result.total_violations}")

            by_impact = result.violations_by_impact()
            if by_impact:
                print(f"  Violations by impact:")
                for impact in ["critical", "serious", "moderate", "minor"]:
                    if impact in by_impact:
                        count = sum(v.count for v in by_impact[impact])
                        print(f"    {impact:>10}: {count} instances")

            if result.violations:
                print(f"  Top violations:")
                for v in sorted(result.violations, key=lambda v: v.count, reverse=True)[:3]:
                    print(f"    [{v.impact}] {v.rule_id}: {v.help_text} ({v.count} instances)")
        except Exception as e:
            print(f"  ERROR: {e}")
        print()


def demo_2_dataset() -> None:
    """Demo 2: Show the 200-site stratified dataset."""
    separator("Dataset — 200 Audited Sites Across 5 Sectors")

    csv_path = Path("data/processed/final_sample.csv")
    if not csv_path.exists():
        print("  Dataset file not found. Run the audit pipeline first.")
        return

    with open(csv_path) as f:
        reader = list(csv.DictReader(f))

    total = len(reader)
    sectors = {}
    scores = []
    for row in reader:
        sector = row["sector"]
        score = float(row["compliance_score"]) if row["compliance_score"] else None
        if score is not None:
            sectors.setdefault(sector, []).append(score)
            scores.append(score)

    print(f"  Total sites: {total}")
    print(f"  Score range: {min(scores):.1%} — {max(scores):.1%}")
    print(f"  Mean score:  {sum(scores)/len(scores):.1%}")
    print()
    print(f"  {'Sector':<15} {'Count':>6} {'Mean':>8} {'Min':>8} {'Max':>8}")
    print(f"  {'-'*15} {'-'*6} {'-'*8} {'-'*8} {'-'*8}")
    for sector in ["ecommerce", "education", "government", "news", "healthcare"]:
        if sector in sectors:
            s = sectors[sector]
            print(f"  {sector:<15} {len(s):>6} {sum(s)/len(s):>7.1%} {min(s):>7.1%} {max(s):>7.1%}")

    # Show top 5 and bottom 5
    scored_rows = [(r["name"], r["sector"], float(r["compliance_score"]))
                   for r in reader if r["compliance_score"]]
    scored_rows.sort(key=lambda x: x[2])

    print(f"\n  Lowest compliance:")
    for name, sector, score in scored_rows[:5]:
        print(f"    {score:>6.1%}  {name:<25} [{sector}]")

    print(f"\n  Highest compliance (sample):")
    perfect = [r for r in scored_rows if r[2] >= 1.0]
    for name, sector, score in perfect[:5]:
        print(f"    {score:>6.1%}  {name:<25} [{sector}]")


async def demo_3_axtree() -> None:
    """Demo 3: Show the Accessibility Tree extraction."""
    separator("Module 2 — Accessibility Tree Observation")
    print("Extracting the browser Accessibility Tree from BBC.com...")
    print("This is what both screen readers AND AI agents consume.\n")

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page(viewport={"width": 1280, "height": 720})
        await page.goto("https://www.bbc.com", wait_until="domcontentloaded")
        await page.wait_for_timeout(2000)

        tree_text = await extract_accessibility_tree(page)
        lines = tree_text.split("\n")

        print(f"  Total nodes: {len(lines)}")
        print(f"  Token estimate: ~{len(tree_text) // 4} tokens")
        print()

        # Get raw DOM size for comparison
        dom_size = await page.evaluate("() => document.documentElement.outerHTML.length")
        print(f"  Raw DOM size: ~{dom_size:,} chars (~{dom_size // 4:,} tokens)")
        print(f"  AXTree size:  ~{len(tree_text):,} chars (~{len(tree_text) // 4:,} tokens)")
        print(f"  Compression:  {dom_size / max(len(tree_text), 1):.1f}x smaller")
        print()

        print("  Accessibility Tree (first 25 nodes):")
        print("  " + "-" * 60)
        for line in lines[:25]:
            print(f"  {line}")
        print("  ...")

        await browser.close()


async def main() -> None:
    print("\n" + "=" * 70)
    print("  WCAG-AGENT DUAL-EVALUATION FRAMEWORK")
    print("  Interim Submission Demo")
    print("  Omal Wijegunawardana — W2053390")
    print("=" * 70)

    # Demo 1
    await demo_1_audit()
    pause()

    # Demo 2
    demo_2_dataset()
    pause()

    # Demo 3
    await demo_3_axtree()

    separator("Demo Complete")
    print("  Framework modules demonstrated:")
    print("    1. Accessibility Evaluation — live WCAG auditing with axe-core")
    print("    2. Dataset Collection — 200 sites, 5 sectors, stratified sampling")
    print("    3. Observation Extraction — Accessibility Tree vs DOM comparison")
    print()
    print("  Next steps: Agent task execution + correlation analysis")
    print()


if __name__ == "__main__":
    asyncio.run(main())
