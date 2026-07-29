"""
Accessibility Evaluation Module - axe-core Auditor

Runs axe-core accessibility audits on web pages using Playwright,
producing per-criterion pass/fail results and WCAG compliance scores.
"""

import asyncio
import json
from dataclasses import dataclass, field
from pathlib import Path

from playwright.async_api import async_playwright


# axe-core CDN URL — pinned version for reproducibility
AXE_CORE_URL = "https://cdnjs.cloudflare.com/ajax/libs/axe-core/4.10.2/axe.min.js"

# Path to local axe-core bundle (downloaded on first use, used as fallback for CSP-restricted sites)
AXE_CORE_LOCAL = Path(__file__).parent / "vendor" / "axe.min.js"

# WCAG tag mapping: axe-core tags -> conformance level
WCAG_LEVEL_MAP = {
    "wcag2a": "A",
    "wcag2aa": "AA",
    "wcag2aaa": "AAA",
    "wcag21a": "A",
    "wcag21aa": "AA",
    "wcag22aa": "AA",
}


@dataclass
class ViolationInstance:
    """A single violation instance (one element failing one rule)."""
    html: str
    target: list[str]
    failure_summary: str


@dataclass
class AuditViolation:
    """A single axe-core rule violation with all affected elements."""
    rule_id: str
    description: str
    help_text: str
    help_url: str
    impact: str  # minor, moderate, serious, critical
    wcag_tags: list[str]
    wcag_level: str
    instances: list[ViolationInstance]

    @property
    def count(self) -> int:
        return len(self.instances)


@dataclass
class AuditResult:
    """Complete audit result for a single URL."""
    url: str
    timestamp: str
    violations: list[AuditViolation] = field(default_factory=list)
    passes: list[dict] = field(default_factory=list)
    incomplete: list[dict] = field(default_factory=list)
    inapplicable: list[dict] = field(default_factory=list)

    @property
    def total_violations(self) -> int:
        return sum(v.count for v in self.violations)

    @property
    def total_rules_violated(self) -> int:
        return len(self.violations)

    @property
    def total_rules_passed(self) -> int:
        return len(self.passes)

    @property
    def violation_rate(self) -> float:
        """Simple failure rate: violated rules / (violated + passed rules)."""
        total = self.total_rules_violated + self.total_rules_passed
        if total == 0:
            return 0.0
        return self.total_rules_violated / total

    @property
    def compliance_score(self) -> float:
        """Compliance score: 1 - violation_rate (higher is better)."""
        return 1.0 - self.violation_rate

    def violations_by_impact(self) -> dict[str, list[AuditViolation]]:
        """Group violations by impact level."""
        groups: dict[str, list[AuditViolation]] = {}
        for v in self.violations:
            groups.setdefault(v.impact, []).append(v)
        return groups

    def violations_by_level(self) -> dict[str, list[AuditViolation]]:
        """Group violations by WCAG conformance level."""
        groups: dict[str, list[AuditViolation]] = {}
        for v in self.violations:
            groups.setdefault(v.wcag_level, []).append(v)
        return groups

    def wab_score(self) -> float:
        """
        Web Accessibility Barrier (WAB) score (Parmanto & Zeng, 2005).
        WAB = sum(violations_per_criterion / total_pages) / total_criteria
        For single-page evaluation: sum(violation_instances) / total_criteria_tested
        Lower is better; 0 = no barriers detected.
        """
        total_criteria = self.total_rules_violated + self.total_rules_passed
        if total_criteria == 0:
            return 0.0
        return self.total_violations / total_criteria

    def to_dict(self) -> dict:
        """Serialize to dictionary for JSON export."""
        return {
            "url": self.url,
            "timestamp": self.timestamp,
            "summary": {
                "total_violations": self.total_violations,
                "total_rules_violated": self.total_rules_violated,
                "total_rules_passed": self.total_rules_passed,
                "violation_rate": round(self.violation_rate, 4),
                "compliance_score": round(self.compliance_score, 4),
                "wab_score": round(self.wab_score(), 4),
            },
            "violations": [
                {
                    "rule_id": v.rule_id,
                    "description": v.description,
                    "impact": v.impact,
                    "wcag_level": v.wcag_level,
                    "wcag_tags": v.wcag_tags,
                    "count": v.count,
                }
                for v in self.violations
            ],
            "passes_count": self.total_rules_passed,
            "incomplete_count": len(self.incomplete),
            "inapplicable_count": len(self.inapplicable),
        }


def _extract_wcag_level(tags: list[str]) -> str:
    """Extract the highest WCAG conformance level from axe-core tags."""
    levels = set()
    for tag in tags:
        if tag in WCAG_LEVEL_MAP:
            levels.add(WCAG_LEVEL_MAP[tag])
    # Return highest level found
    for level in ["AAA", "AA", "A"]:
        if level in levels:
            return level
    return "unknown"


def _parse_violations(raw_violations: list[dict]) -> list[AuditViolation]:
    """Parse raw axe-core violation objects into AuditViolation dataclasses."""
    violations = []
    for v in raw_violations:
        wcag_tags = [t for t in v.get("tags", []) if t.startswith("wcag")]
        instances = [
            ViolationInstance(
                html=node.get("html", ""),
                target=node.get("target", []),
                failure_summary=node.get("failureSummary", ""),
            )
            for node in v.get("nodes", [])
        ]
        violations.append(
            AuditViolation(
                rule_id=v["id"],
                description=v.get("description", ""),
                help_text=v.get("help", ""),
                help_url=v.get("helpUrl", ""),
                impact=v.get("impact", "unknown"),
                wcag_tags=wcag_tags,
                wcag_level=_extract_wcag_level(v.get("tags", [])),
                instances=instances,
            )
        )
    return violations


async def _get_axe_source() -> str:
    """Download and cache axe-core source for inline injection (CSP fallback)."""
    if not AXE_CORE_LOCAL.exists():
        import httpx
        AXE_CORE_LOCAL.parent.mkdir(parents=True, exist_ok=True)
        async with httpx.AsyncClient() as client:
            resp = await client.get(AXE_CORE_URL)
            resp.raise_for_status()
            AXE_CORE_LOCAL.write_text(resp.text)
    return AXE_CORE_LOCAL.read_text()


async def audit_url(
    url: str,
    wcag_tags: list[str] | None = None,
    viewport: tuple[int, int] = (1280, 720),
    timeout: int = 30000,
) -> AuditResult:
    """
    Run an axe-core accessibility audit on a single URL.

    Args:
        url: The URL to audit.
        wcag_tags: axe-core rule tags to filter by (e.g., ["wcag2a", "wcag2aa"]).
        viewport: Browser viewport dimensions (width, height).
        timeout: Page load timeout in milliseconds.

    Returns:
        AuditResult with violations, passes, and computed scores.
    """
    from datetime import datetime, timezone

    if wcag_tags is None:
        wcag_tags = ["wcag2a", "wcag2aa", "wcag21a", "wcag21aa", "wcag22aa"]

    # Build axe-core run options
    axe_options = json.dumps({
        "runOnly": {
            "type": "tag",
            "values": wcag_tags,
        }
    })

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(
            viewport={"width": viewport[0], "height": viewport[1]},
            user_agent=(
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
        )
        page = await context.new_page()

        try:
            await page.goto(url, timeout=timeout, wait_until="domcontentloaded")
            # Wait a bit for JS rendering
            await page.wait_for_timeout(2000)

            # Inject axe-core — try CDN first, fall back to inline eval for CSP-restricted sites
            try:
                await page.add_script_tag(url=AXE_CORE_URL)
            except Exception:
                axe_script = await _get_axe_source()
                await page.evaluate(axe_script)
            # Wait for axe to be available
            await page.wait_for_function("typeof window.axe !== 'undefined'", timeout=10000)

            # Run axe-core audit
            raw_results = await page.evaluate(f"""
                async () => {{
                    const results = await axe.run(document, {axe_options});
                    return JSON.parse(JSON.stringify(results));
                }}
            """)

        finally:
            await browser.close()

    timestamp = datetime.now(timezone.utc).isoformat()
    violations = _parse_violations(raw_results.get("violations", []))

    return AuditResult(
        url=url,
        timestamp=timestamp,
        violations=violations,
        passes=raw_results.get("passes", []),
        incomplete=raw_results.get("incomplete", []),
        inapplicable=raw_results.get("inapplicable", []),
    )


async def audit_urls(
    urls: list[str],
    wcag_tags: list[str] | None = None,
    max_concurrent: int = 3,
) -> list[AuditResult]:
    """
    Audit multiple URLs with concurrency control.

    Args:
        urls: List of URLs to audit.
        wcag_tags: axe-core rule tags to filter by.
        max_concurrent: Maximum concurrent browser instances.

    Returns:
        List of AuditResult objects.
    """
    semaphore = asyncio.Semaphore(max_concurrent)
    results: list[AuditResult] = []

    async def _audit_with_limit(url: str) -> AuditResult | None:
        async with semaphore:
            try:
                return await audit_url(url, wcag_tags=wcag_tags)
            except Exception as e:
                print(f"[ERROR] Failed to audit {url}: {e}")
                return None

    tasks = [_audit_with_limit(url) for url in urls]
    raw_results = await asyncio.gather(*tasks)
    results = [r for r in raw_results if r is not None]
    return results


def save_results(results: list[AuditResult], output_path: str | Path) -> None:
    """Save audit results to a JSON file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data = [r.to_dict() for r in results]
    output_path.write_text(json.dumps(data, indent=2))
    print(f"Results saved to {output_path}")
