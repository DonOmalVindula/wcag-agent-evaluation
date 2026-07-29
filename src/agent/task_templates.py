"""
Per-site task generation from sector-based templates.

Generates comparable tasks across all sites in the evaluation sample.
Each site receives one task per category (navigation, search, info),
with sector-specific parameters so tasks remain realistic while staying
methodologically comparable across the whole sample.

Success is verified post-hoc by keyword matching against the final URL
and visible page text (see task_runner.TaskResult.verified_success),
in addition to the agent's own done(success=...) self-report.
"""

import csv
from dataclasses import dataclass
from pathlib import Path

from .task_runner import WebTask

# Sector-specific parameters for each task category.
# navigation: a page type common to virtually every site in the sector.
# search: a query term relevant to the sector.
SECTOR_PARAMS = {
    "ecommerce": {
        "nav_target": "shipping, delivery or returns information page",
        "nav_keywords": ["shipping", "delivery", "returns", "return policy"],
        "search_term": "gift",
    },
    "education": {
        "nav_target": "admissions or how-to-apply page",
        "nav_keywords": ["admission", "apply", "applying", "enrol", "enroll"],
        "search_term": "scholarship",
    },
    "government": {
        "nav_target": "contact page",
        "nav_keywords": ["contact", "get in touch", "phone", "email us"],
        "search_term": "passport",
    },
    "healthcare": {
        "nav_target": "page about patient services or medical conditions",
        "nav_keywords": ["services", "conditions", "patients", "treatment", "care"],
        "search_term": "flu",
    },
    "news": {
        "nav_target": "business or economy news section",
        "nav_keywords": ["business", "economy", "money", "markets"],
        "search_term": "climate",
    },
}

TASK_CATEGORIES = ["navigation", "search", "info"]


@dataclass
class SiteRecord:
    """One row of the evaluation sample."""
    domain: str
    name: str
    sector: str
    url: str
    compliance_score: float
    compliance_bin: str


def load_sample(path: str | Path) -> list[SiteRecord]:
    """Load the stratified site sample CSV."""
    sites = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            sites.append(SiteRecord(
                domain=row["domain"],
                name=row["name"],
                sector=row["sector"],
                url=row["url"],
                compliance_score=float(row["compliance_score"]),
                compliance_bin=row["compliance_bin"],
            ))
    return sites


def tasks_for_site(site: SiteRecord, categories: list[str] | None = None) -> list[WebTask]:
    """Generate the task set for a single site from its sector template."""
    if categories is None:
        categories = TASK_CATEGORIES
    params = SECTOR_PARAMS[site.sector]
    tasks = []

    if "navigation" in categories:
        tasks.append(WebTask(
            url=site.url,
            description=f"Navigate to this website's {params['nav_target']}.",
            success_criteria=(
                f"The final page is the site's {params['nav_target']} "
                "(check the URL and page headings)."
            ),
            max_steps=8,
            task_id=f"{site.domain}::navigation",
            category="navigation",
            success_keywords=params["nav_keywords"],
        ))

    if "search" in categories:
        term = params["search_term"]
        tasks.append(WebTask(
            url=site.url,
            description=f"Use the site's search function to search for '{term}'.",
            success_criteria=(
                f"Search results for '{term}' are displayed "
                "(the URL or page content shows a search results view)."
            ),
            max_steps=8,
            task_id=f"{site.domain}::search",
            category="search",
            success_keywords=[term, "search", "results"],
        ))

    if "info" in categories:
        tasks.append(WebTask(
            url=site.url,
            description="Find this website's privacy policy page.",
            success_criteria="The final page displays the site's privacy policy.",
            max_steps=8,
            task_id=f"{site.domain}::info",
            category="info",
            success_keywords=["privacy"],
        ))

    return tasks


def generate_all_tasks(
    sample_path: str | Path,
    categories: list[str] | None = None,
    sectors: list[str] | None = None,
    limit_sites: int | None = None,
) -> list[tuple[SiteRecord, WebTask]]:
    """
    Generate (site, task) pairs for the whole sample.

    limit_sites applies per sector after filtering, preserving the
    stratified balance for pilot runs.
    """
    sites = load_sample(sample_path)
    if sectors:
        sites = [s for s in sites if s.sector in sectors]

    if limit_sites:
        by_sector: dict[str, list[SiteRecord]] = {}
        for s in sites:
            by_sector.setdefault(s.sector, []).append(s)
        per_sector = max(1, limit_sites // max(len(by_sector), 1))
        sites = [s for group in by_sector.values() for s in group[:per_sector]]

    pairs = []
    for site in sites:
        for task in tasks_for_site(site, categories):
            pairs.append((site, task))
    return pairs
