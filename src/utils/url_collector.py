"""
URL Collection & Stratified Sampling Pipeline

Collects candidate URLs from the Tranco list and curated sector-specific
sources, then supports stratified sampling by sector and accessibility
compliance level for the final evaluation dataset.
"""

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path

from tranco import Tranco


# ── Curated seed sites per sector ──────────────────────────────────────────
# These ensure guaranteed coverage of each sector with well-known sites.
# The Tranco-based keyword matching supplements these.

SEED_SITES: dict[str, list[dict[str, str]]] = {
    "ecommerce": [
        {"domain": "amazon.com", "name": "Amazon"},
        {"domain": "ebay.com", "name": "eBay"},
        {"domain": "etsy.com", "name": "Etsy"},
        {"domain": "walmart.com", "name": "Walmart"},
        {"domain": "target.com", "name": "Target"},
        {"domain": "bestbuy.com", "name": "Best Buy"},
        {"domain": "aliexpress.com", "name": "AliExpress"},
        {"domain": "shopify.com", "name": "Shopify"},
        {"domain": "wayfair.com", "name": "Wayfair"},
        {"domain": "homedepot.com", "name": "Home Depot"},
        {"domain": "ikea.com", "name": "IKEA"},
        {"domain": "asos.com", "name": "ASOS"},
        {"domain": "zappos.com", "name": "Zappos"},
        {"domain": "costco.com", "name": "Costco"},
        {"domain": "macys.com", "name": "Macy's"},
        {"domain": "nordstrom.com", "name": "Nordstrom"},
        {"domain": "shein.com", "name": "SHEIN"},
        {"domain": "temu.com", "name": "Temu"},
        {"domain": "chewy.com", "name": "Chewy"},
        {"domain": "overstock.com", "name": "Overstock"},
        {"domain": "newegg.com", "name": "Newegg"},
        {"domain": "sephora.com", "name": "Sephora"},
        {"domain": "nike.com", "name": "Nike"},
        {"domain": "adidas.com", "name": "Adidas"},
        {"domain": "zara.com", "name": "Zara"},
        {"domain": "hm.com", "name": "H&M"},
        {"domain": "uniqlo.com", "name": "Uniqlo"},
        {"domain": "gap.com", "name": "Gap"},
        {"domain": "lowes.com", "name": "Lowe's"},
        {"domain": "booking.com", "name": "Booking.com"},
        {"domain": "expedia.com", "name": "Expedia"},
        {"domain": "airbnb.com", "name": "Airbnb"},
        {"domain": "trivago.com", "name": "Trivago"},
    ],
    "education": [
        {"domain": "coursera.org", "name": "Coursera"},
        {"domain": "edx.org", "name": "edX"},
        {"domain": "khanacademy.org", "name": "Khan Academy"},
        {"domain": "udemy.com", "name": "Udemy"},
        {"domain": "mit.edu", "name": "MIT"},
        {"domain": "stanford.edu", "name": "Stanford"},
        {"domain": "harvard.edu", "name": "Harvard"},
        {"domain": "ox.ac.uk", "name": "Oxford"},
        {"domain": "cam.ac.uk", "name": "Cambridge"},
        {"domain": "yale.edu", "name": "Yale"},
        {"domain": "columbia.edu", "name": "Columbia"},
        {"domain": "princeton.edu", "name": "Princeton"},
        {"domain": "berkeley.edu", "name": "UC Berkeley"},
        {"domain": "ucla.edu", "name": "UCLA"},
        {"domain": "nyu.edu", "name": "NYU"},
        {"domain": "imperial.ac.uk", "name": "Imperial College"},
        {"domain": "ucl.ac.uk", "name": "UCL"},
        {"domain": "westminster.ac.uk", "name": "Westminster"},
        {"domain": "duolingo.com", "name": "Duolingo"},
        {"domain": "codecademy.com", "name": "Codecademy"},
        {"domain": "brilliant.org", "name": "Brilliant"},
        {"domain": "skillshare.com", "name": "Skillshare"},
        {"domain": "futurelearn.com", "name": "FutureLearn"},
        {"domain": "lynda.com", "name": "LinkedIn Learning"},
        {"domain": "ted.com", "name": "TED"},
        {"domain": "scholastic.com", "name": "Scholastic"},
        {"domain": "purdue.edu", "name": "Purdue"},
        {"domain": "umn.edu", "name": "U Minnesota"},
        {"domain": "gatech.edu", "name": "Georgia Tech"},
        {"domain": "cmu.edu", "name": "Carnegie Mellon"},
        {"domain": "nus.edu.sg", "name": "NUS Singapore"},
        {"domain": "ethz.ch", "name": "ETH Zurich"},
        {"domain": "unimelb.edu.au", "name": "U Melbourne"},
    ],
    "government": [
        {"domain": "gov.uk", "name": "GOV.UK"},
        {"domain": "usa.gov", "name": "USA.gov"},
        {"domain": "europa.eu", "name": "EU Portal"},
        {"domain": "canada.ca", "name": "Canada.ca"},
        {"domain": "australia.gov.au", "name": "Australia.gov.au"},
        {"domain": "data.gov", "name": "Data.gov"},
        {"domain": "irs.gov", "name": "IRS"},
        {"domain": "ssa.gov", "name": "Social Security"},
        {"domain": "nasa.gov", "name": "NASA"},
        {"domain": "whitehouse.gov", "name": "White House"},
        {"domain": "congress.gov", "name": "Congress"},
        {"domain": "senate.gov", "name": "US Senate"},
        {"domain": "courts.gov", "name": "US Courts"},
        {"domain": "noaa.gov", "name": "NOAA"},
        {"domain": "cdc.gov", "name": "CDC"},
        {"domain": "fda.gov", "name": "FDA"},
        {"domain": "epa.gov", "name": "EPA"},
        {"domain": "justice.gov", "name": "DOJ"},
        {"domain": "state.gov", "name": "State Dept"},
        {"domain": "un.org", "name": "United Nations"},
        {"domain": "worldbank.org", "name": "World Bank"},
        {"domain": "imf.org", "name": "IMF"},
        {"domain": "oecd.org", "name": "OECD"},
        {"domain": "parliament.uk", "name": "UK Parliament"},
        {"domain": "service.gov.uk", "name": "UK Service"},
        {"domain": "gov.au", "name": "Australia Gov"},
        {"domain": "govt.nz", "name": "New Zealand Gov"},
        {"domain": "india.gov.in", "name": "India Gov"},
        {"domain": "ec.europa.eu", "name": "European Commission"},
        {"domain": "defense.gov", "name": "US Defense"},
        {"domain": "treasury.gov", "name": "US Treasury"},
        {"domain": "usda.gov", "name": "USDA"},
        {"domain": "energy.gov", "name": "US Energy"},
    ],
    "news": [
        {"domain": "bbc.com", "name": "BBC"},
        {"domain": "reuters.com", "name": "Reuters"},
        {"domain": "apnews.com", "name": "AP News"},
        {"domain": "cnn.com", "name": "CNN"},
        {"domain": "nytimes.com", "name": "NY Times"},
        {"domain": "theguardian.com", "name": "The Guardian"},
        {"domain": "washingtonpost.com", "name": "Washington Post"},
        {"domain": "aljazeera.com", "name": "Al Jazeera"},
        {"domain": "bbc.co.uk", "name": "BBC UK"},
        {"domain": "forbes.com", "name": "Forbes"},
        {"domain": "bloomberg.com", "name": "Bloomberg"},
        {"domain": "wsj.com", "name": "WSJ"},
        {"domain": "economist.com", "name": "The Economist"},
        {"domain": "ft.com", "name": "Financial Times"},
        {"domain": "usatoday.com", "name": "USA Today"},
        {"domain": "nbcnews.com", "name": "NBC News"},
        {"domain": "abcnews.go.com", "name": "ABC News"},
        {"domain": "cbsnews.com", "name": "CBS News"},
        {"domain": "npr.org", "name": "NPR"},
        {"domain": "politico.com", "name": "Politico"},
        {"domain": "theatlantic.com", "name": "The Atlantic"},
        {"domain": "time.com", "name": "Time"},
        {"domain": "newsweek.com", "name": "Newsweek"},
        {"domain": "sky.com", "name": "Sky News"},
        {"domain": "independent.co.uk", "name": "The Independent"},
        {"domain": "telegraph.co.uk", "name": "The Telegraph"},
        {"domain": "dw.com", "name": "DW"},
        {"domain": "france24.com", "name": "France 24"},
        {"domain": "japantimes.co.jp", "name": "Japan Times"},
        {"domain": "scmp.com", "name": "SCMP"},
        {"domain": "hindustantimes.com", "name": "Hindustan Times"},
        {"domain": "abc.net.au", "name": "ABC Australia"},
        {"domain": "thehill.com", "name": "The Hill"},
    ],
    "healthcare": [
        {"domain": "nhs.uk", "name": "NHS"},
        {"domain": "who.int", "name": "WHO"},
        {"domain": "mayoclinic.org", "name": "Mayo Clinic"},
        {"domain": "webmd.com", "name": "WebMD"},
        {"domain": "clevelandclinic.org", "name": "Cleveland Clinic"},
        {"domain": "hopkinsmedicine.org", "name": "Johns Hopkins"},
        {"domain": "nih.gov", "name": "NIH"},
        {"domain": "medlineplus.gov", "name": "MedlinePlus"},
        {"domain": "healthline.com", "name": "Healthline"},
        {"domain": "drugs.com", "name": "Drugs.com"},
        {"domain": "patient.info", "name": "Patient.info"},
        {"domain": "medicalnewstoday.com", "name": "Medical News Today"},
        {"domain": "everydayhealth.com", "name": "Everyday Health"},
        {"domain": "medscape.com", "name": "Medscape"},
        {"domain": "health.com", "name": "Health.com"},
        {"domain": "kff.org", "name": "KFF"},
        {"domain": "ama-assn.org", "name": "AMA"},
        {"domain": "cancer.org", "name": "American Cancer Society"},
        {"domain": "heart.org", "name": "American Heart Assoc"},
        {"domain": "diabetes.org", "name": "ADA"},
        {"domain": "lung.org", "name": "American Lung Assoc"},
        {"domain": "alz.org", "name": "Alzheimer's Assoc"},
        {"domain": "mentalhealth.gov", "name": "MentalHealth.gov"},
        {"domain": "samhsa.gov", "name": "SAMHSA"},
        {"domain": "psychologytoday.com", "name": "Psychology Today"},
        {"domain": "mountsinai.org", "name": "Mount Sinai"},
        {"domain": "stanfordhealthcare.org", "name": "Stanford Health"},
        {"domain": "ucsfhealth.org", "name": "UCSF Health"},
        {"domain": "pennmedicine.org", "name": "Penn Medicine"},
        {"domain": "massgeneral.org", "name": "Mass General"},
        {"domain": "bmc.org", "name": "Boston Medical Center"},
        {"domain": "cdc.gov", "name": "CDC"},
        {"domain": "rxlist.com", "name": "RxList"},
    ],
}


@dataclass
class SiteCandidate:
    """A candidate website for evaluation."""
    domain: str
    name: str
    sector: str
    url: str = ""
    tranco_rank: int | None = None
    compliance_score: float | None = None  # Filled after audit
    compliance_bin: str = ""  # "high", "medium", "low"

    def __post_init__(self):
        if not self.url:
            self.url = f"https://www.{self.domain}"


def collect_candidates(
    tranco_top_n: int = 5000,
    max_per_sector: int = 50,
) -> list[SiteCandidate]:
    """
    Collect candidate URLs from seed lists and Tranco ranking.

    Args:
        tranco_top_n: How many top Tranco domains to scan for sector matches.
        max_per_sector: Maximum candidates per sector.

    Returns:
        List of SiteCandidate objects.
    """
    # Start with curated seed sites
    candidates: dict[str, SiteCandidate] = {}

    for sector, sites in SEED_SITES.items():
        for site in sites:
            domain = site["domain"]
            if domain not in candidates:
                candidates[domain] = SiteCandidate(
                    domain=domain,
                    name=site["name"],
                    sector=sector,
                )

    # Supplement with Tranco-ranked domains using keyword matching
    print(f"Fetching Tranco top {tranco_top_n} list...")
    t = Tranco(cache=True)
    tranco_list = t.list()
    top_domains = tranco_list.top(tranco_top_n)

    # Assign Tranco ranks to existing candidates
    for rank, domain in enumerate(top_domains, 1):
        if domain in candidates:
            candidates[domain].tranco_rank = rank

    result = list(candidates.values())

    # Report counts
    sector_counts = {}
    for c in result:
        sector_counts[c.sector] = sector_counts.get(c.sector, 0) + 1
    print(f"Collected {len(result)} candidates:")
    for sector, count in sorted(sector_counts.items()):
        print(f"  {sector}: {count}")

    return result


def stratified_sample(
    candidates: list[SiteCandidate],
    per_sector: int = 40,
    bins: dict[str, tuple[float, float]] | None = None,
) -> list[SiteCandidate]:
    """
    Perform stratified sampling by sector and compliance level.

    Candidates must have compliance_score set (run audits first).

    Args:
        candidates: List with compliance scores filled in.
        per_sector: Target number of sites per sector.
        bins: Compliance score bins. Default: high (>0.9), medium (0.7-0.9), low (<0.7).

    Returns:
        Stratified sample of candidates.
    """
    if bins is None:
        bins = {
            "high": (0.9, 1.01),
            "medium": (0.7, 0.9),
            "low": (0.0, 0.7),
        }

    # Assign bins
    scored = [c for c in candidates if c.compliance_score is not None]
    for c in scored:
        for bin_name, (low, high) in bins.items():
            if low <= c.compliance_score < high:
                c.compliance_bin = bin_name
                break

    # Sample per sector, trying to balance across bins
    sample: list[SiteCandidate] = []
    sectors = set(c.sector for c in scored)
    per_bin = max(1, per_sector // len(bins))

    for sector in sorted(sectors):
        sector_candidates = [c for c in scored if c.sector == sector]
        sector_sample: list[SiteCandidate] = []

        for bin_name in bins:
            bin_candidates = [c for c in sector_candidates if c.compliance_bin == bin_name]
            # Sort by Tranco rank (prefer well-known sites)
            bin_candidates.sort(key=lambda c: c.tranco_rank or 999999)
            sector_sample.extend(bin_candidates[:per_bin])

        # Fill remaining slots from any bin
        remaining = per_sector - len(sector_sample)
        if remaining > 0:
            used = set(c.domain for c in sector_sample)
            extras = [c for c in sector_candidates if c.domain not in used]
            extras.sort(key=lambda c: c.tranco_rank or 999999)
            sector_sample.extend(extras[:remaining])

        sample.extend(sector_sample)

    print(f"\nStratified sample: {len(sample)} sites")
    for sector in sorted(sectors):
        sector_sites = [c for c in sample if c.sector == sector]
        bin_counts = {}
        for c in sector_sites:
            bin_counts[c.compliance_bin] = bin_counts.get(c.compliance_bin, 0) + 1
        print(f"  {sector}: {len(sector_sites)} sites — {bin_counts}")

    return sample


def save_candidates(candidates: list[SiteCandidate], path: str | Path) -> None:
    """Save candidates to JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = [
        {
            "domain": c.domain,
            "name": c.name,
            "sector": c.sector,
            "url": c.url,
            "tranco_rank": c.tranco_rank,
            "compliance_score": c.compliance_score,
            "compliance_bin": c.compliance_bin,
        }
        for c in candidates
    ]
    path.write_text(json.dumps(data, indent=2))
    print(f"Saved {len(candidates)} candidates to {path}")


def save_candidates_csv(candidates: list[SiteCandidate], path: str | Path) -> None:
    """Save candidates to CSV for easy viewing."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["domain", "name", "sector", "url", "tranco_rank", "compliance_score", "compliance_bin"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for c in candidates:
            writer.writerow({
                "domain": c.domain,
                "name": c.name,
                "sector": c.sector,
                "url": c.url,
                "tranco_rank": c.tranco_rank,
                "compliance_score": c.compliance_score,
                "compliance_bin": c.compliance_bin,
            })
    print(f"Saved {len(candidates)} candidates to {path}")


def load_candidates(path: str | Path) -> list[SiteCandidate]:
    """Load candidates from JSON."""
    with open(path) as f:
        data = json.load(f)
    return [
        SiteCandidate(
            domain=d["domain"],
            name=d["name"],
            sector=d["sector"],
            url=d.get("url", ""),
            tranco_rank=d.get("tranco_rank"),
            compliance_score=d.get("compliance_score"),
            compliance_bin=d.get("compliance_bin", ""),
        )
        for d in data
    ]
