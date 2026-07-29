"""
Run the detailed WCAG audit across the full evaluation sample.

The stratified sample (final_sample.json) stores only the summary
compliance score used for sampling. This driver re-audits every site to
produce the full per-rule violation breakdown (impact, WCAG level, rule
id, counts, WAB score) required by the correlation and feature-importance
analysis, and writes it in the format expected by
analysis.correlation.load_audit_results.

The audit is refreshed here so that the accessibility measurement is
contemporaneous with the agent execution phase (both July 2026) rather
than relying on the earlier sampling-time scores.

Usage:
    python -m src.accessibility.audit_sample \
        --sample data/processed/final_sample.json \
        --out data/processed/audit_detailed.json \
        --exclude data/results/excluded_robots.json
"""

import argparse
import asyncio
import json
from pathlib import Path

from .axe_auditor import audit_url, save_results


async def main() -> None:
    parser = argparse.ArgumentParser(description="Detailed WCAG audit of the sample")
    parser.add_argument("--sample", default="data/processed/final_sample.json")
    parser.add_argument("--out", default="data/processed/audit_detailed.json")
    parser.add_argument("--exclude", default="data/results/excluded_robots.json",
                        help="JSON list of domains to skip (robots.txt exclusions)")
    parser.add_argument("--concurrent", "-c", type=int, default=3)
    args = parser.parse_args()

    sites = json.load(open(args.sample))
    excluded = set()
    if Path(args.exclude).exists():
        excluded = set(json.load(open(args.exclude)))
    sites = [s for s in sites if s["domain"] not in excluded]
    print(f"Auditing {len(sites)} sites ({len(excluded)} excluded) at concurrency {args.concurrent}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Resume support: keep already-audited URLs
    done_urls = set()
    results: list = []
    if out_path.exists():
        results = json.load(open(out_path))
        done_urls = {r["url"] for r in results}
        print(f"Resuming: {len(done_urls)} already audited")

    sem = asyncio.Semaphore(args.concurrent)
    lock = asyncio.Lock()

    async def one(site):
        if site["url"] in done_urls:
            return
        async with sem:
            try:
                res = await audit_url(site["url"])
                rec = res.to_dict()
            except Exception as e:
                rec = {"url": site["url"], "error": str(e)[:200], "summary": None}
            rec["domain"] = site["domain"]
            rec["sector"] = site["sector"]
            async with lock:
                results.append(rec)
                ok = rec.get("summary") is not None
                cs = rec["summary"]["compliance_score"] if ok else "ERR"
                print(f"[{len(results)}/{len(sites)}] {site['domain']:24s} compliance={cs}")
                # checkpoint every 10 audits
                if len(results) % 10 == 0:
                    with open(out_path, "w") as f:
                        json.dump(results, f, indent=2)

    await asyncio.gather(*(one(s) for s in sites))
    # keep only successful audits in the final file
    ok_results = [r for r in results if r.get("summary") is not None]
    with open(out_path, "w") as f:
        json.dump(ok_results, f, indent=2)
    print(f"\nDone: {len(ok_results)}/{len(sites)} audited successfully -> {out_path}")
    failed = [r["domain"] for r in results if r.get("summary") is None]
    if failed:
        print(f"Audit failures ({len(failed)}): {', '.join(failed)}")


if __name__ == "__main__":
    asyncio.run(main())
