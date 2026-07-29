"""
Master analysis pipeline for the dual-evaluation study.

Consumes the agent execution results (agent_runs.jsonl) and the detailed
accessibility audit (audit_detailed.json), then produces every quantitative
output required for the Results chapter:

  1. Per-run and per-site merged dataset (CSV).
  2. Overall success rates and descriptive statistics by sector, mode,
     task category, and compliance band.
  3. Spearman correlations between accessibility metrics and agent
     performance, with Bonferroni correction for multiple comparisons.
  4. Per-observation-mode correlations (Sub-RQ3: is the accessibility ->
     performance link stronger for the Accessibility Tree mode?).
  5. Feature importance: which WCAG rules most predict agent success
     (basis for the proposed Agent Accessibility Metric).
  6. Publication-ready figures (scatter + fit, per-mode bars, success-by-
     band, feature-importance ranking).

Outputs land in data/analysis/ (tables as CSV, figures as PNG/PDF).

Usage:
    python -m src.analysis.run_analysis
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from .correlation import (
    feature_importance_analysis,
    load_agent_results,
    load_audit_results,
    run_correlation_analysis,
    spearman_correlation,
)

RESULTS = "data/results/agent_runs.jsonl"
AUDIT = "data/processed/audit_detailed.json"
OUTDIR = Path("data/analysis")

# Accessibility (independent) and performance (dependent) columns of interest
ACCESS_COLS = ["compliance_score", "violation_rate", "wab_score", "total_violations"]
PERF_COLS = ["success_rate", "avg_steps", "avg_duration"]


def load_agent_jsonl(path: str) -> pd.DataFrame:
    """Load agent_runs.jsonl, keeping only the latest record per run-key and
    dropping infrastructure errors (network/API failures are not task outcomes)."""
    latest = {}
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            latest[(r["task_id"], r["observation_mode"], r["run_index"])] = r
    recs = [r for r in latest.values() if r["status"] != "error"]
    rows = []
    for r in recs:
        rows.append({
            "url": r["url"],
            "domain": r.get("domain", ""),
            "sector": r.get("sector", ""),
            "task_id": r["task_id"],
            "category": r.get("category", ""),
            "observation_mode": r["observation_mode"],
            # Success is the AND of the agent's own report and independent
            # verification, so a claimed success that cannot be verified does
            # not count (conservative, avoids inflating the headline metric).
            "success": 1 if (r["status"] == "success" and r.get("verified_success")) else 0,
            "verified_success": 1 if r.get("verified_success") else 0,
            "self_reported_success": 1 if r["status"] == "success" else 0,
            "total_steps": r["total_steps"],
            "duration_seconds": r["duration_seconds"],
            "compliance_bin": r.get("compliance_bin", ""),
        })
    return pd.DataFrame(rows)


def merge_with_audit(agent_df: pd.DataFrame, audit_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate agent results per (url, mode) and join detailed audit by url."""
    agg = (
        agent_df.groupby(["url", "observation_mode"])
        .agg(
            success_rate=("success", "mean"),
            avg_steps=("total_steps", "mean"),
            avg_duration=("duration_seconds", "mean"),
            n_tasks=("success", "count"),
            n_success=("success", "sum"),
        )
        .reset_index()
    )
    merged = agg.merge(audit_df, on="url", how="inner")
    return merged


def _bonferroni(results, n_tests):
    """Apply Bonferroni correction: significant iff p < alpha/n_tests."""
    alpha = 0.05
    out = []
    for c in results:
        d = c.to_dict()
        d["bonferroni_significant"] = bool(c.p_value < (alpha / n_tests))
        out.append(d)
    return out


def make_figures(agent_df, merged, importance, outdir):
    """Generate all Results-chapter figures. Matplotlib only (no seaborn dep)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"figure.dpi": 150, "font.size": 10, "savefig.bbox": "tight"})

    def save(fig, name):
        fig.savefig(outdir / f"{name}.png")
        fig.savefig(outdir / f"{name}.pdf")
        plt.close(fig)

    # Fig 1: compliance vs success rate, per mode, with Spearman fit line
    fig, ax = plt.subplots(figsize=(7, 5))
    for mode in ["accessibility_tree", "dom", "screenshot"]:
        sub = merged[merged["observation_mode"] == mode]
        if sub.empty:
            continue
        ax.scatter(sub["compliance_score"], sub["success_rate"], alpha=0.5, label=mode, s=25)
    ax.set_xlabel("WCAG compliance score")
    ax.set_ylabel("Agent task success rate")
    ax.set_title("Accessibility compliance vs. agent success, by observation mode")
    ax.legend()
    save(fig, "fig_compliance_vs_success")

    # Fig 2: mean success rate by observation mode
    fig, ax = plt.subplots(figsize=(6, 4))
    ms = agent_df.groupby("observation_mode")["success"].mean().reindex(
        ["accessibility_tree", "dom", "screenshot"]).dropna()
    ax.bar(range(len(ms)), ms.values)
    ax.set_xticks(range(len(ms)))
    ax.set_xticklabels(ms.index, rotation=15)
    ax.set_ylabel("Mean success rate")
    ax.set_title("Agent success rate by observation mode")
    save(fig, "fig_success_by_mode")

    # Fig 3: success rate by compliance band
    fig, ax = plt.subplots(figsize=(6, 4))
    order = ["low", "medium", "high"]
    band = agent_df[agent_df["compliance_bin"].isin(order)]
    if not band.empty:
        bs = band.groupby("compliance_bin")["success"].mean().reindex(order).dropna()
        ax.bar(range(len(bs)), bs.values)
        ax.set_xticks(range(len(bs)))
        ax.set_xticklabels(bs.index)
        ax.set_ylabel("Mean success rate")
        ax.set_title("Agent success rate by WCAG compliance band")
        save(fig, "fig_success_by_band")

    # Fig 4: top feature importances
    if importance is not None and not importance.empty:
        fig, ax = plt.subplots(figsize=(7, 5))
        top = importance.head(12).iloc[::-1]
        ax.barh(range(len(top)), top["importance_mean"].values,
                xerr=top["importance_std"].values)
        ax.set_yticks(range(len(top)))
        ax.set_yticklabels(top["feature"].values, fontsize=8)
        ax.set_xlabel("Permutation importance")
        ax.set_title("WCAG features most predictive of agent success")
        save(fig, "fig_feature_importance")


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    agent_df = load_agent_jsonl(RESULTS)
    print(f"Loaded {len(agent_df)} valid agent runs across "
          f"{agent_df['url'].nunique()} sites, {agent_df['observation_mode'].nunique()} modes")

    # --- descriptive stats ---
    desc = {
        "overall_success_rate": float(agent_df["success"].mean()),
        "n_runs": int(len(agent_df)),
        "n_sites": int(agent_df["url"].nunique()),
        "self_vs_verified_agreement": float(
            (agent_df["self_reported_success"] == agent_df["verified_success"]).mean()),
    }
    agent_df.groupby("observation_mode")["success"].mean().to_csv(OUTDIR / "success_by_mode.csv")
    agent_df.groupby("sector")["success"].mean().to_csv(OUTDIR / "success_by_sector.csv")
    agent_df.groupby("category")["success"].mean().to_csv(OUTDIR / "success_by_category.csv")

    # --- merge with detailed audit (if present) ---
    audit_path = Path(AUDIT)
    if not audit_path.exists():
        print(f"\nNOTE: {AUDIT} not found — running agent-only descriptives. "
              "Run src.accessibility.audit_sample first for correlations.")
        (OUTDIR / "descriptives.json").write_text(json.dumps(desc, indent=2))
        agent_df.to_csv(OUTDIR / "agent_runs_clean.csv", index=False)
        print(json.dumps(desc, indent=2))
        return

    audit_df = load_audit_results(AUDIT)
    merged = merge_with_audit(agent_df, audit_df)
    merged.to_csv(OUTDIR / "merged_site_mode.csv", index=False)
    print(f"Merged dataset: {len(merged)} (site x mode) rows")

    # --- overall Spearman correlations (Bonferroni-corrected) ---
    corrs = run_correlation_analysis(merged, ACCESS_COLS, PERF_COLS)
    n_tests = len(corrs)
    corr_rows = _bonferroni(corrs, n_tests)
    pd.DataFrame(corr_rows).to_csv(OUTDIR / "correlations_overall.csv", index=False)

    # --- per-mode correlations (Sub-RQ3) ---
    per_mode = []
    for mode in ["accessibility_tree", "dom", "screenshot"]:
        sub = merged[merged["observation_mode"] == mode]
        if len(sub) < 3:
            continue
        for xc in ACCESS_COLS:
            for yc in PERF_COLS:
                if xc in sub and yc in sub:
                    r = spearman_correlation(sub, xc, yc)
                    d = r.to_dict()
                    d["observation_mode"] = mode
                    per_mode.append(d)
    pd.DataFrame(per_mode).to_csv(OUTDIR / "correlations_by_mode.csv", index=False)

    # --- feature importance (AAM basis), on accessibility_tree mode ---
    axtree = merged[merged["observation_mode"] == "accessibility_tree"]
    importance = None
    if len(axtree) >= 20:
        try:
            importance = feature_importance_analysis(axtree, target_col="success_rate")
            importance.to_csv(OUTDIR / "feature_importance.csv", index=False)
        except Exception as e:
            print(f"Feature importance skipped: {e}")

    # --- headline correlation for the abstract ---
    head = spearman_correlation(
        merged[merged["observation_mode"] == "accessibility_tree"],
        "compliance_score", "success_rate")
    desc["headline_axtree_compliance_vs_success"] = head.to_dict()
    (OUTDIR / "descriptives.json").write_text(json.dumps(desc, indent=2))

    make_figures(agent_df, merged, importance, OUTDIR)

    # --- console summary ---
    print("\n=== HEADLINE (Accessibility Tree mode) ===")
    print(f"compliance_score vs success_rate: rho={head.coefficient:+.3f} "
          f"p={head.p_value:.4g} n={head.n} significant={head.significant}")
    print(f"\nOverall success rate: {desc['overall_success_rate']:.1%}")
    print("Success by mode:")
    print(agent_df.groupby("observation_mode")["success"].mean().to_string())
    sig = sum(1 for c in corr_rows if c["bonferroni_significant"])
    print(f"\n{sig}/{n_tests} correlations significant after Bonferroni correction")
    if importance is not None:
        print("\nTop 5 predictive WCAG features:")
        print(importance.head(5)[["feature", "importance_mean"]].to_string(index=False))
    print(f"\nAll outputs written to {OUTDIR}/")


if __name__ == "__main__":
    main()
