"""
Inferential statistics and publication figures for the Results chapter.

run_analysis.py produces descriptive statistics and the Spearman
correlations. This module adds the hypothesis tests that those
descriptives require for defensible interpretation:

  1. Kruskal-Wallis across WCAG compliance bands (do success rates
     genuinely differ by band, or is the visible gradient noise?).
  2. Friedman + post-hoc Wilcoxon across observation modes. The three
     modes are run on the *same* sites, so the comparison is paired;
     an unpaired test would overstate significance.
  3. Steiger's z for dependent correlations, which is the correct test
     of Sub-RQ3: not "is each mode's correlation significant?" (all
     are) but "is the Accessibility Tree correlation *stronger* than
     the others?".

All tests are non-parametric: site-level success rates are bounded
proportions over 9 runs and are not normally distributed.

Usage:
    python -m src.analysis.inferential
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

MERGED = "data/analysis/merged_site_mode.csv"
OUTDIR = Path("data/analysis")
MODES = ["accessibility_tree", "dom", "screenshot"]
# Band edges chosen a priori on the compliance scale, not from the data.
BAND_EDGES = [0, 0.85, 0.95, 1.01]
BAND_LABELS = ["low", "medium", "high"]


def load_pivot(merged: pd.DataFrame):
    """Site x mode success-rate matrix (paired) plus per-site compliance."""
    pivot = merged.pivot_table(
        index="url", columns="observation_mode", values="success_rate"
    ).dropna()
    compliance = (
        merged[merged.observation_mode == "accessibility_tree"]
        .set_index("url")["compliance_score"]
        .reindex(pivot.index)
    )
    return pivot, compliance


def band_test(pivot, compliance):
    """Kruskal-Wallis across compliance bands (Accessibility Tree mode)."""
    df = pd.DataFrame({
        "success_rate": pivot["accessibility_tree"],
        "compliance_score": compliance,
    })
    df["band"] = pd.cut(df.compliance_score, BAND_EDGES, labels=BAND_LABELS)
    groups = [df[df.band == b]["success_rate"].dropna() for b in BAND_LABELS]
    H, p = stats.kruskal(*groups)
    summary = df.groupby("band", observed=True)["success_rate"].agg(
        ["count", "mean", "std"]
    )
    return {
        "test": "kruskal_wallis_across_compliance_bands",
        "H": float(H),
        "p_value": float(p),
        "bands": {
            b: {
                "n": int(summary.loc[b, "count"]),
                "mean_success": float(summary.loc[b, "mean"]),
                "sd": float(summary.loc[b, "std"]),
            }
            for b in summary.index
        },
    }, df


def mode_tests(pivot):
    """Friedman omnibus + pairwise Wilcoxon (Bonferroni-corrected)."""
    chi2, p = stats.friedmanchisquare(*[pivot[m] for m in MODES])
    pairs = []
    comparisons = [
        ("accessibility_tree", "screenshot"),
        ("dom", "screenshot"),
        ("accessibility_tree", "dom"),
    ]
    for a, b in comparisons:
        stat, pw = stats.wilcoxon(pivot[a], pivot[b])
        diff = float((pivot[a] - pivot[b]).mean())
        # Matched-pairs rank-biserial correlation as the effect size
        d = pivot[a] - pivot[b]
        nonzero = d[d != 0]
        if len(nonzero):
            ranks = stats.rankdata(nonzero.abs())
            r_rb = float((ranks[nonzero > 0].sum() - ranks[nonzero < 0].sum())
                         / ranks.sum())
        else:
            r_rb = 0.0
        pairs.append({
            "comparison": f"{a}_vs_{b}",
            "mean_difference": diff,
            "wilcoxon_p": float(pw),
            "bonferroni_significant": bool(pw < 0.05 / len(comparisons)),
            "rank_biserial_effect_size": r_rb,
        })
    return {
        "test": "friedman_across_observation_modes",
        "chi2": float(chi2),
        "p_value": float(p),
        "n_sites": int(len(pivot)),
        "mean_success_by_mode": {m: float(pivot[m].mean()) for m in MODES},
        "pairwise": pairs,
    }


def _steiger_z(r12, r13, r23, n):
    """
    Steiger's z for two dependent correlations sharing one variable.
    r12, r13 share variable 1 (compliance); r23 is the correlation
    between the two success-rate vectors being compared.
    """
    z12, z13 = np.arctanh(r12), np.arctanh(r13)
    rm2 = (r12 ** 2 + r13 ** 2) / 2
    f = (1 - r23) / (2 * (1 - rm2))
    h = (1 - f * rm2) / (1 - rm2)
    return (z12 - z13) * np.sqrt((n - 3) / (2 * (1 - r23) * h))


def correlation_strength_tests(pivot, compliance):
    """Sub-RQ3: are the per-mode correlations significantly different?"""
    n = len(pivot)
    rho = {m: float(stats.spearmanr(compliance, pivot[m])[0]) for m in MODES}
    out = []
    for a, b in [("accessibility_tree", "screenshot"),
                 ("dom", "screenshot"),
                 ("accessibility_tree", "dom")]:
        r23 = float(stats.spearmanr(pivot[a], pivot[b])[0])
        z = _steiger_z(rho[a], rho[b], r23, n)
        p = float(2 * (1 - stats.norm.cdf(abs(z))))
        out.append({
            "comparison": f"{a}_vs_{b}",
            "rho_a": rho[a],
            "rho_b": rho[b],
            "steiger_z": float(z),
            "p_value": p,
            "significant": bool(p < 0.05),
        })
    return {"test": "steiger_dependent_correlation_comparison",
            "rho_by_mode": rho, "comparisons": out}


def make_figures(pivot, compliance, band_df, mode_res):
    """Publication figures with uncertainty shown, replacing the drafts."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"figure.dpi": 150, "font.size": 10,
                         "savefig.bbox": "tight"})

    def save(fig, name):
        fig.savefig(OUTDIR / f"{name}.png")
        fig.savefig(OUTDIR / f"{name}.pdf")
        plt.close(fig)

    # Fig 1: binned means with 95% CI — readable where a raw scatter is not,
    # because compliance clusters near 1.0 and success rates are discrete.
    fig, ax = plt.subplots(figsize=(7, 4.6))
    bins = [0, 0.80, 0.86, 0.90, 0.94, 0.97, 1.01]
    centres, means, errs = [], [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        sel = (compliance >= lo) & (compliance < hi)
        vals = pivot.loc[sel, "accessibility_tree"]
        if len(vals) < 3:
            continue
        centres.append((lo + min(hi, 1.0)) / 2)
        means.append(vals.mean())
        errs.append(1.96 * vals.std() / np.sqrt(len(vals)))
    # Markers only, no connecting line: compliance scores cluster above 0.75,
    # so a line would interpolate across a region containing no observations.
    ax.errorbar(centres, means, yerr=errs, fmt="o", markersize=7,
                capsize=4, color="#1f77b4", label="binned mean ± 95% CI")
    ax.scatter(compliance, pivot["accessibility_tree"], alpha=0.12,
               s=18, color="grey", label="individual sites")
    rho, p = stats.spearmanr(compliance, pivot["accessibility_tree"])
    ax.set_xlabel("WCAG compliance score")
    ax.set_ylabel("Agent task success rate")
    ax.set_title("Compliance vs. agent success (Accessibility Tree mode)\n"
                 rf"Spearman $\rho$ = {rho:.3f}, p = {p:.1e}, n = {len(pivot)}")
    ax.legend(loc="upper left", fontsize=8)
    ax.set_ylim(-0.05, 1.05)
    save(fig, "fig_compliance_vs_success")

    # Fig 2: success by mode, paired 95% CI
    fig, ax = plt.subplots(figsize=(6, 4))
    means = [pivot[m].mean() for m in MODES]
    cis = [1.96 * pivot[m].std() / np.sqrt(len(pivot)) for m in MODES]
    labels = ["Accessibility\nTree", "DOM", "Screenshot"]
    ax.bar(range(3), means, yerr=cis, capsize=5,
           color=["#1f77b4", "#4c9ed9", "#c44e52"])
    ax.set_xticks(range(3))
    ax.set_xticklabels(labels)
    ax.set_ylabel("Mean success rate")
    ax.set_title(f"Agent success by observation mode (n = {len(pivot)} sites)\n"
                 rf"Friedman $\chi^2$ = {mode_res['chi2']:.1f}, "
                 f"p = {mode_res['p_value']:.1e}")
    for i, (m, c) in enumerate(zip(means, cis)):
        ax.text(i, m + c + 0.015, f"{m:.1%}", ha="center", fontsize=9)
    ax.set_ylim(0, max(means) + 0.15)
    save(fig, "fig_success_by_mode")

    # Fig 3: success by compliance band, with n and CI
    fig, ax = plt.subplots(figsize=(6, 4))
    grp = band_df.groupby("band", observed=True)["success_rate"]
    means = grp.mean().reindex(BAND_LABELS)
    ns = grp.count().reindex(BAND_LABELS)
    cis = (1.96 * grp.std() / np.sqrt(grp.count())).reindex(BAND_LABELS)
    ax.bar(range(3), means.values, yerr=cis.values, capsize=5, color="#4c72b0")
    ax.set_xticks(range(3))
    ax.set_xticklabels([f"{b}\n(n={int(n)})" for b, n in zip(BAND_LABELS, ns.values)])
    ax.set_ylabel("Mean success rate")
    ax.set_title("Agent success by WCAG compliance band\n"
                 "(Accessibility Tree mode)")
    for i, (m, c) in enumerate(zip(means.values, cis.values)):
        ax.text(i, m + c + 0.015, f"{m:.1%}", ha="center", fontsize=9)
    ax.set_ylim(0, 1.0)
    save(fig, "fig_success_by_band")


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    merged = pd.read_csv(MERGED)
    pivot, compliance = load_pivot(merged)

    band_res, band_df = band_test(pivot, compliance)
    mode_res = mode_tests(pivot)
    corr_res = correlation_strength_tests(pivot, compliance)

    results = {"band_test": band_res, "mode_tests": mode_res,
               "correlation_strength": corr_res}
    (OUTDIR / "inferential_tests.json").write_text(json.dumps(results, indent=2))

    make_figures(pivot, compliance, band_df, mode_res)

    print("=== COMPLIANCE BANDS (Accessibility Tree) ===")
    for b, d in band_res["bands"].items():
        print(f"  {b:7s} n={d['n']:3d}  success={d['mean_success']:.1%}")
    print(f"  Kruskal-Wallis H={band_res['H']:.2f} p={band_res['p_value']:.3e}")

    print("\n=== OBSERVATION MODES (paired, n=%d) ===" % mode_res["n_sites"])
    for m, v in mode_res["mean_success_by_mode"].items():
        print(f"  {m:20s} {v:.1%}")
    print(f"  Friedman chi2={mode_res['chi2']:.2f} p={mode_res['p_value']:.3e}")
    for pr in mode_res["pairwise"]:
        flag = "SIG" if pr["bonferroni_significant"] else "ns"
        print(f"  {pr['comparison']:40s} diff={pr['mean_difference']:+.3f} "
              f"p={pr['wilcoxon_p']:.2e} [{flag}]")

    print("\n=== Sub-RQ3: IS THE AXTREE CORRELATION STRONGER? ===")
    for c in corr_res["comparisons"]:
        flag = "SIG" if c["significant"] else "ns"
        print(f"  {c['comparison']:40s} {c['rho_a']:.3f} vs {c['rho_b']:.3f} "
              f"z={c['steiger_z']:+.2f} p={c['p_value']:.3f} [{flag}]")

    print(f"\nWritten to {OUTDIR}/inferential_tests.json + figures")


if __name__ == "__main__":
    main()
