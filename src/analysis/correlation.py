"""
Correlation Analysis Module

Statistical analysis measuring the relationship between WCAG compliance
scores and AI agent task success rates. Implements Spearman rank correlation,
regression analysis, and feature importance ranking.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance


@dataclass
class CorrelationResult:
    """Result of a correlation analysis between two variables."""
    variable_x: str
    variable_y: str
    method: str
    coefficient: float
    p_value: float
    n: int
    significant: bool  # p < alpha

    def to_dict(self) -> dict:
        return {
            "x": self.variable_x,
            "y": self.variable_y,
            "method": self.method,
            "coefficient": round(self.coefficient, 4),
            "p_value": round(self.p_value, 6),
            "n": self.n,
            "significant": self.significant,
        }


def load_audit_results(path: str | Path) -> pd.DataFrame:
    """Load accessibility audit results JSON into a DataFrame."""
    with open(path) as f:
        data = json.load(f)

    rows = []
    for entry in data:
        row = {
            "url": entry["url"],
            "compliance_score": entry["summary"]["compliance_score"],
            "violation_rate": entry["summary"]["violation_rate"],
            "wab_score": entry["summary"]["wab_score"],
            "total_violations": entry["summary"]["total_violations"],
            "rules_violated": entry["summary"]["total_rules_violated"],
            "rules_passed": entry["summary"]["total_rules_passed"],
        }
        # Add per-impact counts
        for v in entry.get("violations", []):
            impact_key = f"impact_{v['impact']}"
            row[impact_key] = row.get(impact_key, 0) + v["count"]
            # Per-level counts
            level_key = f"level_{v['wcag_level']}"
            row[level_key] = row.get(level_key, 0) + v["count"]
            # Per-rule binary flags
            row[f"rule_{v['rule_id']}"] = 1
        rows.append(row)

    df = pd.DataFrame(rows).fillna(0)
    return df


def load_agent_results(path: str | Path) -> pd.DataFrame:
    """Load agent task execution results JSON into a DataFrame."""
    with open(path) as f:
        data = json.load(f)

    rows = []
    for entry in data:
        rows.append({
            "url": entry["url"],
            "task": entry["task"],
            "observation_mode": entry["observation_mode"],
            "success": 1 if entry["status"] == "success" else 0,
            "total_steps": entry["total_steps"],
            "duration_seconds": entry["duration_seconds"],
        })

    return pd.DataFrame(rows)


def merge_datasets(
    audit_df: pd.DataFrame,
    agent_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge audit and agent results by URL.
    Aggregates agent results per URL (success rate across tasks).
    """
    # Aggregate agent results per URL and mode
    agent_agg = (
        agent_df.groupby(["url", "observation_mode"])
        .agg(
            success_rate=("success", "mean"),
            avg_steps=("total_steps", "mean"),
            avg_duration=("duration_seconds", "mean"),
            total_tasks=("success", "count"),
            total_successes=("success", "sum"),
        )
        .reset_index()
    )

    # Merge with audit data
    merged = agent_agg.merge(audit_df, on="url", how="inner")
    return merged


def spearman_correlation(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    alpha: float = 0.05,
) -> CorrelationResult:
    """Compute Spearman rank correlation between two columns."""
    valid = df[[x_col, y_col]].dropna()
    n = len(valid)

    if n < 3:
        return CorrelationResult(
            variable_x=x_col, variable_y=y_col, method="spearman",
            coefficient=0.0, p_value=1.0, n=n, significant=False,
        )

    rho, p_value = stats.spearmanr(valid[x_col], valid[y_col])
    # nan guard: spearmanr returns nan for a constant column
    rho = 0.0 if np.isnan(rho) else float(rho)
    p_value = 1.0 if np.isnan(p_value) else float(p_value)
    return CorrelationResult(
        variable_x=x_col,
        variable_y=y_col,
        method="spearman",
        coefficient=rho,
        p_value=p_value,
        n=n,
        significant=bool(p_value < alpha),
    )


def run_correlation_analysis(
    df: pd.DataFrame,
    accessibility_cols: list[str] | None = None,
    performance_cols: list[str] | None = None,
    alpha: float = 0.05,
) -> list[CorrelationResult]:
    """
    Run Spearman correlations between all accessibility metrics
    and all performance metrics.
    """
    if accessibility_cols is None:
        accessibility_cols = [
            "compliance_score", "violation_rate", "wab_score", "total_violations",
        ]
    if performance_cols is None:
        performance_cols = ["success_rate", "avg_steps", "avg_duration"]

    # Filter to columns that exist in the DataFrame
    accessibility_cols = [c for c in accessibility_cols if c in df.columns]
    performance_cols = [c for c in performance_cols if c in df.columns]

    results = []
    for x_col in accessibility_cols:
        for y_col in performance_cols:
            result = spearman_correlation(df, x_col, y_col, alpha)
            results.append(result)
    return results


def feature_importance_analysis(
    df: pd.DataFrame,
    feature_cols: list[str] | None = None,
    target_col: str = "success_rate",
    method: str = "random_forest",
) -> pd.DataFrame:
    """
    Identify which WCAG features most strongly predict agent performance.

    Uses tree-based models with permutation importance to rank features.
    Returns a DataFrame with feature names and importance scores.
    """
    if feature_cols is None:
        # Use all rule_ columns plus standard metrics
        rule_cols = [c for c in df.columns if c.startswith("rule_")]
        metric_cols = ["compliance_score", "wab_score", "total_violations"]
        impact_cols = [c for c in df.columns if c.startswith("impact_")]
        level_cols = [c for c in df.columns if c.startswith("level_")]
        feature_cols = metric_cols + impact_cols + level_cols + rule_cols

    feature_cols = [c for c in feature_cols if c in df.columns]
    valid = df[feature_cols + [target_col]].dropna()

    if len(valid) < 10:
        print(f"Warning: Only {len(valid)} samples — feature importance may be unreliable")

    X = valid[feature_cols]
    y = valid[target_col]

    # Binarize a continuous/rate target into high vs low performers at the
    # median so the tree classifier has discrete classes. (A raw success
    # rate is continuous; feeding it to a classifier errors.) An already
    # binary target passes through unchanged.
    if y.nunique() > 2:
        y_binary = (y >= y.median()).astype(int)
    else:
        y_binary = y.astype(int)

    if method == "random_forest":
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    elif method == "gradient_boosting":
        model = GradientBoostingClassifier(n_estimators=100, random_state=42)
    else:
        raise ValueError(f"Unknown method: {method}")

    model.fit(X, y_binary)

    # Permutation importance (more reliable than built-in feature importance)
    perm_imp = permutation_importance(model, X, y_binary, n_repeats=10, random_state=42)

    importance_df = pd.DataFrame({
        "feature": feature_cols,
        "importance_mean": perm_imp.importances_mean,
        "importance_std": perm_imp.importances_std,
    }).sort_values("importance_mean", ascending=False)

    return importance_df


def generate_report(
    correlations: list[CorrelationResult],
    importance_df: pd.DataFrame | None = None,
) -> str:
    """Generate a text summary of the analysis results."""
    lines = ["=" * 70, "CORRELATION ANALYSIS REPORT", "=" * 70, ""]

    lines.append("SPEARMAN RANK CORRELATIONS")
    lines.append("-" * 40)
    for c in correlations:
        sig = "*" if c.significant else " "
        lines.append(
            f"  {sig} {c.variable_x} vs {c.variable_y}: "
            f"rho={c.coefficient:+.4f}, p={c.p_value:.6f}, n={c.n}"
        )
    lines.append("")
    sig_count = sum(1 for c in correlations if c.significant)
    lines.append(f"  {sig_count}/{len(correlations)} correlations significant at alpha=0.05")
    lines.append("")

    if importance_df is not None and not importance_df.empty:
        lines.append("FEATURE IMPORTANCE (top 10)")
        lines.append("-" * 40)
        for _, row in importance_df.head(10).iterrows():
            lines.append(
                f"  {row['feature']:30s} {row['importance_mean']:.4f} "
                f"(+/- {row['importance_std']:.4f})"
            )

    return "\n".join(lines)
