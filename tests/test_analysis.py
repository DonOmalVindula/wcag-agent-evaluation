"""
Unit tests for the analysis pipeline (correlation, audit loading, feature
importance) and the Accessibility Tree formatter. Statistical functions
are validated on synthetic data with known properties.
"""

import json

import numpy as np
import pandas as pd
import pytest

from src.agent.browser_env import _format_cdp_ax_tree
from src.analysis.correlation import (
    feature_importance_analysis,
    load_audit_results,
    spearman_correlation,
)


# ---------------------------------------------------------------------------
# Spearman correlation on synthetic data with known ground truth
# ---------------------------------------------------------------------------

class TestSpearman:
    def test_perfect_monotonic_relationship(self):
        df = pd.DataFrame({"x": range(20), "y": [v * 2 + 1 for v in range(20)]})
        r = spearman_correlation(df, "x", "y")
        assert r.coefficient == pytest.approx(1.0)
        assert r.significant

    def test_perfect_inverse_relationship(self):
        df = pd.DataFrame({"x": range(20), "y": [-v for v in range(20)]})
        r = spearman_correlation(df, "x", "y")
        assert r.coefficient == pytest.approx(-1.0)

    def test_independent_variables_not_significant(self):
        rng = np.random.default_rng(42)
        df = pd.DataFrame({"x": rng.normal(size=200), "y": rng.normal(size=200)})
        r = spearman_correlation(df, "x", "y")
        assert abs(r.coefficient) < 0.15
        assert not r.significant

    def test_insufficient_data_guard(self):
        df = pd.DataFrame({"x": [1, 2], "y": [2, 1]})
        r = spearman_correlation(df, "x", "y")
        assert r.n == 2
        assert not r.significant

    def test_nan_rows_dropped(self):
        df = pd.DataFrame({"x": [1, 2, 3, 4, None], "y": [1, 2, 3, 4, 5]})
        r = spearman_correlation(df, "x", "y")
        assert r.n == 4


# ---------------------------------------------------------------------------
# Audit result loading — schema produced by the accessibility module
# ---------------------------------------------------------------------------

def _audit_record(url="https://a.example", compliance=0.9):
    return {
        "url": url,
        "summary": {
            "compliance_score": compliance,
            "violation_rate": round(1 - compliance, 3),
            "wab_score": 0.02,
            "total_violations": 5,
            "total_rules_violated": 2,
            "total_rules_passed": 28,
        },
        "violations": [
            {"rule_id": "image-alt", "impact": "critical", "wcag_level": "A",
             "count": 3, "description": "", "wcag_tags": []},
            {"rule_id": "color-contrast", "impact": "serious", "wcag_level": "AA",
             "count": 2, "description": "", "wcag_tags": []},
        ],
    }


class TestAuditLoading:
    def test_loads_summary_and_per_rule_columns(self, tmp_path):
        p = tmp_path / "audit.json"
        p.write_text(json.dumps([_audit_record()]))
        df = load_audit_results(p)
        row = df.iloc[0]
        assert row["compliance_score"] == 0.9
        assert row["impact_critical"] == 3
        assert row["level_AA"] == 2
        assert row["rule_image-alt"] == 1

    def test_sites_without_violations_fill_zero(self, tmp_path):
        clean = _audit_record(url="https://clean.example", compliance=1.0)
        clean["violations"] = []
        p = tmp_path / "audit.json"
        p.write_text(json.dumps([_audit_record(), clean]))
        df = load_audit_results(p)
        clean_row = df[df.url == "https://clean.example"].iloc[0]
        assert clean_row["rule_image-alt"] == 0


# ---------------------------------------------------------------------------
# Feature importance — continuous target must be binarised, not crash
# ---------------------------------------------------------------------------

class TestFeatureImportance:
    def test_continuous_success_rate_target(self):
        rng = np.random.default_rng(7)
        n = 60
        compliance = rng.uniform(0.4, 1.0, n)
        df = pd.DataFrame({
            "compliance_score": compliance,
            "wab_score": 1 - compliance + rng.normal(0, 0.05, n),
            "total_violations": rng.integers(0, 40, n),
            # success strongly driven by compliance -> should rank first
            "success_rate": np.clip(compliance + rng.normal(0, 0.08, n), 0, 1),
        })
        imp = feature_importance_analysis(
            df, feature_cols=["compliance_score", "wab_score", "total_violations"],
            target_col="success_rate")
        assert list(imp.columns[:2]) == ["feature", "importance_mean"]
        assert imp.iloc[0]["feature"] == "compliance_score"


# ---------------------------------------------------------------------------
# Accessibility Tree formatting
# ---------------------------------------------------------------------------

def _node(nid, role, name="", children=(), ignored=False):
    return {"nodeId": nid, "role": {"value": role}, "name": {"value": name},
            "childIds": list(children), "ignored": ignored, "properties": []}


class TestAXTreeFormatter:
    def test_inline_text_boxes_are_skipped(self):
        # InlineTextBox duplicates its StaticText parent -> pure token waste
        nodes = [
            _node("1", "RootWebArea", "Home", children=["2"]),
            _node("2", "StaticText", "Welcome", children=["3"]),
            _node("3", "InlineTextBox", "Welcome"),
        ]
        out = _format_cdp_ax_tree(nodes)
        assert out.count("Welcome") == 1
        assert "InlineTextBox" not in out

    def test_ignored_and_anonymous_generics_skipped_but_children_kept(self):
        nodes = [
            _node("1", "RootWebArea", "Home", children=["2"]),
            _node("2", "generic", "", children=["3"]),
            _node("3", "link", "Contact us"),
        ]
        out = _format_cdp_ax_tree(nodes)
        assert "[link]" in out and "Contact us" in out
        assert "generic" not in out

    def test_output_is_capped(self):
        # A pathological page must not produce an unbounded observation
        nodes = [_node("1", "RootWebArea", "Big",
                       children=[str(i) for i in range(2, 4002)])]
        nodes += [_node(str(i), "link", f"Item {i} " + "x" * 40)
                  for i in range(2, 4002)]
        out = _format_cdp_ax_tree(nodes)
        assert len(out) <= 60100
        assert "truncated" in out

    def test_empty_tree(self):
        assert _format_cdp_ax_tree([]) == ""
