"""Checks for the discussion metrics consumed by the Dash dashboard."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "Adolfo" / "results" / "metrics" / "dashboard_discussion_summary.json"


def test_dashboard_discussion_summary_exists_and_has_core_sections():
    assert SUMMARY.exists(), "Run: python etl/generate_dashboard_metrics.py"
    data = json.loads(SUMMARY.read_text(encoding="utf-8"))
    for key in [
        "executive_summary",
        "sample",
        "platform_metrics",
        "ranking",
        "kruskal_wallis",
        "hypotheses",
        "limitations",
        "recommended_conclusion",
    ]:
        assert key in data


def test_dashboard_discussion_summary_keeps_limits_explicit():
    data = json.loads(SUMMARY.read_text(encoding="utf-8"))
    assert data["sample"]["total_comments"] > 0
    assert data["ranking"][0]["plataforma"] == "youtube"
    assert data["kruskal_wallis"]["significant"] is True
    assert data["limitations"], "The dashboard must expose methodological limits."
    assert "universal" in " ".join(data["what_not_to_conclude"]).lower()