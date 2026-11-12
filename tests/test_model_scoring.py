"""Tests for model training, evaluation and the real-time scorer."""

from __future__ import annotations

import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from aml_graph import config
from aml_graph.data_pipeline import save_ledger
from aml_graph.model import (
    FEATURE_COLUMNS,
    ModelBundle,
    build_feature_frame,
    chronological_split,
    evaluate,
    load_bundle,
    select_threshold,
    train_model,
)
from aml_graph.scoring import RiskScorer


# ---------------------------------------------------------------------- #
# Features and splits
# ---------------------------------------------------------------------- #
def test_feature_frame_has_all_model_columns(small_ledger: pd.DataFrame) -> None:
    frame = build_feature_frame(small_ledger)
    assert set(FEATURE_COLUMNS) <= set(frame.columns)
    assert not frame[FEATURE_COLUMNS].isna().any().any()
    assert len(frame) == len(small_ledger)


def test_chronological_split_is_disjoint_and_ordered(small_ledger: pd.DataFrame) -> None:
    frame = build_feature_frame(small_ledger)
    train, val, test = chronological_split(frame)
    assert not (train & val).any() and not (train & test).any() and not (val & test).any()
    ts = frame["timestamp"]
    assert ts[train].max() <= ts[val].min()
    assert ts[val].max() <= ts[test].min()
    assert not frame.loc[~frame["is_warm"], "is_illicit"].eq(2).any()
    assert not (train & ~frame["is_warm"]).any(), "warm-up rows must be excluded"


def test_threshold_selection_maximises_f1() -> None:
    y = np.array([0, 0, 0, 0, 1, 1, 1, 0])
    p = np.array([0.1, 0.2, 0.15, 0.3, 0.9, 0.8, 0.7, 0.6])
    threshold = select_threshold(y, p)
    assert evaluate(y, p, threshold)["f1"] == pytest.approx(1.0)


def test_evaluate_reports_expected_keys() -> None:
    y = np.array([0, 1, 0, 1])
    p = np.array([0.2, 0.9, 0.4, 0.6])
    metrics = evaluate(y, p, 0.5)
    assert {"pr_auc", "f1", "precision", "recall", "roc_auc", "confusion_matrix"} <= set(metrics)
    assert metrics["confusion_matrix"] == {"tn": 2, "fp": 0, "fn": 0, "tp": 2}


# ---------------------------------------------------------------------- #
# Training
# ---------------------------------------------------------------------- #
def test_trained_model_beats_the_random_baseline(bundle: ModelBundle) -> None:
    test = bundle.metrics["test"]
    assert test["pr_auc"] > 3 * test["prevalence"]
    assert 0.0 < bundle.threshold < 1.0
    assert bundle.feature_names == FEATURE_COLUMNS
    assert bundle.metrics["train_scale_pos_weight"] > 1.0


def test_ablation_reports_graph_feature_lift(small_ledger: pd.DataFrame) -> None:
    result = train_model(small_ledger, backend="sklearn", run_ablation=True)
    assert "ablation_tabular_only" in result.metrics
    assert "graph_feature_pr_auc_lift" in result.metrics


def test_unknown_backend_is_rejected(small_ledger: pd.DataFrame) -> None:
    with pytest.raises(ValueError):
        train_model(small_ledger, backend="nope")


def test_xgboost_backend_trains(small_ledger: pd.DataFrame) -> None:
    pytest.importorskip("xgboost")
    result = train_model(small_ledger, backend="xgboost", run_ablation=False)
    assert result.metrics["test"]["pr_auc"] > 3 * result.metrics["test"]["prevalence"]


def test_bundle_written_by_the_cli_loads_in_another_process(
    tmp_path, small_ledger: pd.DataFrame
) -> None:
    """Regression: ``python -m`` must not pickle the bundle under ``__main__``."""
    ledger_path = tmp_path / "ledger.csv"
    model_path = tmp_path / "model.joblib"
    save_ledger(small_ledger, ledger_path)
    subprocess.run(
        [
            sys.executable, "-m", "aml_graph.model",
            "--ledger", str(ledger_path),
            "--out", str(model_path),
            "--metrics-out", str(tmp_path / "metrics.json"),
            "--backend", "sklearn",
            "--no-ablation",
        ],
        cwd=config.ROOT_DIR,
        check=True,
        capture_output=True,
    )
    loaded = load_bundle(model_path)
    assert isinstance(loaded, ModelBundle)
    assert (tmp_path / "metrics.json").exists()


# ---------------------------------------------------------------------- #
# Scoring
# ---------------------------------------------------------------------- #
def test_analyze_returns_a_complete_assessment(scorer: RiskScorer) -> None:
    result = scorer.analyze("UNSEEN_A", "UNSEEN_B", 42.0, "EUR")
    assert 0.0 <= result["risk_score"] <= 1.0
    assert result["risk_tier"] in {"LOW", "MEDIUM", "HIGH"}
    assert result["src_known"] is False and result["dst_known"] is False
    assert set(FEATURE_COLUMNS) <= set(result["features"])
    assert result["committed"] is False


def test_analyze_validates_input(scorer: RiskScorer) -> None:
    with pytest.raises(ValueError):
        scorer.analyze("X", "X", 10.0)
    with pytest.raises(ValueError):
        scorer.analyze("X", "Y", -5.0)
    with pytest.raises(ValueError):
        scorer.analyze("X", "Y", 10.0, "JPY")


def test_closing_a_ring_scores_higher_and_explains_why(fresh_scorer: RiskScorer) -> None:
    scorer = fresh_scorer
    scorer.analyze("RING_A", "RING_B", 40_000.0, commit=True)
    scorer.analyze("RING_B", "RING_C", 39_000.0, commit=True)

    closing = scorer.analyze("RING_C", "RING_A", 38_000.0)
    baseline = scorer.analyze("RING_C", "RING_D", 38_000.0)

    assert closing["features"]["closes_cycle"] == 1.0
    assert baseline["features"]["closes_cycle"] == 0.0
    assert closing["risk_score"] > baseline["risk_score"]
    assert any("circular" in reason.lower() for reason in closing["reasons"])


def test_commit_updates_the_live_graph_and_alert_queue(fresh_scorer: RiskScorer) -> None:
    scorer = fresh_scorer
    scorer.analyze("LIVE_SRC", "LIVE_DST", 9_500.0, commit=True)

    payload = scorer.ego_graph("LIVE_SRC")
    assert {n["id"] for n in payload["nodes"]} == {"LIVE_SRC", "LIVE_DST"}
    assert payload["edges"][0]["txn_count"] == 1
    assert any(a["txn_id"].startswith("LIVE") for a in scorer.alerts(limit=10_000))


def test_alerts_are_sorted_by_descending_risk(scorer: RiskScorer) -> None:
    alerts = scorer.alerts(limit=15)
    scores = [a["risk_score"] for a in alerts]
    assert len(alerts) <= 15
    assert scores == sorted(scores, reverse=True)
    assert all(a["flagged"] == (a["risk_score"] >= scorer.bundle.threshold) for a in alerts)


def test_ego_graph_payload_is_self_consistent(scorer: RiskScorer) -> None:
    account = scorer.alerts(limit=1)[0]["src_account"]
    payload = scorer.ego_graph(account, radius=2, max_nodes=30)
    ids = {n["id"] for n in payload["nodes"]}
    assert account in ids and len(ids) <= 30
    assert sum(n["is_center"] for n in payload["nodes"]) == 1
    assert all(e["source"] in ids and e["target"] in ids for e in payload["edges"])


def test_ego_graph_for_unknown_account_raises(scorer: RiskScorer) -> None:
    with pytest.raises(KeyError):
        scorer.ego_graph("DOES_NOT_EXIST")


def test_model_info_exposes_training_metrics(scorer: RiskScorer) -> None:
    info = scorer.model_info()
    assert info["metrics"]["test"]["pr_auc"] > 0
    assert info["reference_graph"]["accounts"] > 0
