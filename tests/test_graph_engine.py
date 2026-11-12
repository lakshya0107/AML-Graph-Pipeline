"""Tests for the streaming graph topology engine."""

from __future__ import annotations

import pandas as pd
import pytest

from aml_graph.graph_engine import GRAPH_FEATURES, GraphConfig, GraphEngine

HOUR = 3_600.0
DAY = 86_400.0


def make_engine() -> GraphEngine:
    return GraphEngine(GraphConfig(window_days=7))


def make_ledger(rows: list[tuple[str, str, float, float]]) -> pd.DataFrame:
    """rows: (src, dst, amount_usd, seconds since 2026-01-01)."""
    start = pd.Timestamp("2026-01-01")
    return pd.DataFrame(
        {
            "txn_id": [f"T{i}" for i in range(len(rows))],
            "timestamp": [start + pd.Timedelta(seconds=r[3]) for r in rows],
            "src_account": [r[0] for r in rows],
            "dst_account": [r[1] for r in rows],
            "amount": [r[2] for r in rows],
            "currency": "USD",
        }
    )


def test_transfer_that_closes_a_ring_is_flagged_before_it_is_added() -> None:
    engine = make_engine()
    engine.add_transaction("A", "B", 100.0, 0.0)
    engine.add_transaction("B", "C", 99.0, 10.0)

    features = engine.features_for("C", "A", 98.0)

    assert features["closes_cycle"] == 1.0
    assert features["min_return_hops"] == 2.0
    assert features["same_ring"] == 1.0
    assert features["src_scc_size"] == features["dst_scc_size"] == 3.0
    assert features["src_in_ring"] == 1.0
    assert not engine.graph.has_edge("C", "A"), "scoring must not mutate the graph"


def test_scc_membership_after_ring_completes() -> None:
    engine = make_engine()
    for src, dst, ts in [("A", "B", 0.0), ("B", "C", 10.0), ("C", "A", 20.0)]:
        engine.add_transaction(src, dst, 100.0, ts)
    engine.refresh()

    for account in "ABC":
        node = engine.describe_node(account)
        assert node["scc_size"] == 3.0
        assert node["in_ring"] == 1.0
    # A second lap of the ring still looks like a cycle.
    assert engine.features_for("A", "B", 95.0)["closes_cycle"] == 1.0


def test_reciprocal_pair_is_not_a_ring() -> None:
    engine = make_engine()
    engine.add_transaction("X", "Y", 10.0, 0.0)
    assert engine.features_for("Y", "X", 10.0)["closes_cycle"] == 0.0
    engine.add_transaction("Y", "X", 10.0, 5.0)
    engine.refresh()
    node = engine.describe_node("X")
    assert node["scc_size"] == 2.0
    assert node["in_ring"] == 0.0


def test_open_chain_and_unknown_accounts_are_not_flagged() -> None:
    engine = make_engine()
    engine.add_transaction("A", "B", 10.0, 0.0)
    engine.add_transaction("B", "C", 10.0, 1.0)
    assert engine.features_for("C", "D", 10.0)["closes_cycle"] == 0.0
    unknown = engine.features_for("N1", "N2", 5.0)
    assert unknown["src_out_degree"] == 1.0 and unknown["dst_in_degree"] == 1.0
    assert unknown["src_scc_size"] == 1.0 and unknown["closes_cycle"] == 0.0


def test_degree_features_include_the_transfer_being_scored() -> None:
    engine = make_engine()
    for i in range(5):
        engine.add_transaction("HUB", f"R{i}", 100.0, float(i))

    new_recipient = engine.features_for("HUB", "R9", 100.0)
    repeat_recipient = engine.features_for("HUB", "R0", 100.0)

    assert new_recipient["src_out_degree"] == 6.0
    assert repeat_recipient["src_out_degree"] == 5.0
    assert repeat_recipient["pair_txn_count"] == 2.0
    assert repeat_recipient["pair_total_usd"] == pytest.approx(200.0)


def test_pass_through_account_has_high_flow_balance() -> None:
    engine = make_engine()
    engine.add_transaction("S", "M", 1_000.0, 0.0)
    engine.add_transaction("M", "T", 950.0, 10.0)
    assert engine.describe_node("M")["flow_balance"] == pytest.approx(0.95)
    assert engine.describe_node("S")["flow_balance"] == 0.0


def test_pagerank_is_a_distribution() -> None:
    engine = make_engine()
    for i, (s, d) in enumerate([("A", "B"), ("B", "C"), ("C", "A"), ("D", "A")]):
        engine.add_transaction(s, d, 10.0 * (i + 1), float(i))
    engine.refresh()
    total = sum(engine.describe_node(n)["pagerank"] for n in "ABCD")
    assert total == pytest.approx(1.0)


def test_transfers_expire_after_the_window() -> None:
    engine = make_engine()
    engine.add_transaction("A", "B", 10.0, 0.0)
    engine.add_transaction("C", "D", 10.0, 6 * DAY)

    engine.expire(8 * DAY)

    assert not engine.has_account("A") and not engine.has_account("B")
    assert engine.graph.has_edge("C", "D")


def test_expiry_only_removes_an_edge_when_all_its_transfers_age_out() -> None:
    engine = make_engine()
    engine.add_transaction("A", "B", 10.0, 0.0)
    engine.add_transaction("A", "B", 20.0, 6 * DAY)
    engine.expire(8 * DAY)
    data = engine.graph.edges["A", "B"]
    assert data["txn_count"] == 1
    assert data["weight"] == pytest.approx(20.0)


def test_replay_features_are_causal() -> None:
    ledger = make_ledger([("A", "B", 100, 0), ("B", "C", 99, HOUR), ("C", "A", 98, 2 * HOUR)])
    features = GraphEngine().replay(ledger)

    assert list(features.columns) == GRAPH_FEATURES
    assert features["closes_cycle"].tolist() == [0.0, 0.0, 1.0]
    # The first transfer cannot know about the hub activity that follows it.
    assert features["dst_out_degree"].iloc[0] == 0.0


def test_replay_is_aligned_to_the_original_index_and_leaves_engine_live() -> None:
    ledger = make_ledger([("A", "B", 1, 0), ("B", "C", 1, 10), ("C", "A", 1, 20)])
    shuffled = ledger.sample(frac=1.0, random_state=1)
    engine = GraphEngine()
    features = engine.replay(shuffled)

    assert features.index.equals(shuffled.index)
    assert features.loc[ledger.index[2], "closes_cycle"] == 1.0
    assert engine.graph.number_of_edges() == 3


def test_replay_matches_manual_streaming() -> None:
    rows = [
        ("A", "B", 50, 0),
        ("B", "C", 40, HOUR),
        ("A", "C", 30, 2 * HOUR),
        ("C", "A", 20, 3 * HOUR),
    ]
    ledger = make_ledger(rows)
    replayed = GraphEngine().replay(ledger)

    manual = GraphEngine()
    for i, (src, dst, usd, ts) in enumerate(rows):
        manual.expire(ts)
        manual.maybe_refresh(ts)
        expected = manual.features_for(src, dst, usd)
        for name in GRAPH_FEATURES:
            assert replayed.iloc[i][name] == pytest.approx(expected[name]), (i, name)
        manual.add_transaction(src, dst, usd, ts)


def test_ego_graph_is_capped_and_keeps_the_centre() -> None:
    engine = make_engine()
    for i in range(100):
        engine.add_transaction(f"L{i}", "HUB", 10.0, float(i))
    engine.refresh()

    ego = engine.ego_graph("L3", radius=2, max_nodes=20)

    assert ego.number_of_nodes() <= 20
    assert "L3" in ego and "HUB" in ego
    with pytest.raises(KeyError):
        engine.ego_graph("missing")
