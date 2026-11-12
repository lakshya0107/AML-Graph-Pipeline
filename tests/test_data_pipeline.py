"""Tests for the synthetic ledger generator."""

from __future__ import annotations

import networkx as nx
import pandas as pd

from aml_graph import config
from aml_graph.data_pipeline import (
    LEDGER_COLUMNS,
    GeneratorConfig,
    add_amount_usd,
    generate_ledger,
    load_ledger,
    save_ledger,
)

TINY = GeneratorConfig(
    n_customers=200, n_merchants=10, n_employers=2, n_normal_txns=500, typology_count=3, days=20
)


def test_schema_and_basic_invariants(small_ledger: pd.DataFrame) -> None:
    ledger = small_ledger
    assert list(ledger.columns) == LEDGER_COLUMNS
    assert ledger["timestamp"].is_monotonic_increasing
    assert ledger["txn_id"].is_unique
    assert (ledger["src_account"] != ledger["dst_account"]).all()
    assert (ledger["amount"] > 0).all()
    assert set(ledger["currency"]) <= set(config.CURRENCIES)


def test_labels_match_typologies(small_ledger: pd.DataFrame) -> None:
    ledger = small_ledger
    assert set(ledger["typology"]) == {"normal", "fan_out", "fan_in", "circular"}
    assert ((ledger["typology"] != "normal").astype(int) == ledger["is_illicit"]).all()
    assert 0.0 < ledger["is_illicit"].mean() < 0.2


def test_generation_is_deterministic() -> None:
    pd.testing.assert_frame_equal(generate_ledger(TINY), generate_ledger(TINY))


def test_different_seeds_differ() -> None:
    other = GeneratorConfig(**{**TINY.__dict__, "seed": 99})
    assert not generate_ledger(TINY)["amount"].equals(generate_ledger(other)["amount"])


def test_fan_out_hub_pays_many_distinct_recipients(small_ledger: pd.DataFrame) -> None:
    fan_out = small_ledger[small_ledger["typology"] == "fan_out"]
    recipients = fan_out.groupby("src_account")["dst_account"].nunique()
    assert recipients.max() >= 10


def test_fan_in_collector_receives_from_many_senders(small_ledger: pd.DataFrame) -> None:
    fan_in = small_ledger[small_ledger["typology"] == "fan_in"]
    senders = fan_in.groupby("dst_account")["src_account"].nunique()
    assert senders.max() >= 10


def test_circular_flows_form_rings(small_ledger: pd.DataFrame) -> None:
    circular = small_ledger[small_ledger["typology"] == "circular"]
    graph = nx.DiGraph(zip(circular["src_account"], circular["dst_account"], strict=True))
    ring_sizes = [len(c) for c in nx.strongly_connected_components(graph)]
    assert max(ring_sizes) >= 3


def test_structuring_amounts_sit_below_threshold(small_ledger: pd.DataFrame) -> None:
    usd = add_amount_usd(small_ledger)
    fan_out = usd[usd["typology"] == "fan_out"]
    structured = fan_out[
        fan_out["amount_usd"].between(
            config.STRUCTURING_BAND * config.REPORTING_THRESHOLD_USD,
            config.REPORTING_THRESHOLD_USD,
            inclusive="left",
        )
    ]
    assert len(structured) > 0


def test_csv_round_trip(tmp_path, small_ledger: pd.DataFrame) -> None:
    path = tmp_path / "ledger.csv"
    save_ledger(small_ledger, path)
    loaded = load_ledger(path)
    assert len(loaded) == len(small_ledger)
    assert pd.api.types.is_datetime64_any_dtype(loaded["timestamp"])


def test_unsupported_currency_rejected() -> None:
    bad = pd.DataFrame({"amount": [10.0], "currency": ["JPY"]})
    try:
        add_amount_usd(bad)
    except ValueError as exc:
        assert "JPY" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected ValueError")
