"""Shared fixtures: a small deterministic ledger, trained model and scorer."""

from __future__ import annotations

import pandas as pd
import pytest

from aml_graph.data_pipeline import GeneratorConfig, generate_ledger
from aml_graph.model import ModelBundle, train_model
from aml_graph.scoring import RiskScorer

SMALL_CONFIG = GeneratorConfig(
    n_customers=500,
    n_merchants=30,
    n_employers=3,
    n_normal_txns=6_000,
    typology_count=12,
    days=45,
    seed=7,
)


@pytest.fixture(scope="session")
def small_ledger() -> pd.DataFrame:
    return generate_ledger(SMALL_CONFIG)


@pytest.fixture(scope="session")
def bundle(small_ledger: pd.DataFrame) -> ModelBundle:
    # scikit-learn backend keeps the suite runnable without XGBoost installed.
    return train_model(small_ledger, backend="sklearn", run_ablation=False)


@pytest.fixture(scope="session")
def scorer(bundle: ModelBundle, small_ledger: pd.DataFrame) -> RiskScorer:
    """Read-only scorer shared across tests (do not commit transfers to it)."""
    return RiskScorer(bundle, small_ledger)


@pytest.fixture()
def fresh_scorer(bundle: ModelBundle, small_ledger: pd.DataFrame) -> RiskScorer:
    """Per-test scorer for tests that mutate the live graph."""
    return RiskScorer(bundle, small_ledger)
