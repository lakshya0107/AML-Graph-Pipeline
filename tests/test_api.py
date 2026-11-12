"""HTTP-level tests for the FastAPI service (skipped if FastAPI is missing)."""

from __future__ import annotations

from collections.abc import Iterator

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from aml_graph.app import create_app  # noqa: E402
from aml_graph.scoring import RiskScorer  # noqa: E402


@pytest.fixture()
def client(fresh_scorer: RiskScorer) -> Iterator[TestClient]:
    with TestClient(create_app(fresh_scorer)) as test_client:
        yield test_client


def test_health(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_analyze_transaction_happy_path(client: TestClient) -> None:
    response = client.post(
        "/analyze_transaction",
        json={"src_account": "API_A", "dst_account": "API_B", "amount": 9500, "currency": "usd"},
    )
    assert response.status_code == 200
    body = response.json()
    assert 0.0 <= body["risk_score"] <= 1.0
    assert body["risk_tier"] in {"LOW", "MEDIUM", "HIGH"}
    assert isinstance(body["reasons"], list)


@pytest.mark.parametrize(
    "payload",
    [
        {"src_account": "A", "dst_account": "B", "amount": 10, "currency": "JPY"},
        {"src_account": "A", "dst_account": "A", "amount": 10, "currency": "USD"},
        {"src_account": "A", "dst_account": "B", "amount": -1, "currency": "USD"},
        {"src_account": "A", "dst_account": "B", "amount": 0, "currency": "USD"},
        {"src_account": "", "dst_account": "B", "amount": 5, "currency": "USD"},
        {"dst_account": "B", "amount": 5},
    ],
)
def test_invalid_payloads_are_rejected(client: TestClient, payload: dict) -> None:
    assert client.post("/analyze_transaction", json=payload).status_code == 422


def test_commit_flag_adds_the_transfer_to_the_graph(client: TestClient) -> None:
    payload = {"src_account": "C_SRC", "dst_account": "C_DST", "amount": 120, "currency": "GBP"}
    assert client.get("/ego_graph/C_SRC").status_code == 404
    committed = client.post("/analyze_transaction", params={"commit": "true"}, json=payload)
    assert committed.json()["committed"] is True
    assert client.get("/ego_graph/C_SRC").status_code == 200


def test_alerts_endpoint_respects_limit(client: TestClient) -> None:
    response = client.get("/alerts", params={"limit": 5})
    assert response.status_code == 200
    assert len(response.json()) <= 5
    assert client.get("/alerts", params={"limit": 0}).status_code == 422


def test_ego_graph_endpoint(client: TestClient) -> None:
    account = client.get("/alerts", params={"limit": 1}).json()[0]["src_account"]
    body = client.get(f"/ego_graph/{account}", params={"radius": 1}).json()
    assert body["center"] == account
    assert any(node["is_center"] for node in body["nodes"])
    assert client.get("/ego_graph/NOPE_NOPE").status_code == 404


def test_model_info(client: TestClient) -> None:
    body = client.get("/model_info").json()
    assert "pr_auc" in body["metrics"]["test"]
