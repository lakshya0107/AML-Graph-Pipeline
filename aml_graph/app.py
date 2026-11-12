"""FastAPI service exposing real-time transaction risk scoring.

Run with::

    uvicorn aml_graph.app:app --reload

Endpoints
---------
``GET  /health``                  liveness and model summary
``POST /analyze_transaction``     risk score, tier and reason codes for a transfer
                                  (``?commit=true`` also adds it to the live graph)
``GET  /alerts``                  highest-risk recent transactions
``GET  /ego_graph/{account_id}``  neighbourhood of an account (nodes + edges)
``GET  /model_info``              training metrics and feature importances
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Query, Request
from pydantic import BaseModel, Field, field_validator, model_validator

from aml_graph import __version__, config
from aml_graph.scoring import RiskScorer


# ---------------------------------------------------------------------- #
# Schemas
# ---------------------------------------------------------------------- #
class TransactionRequest(BaseModel):
    """An incoming transfer to be evaluated."""

    src_account: str = Field(..., min_length=1, max_length=64, examples=["ACC000123"])
    dst_account: str = Field(..., min_length=1, max_length=64, examples=["ACC000456"])
    amount: float = Field(..., gt=0, allow_inf_nan=False, examples=[9500.0])
    currency: str = Field("USD", min_length=3, max_length=3)
    timestamp: datetime | None = Field(
        None,
        description=(
            "Event time (wall-clock). Defaults to the reference ledger's latest "
            "timestamp so demos on a historical ledger stay inside the graph window."
        ),
    )

    @field_validator("currency")
    @classmethod
    def _supported_currency(cls, value: str) -> str:
        value = value.upper()
        if value not in config.FX_TO_USD:
            supported = sorted(config.FX_TO_USD)
            raise ValueError(f"Unsupported currency {value!r}; use one of {supported}")
        return value

    @model_validator(mode="after")
    def _distinct_accounts(self) -> TransactionRequest:
        if self.src_account == self.dst_account:
            raise ValueError("src_account and dst_account must differ")
        return self


class AnalysisResponse(BaseModel):
    """Risk assessment for a single transfer."""

    risk_score: float = Field(..., ge=0, le=1)
    risk_tier: str
    flagged: bool
    threshold: float
    reasons: list[str]
    features: dict[str, float]
    src_known: bool
    dst_known: bool
    committed: bool = False


# ---------------------------------------------------------------------- #
# Application factory
# ---------------------------------------------------------------------- #
def get_scorer(request: Request) -> RiskScorer:
    """Dependency returning the scorer created at start-up."""
    scorer = getattr(request.app.state, "scorer", None)
    if scorer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return scorer


def create_app(scorer: RiskScorer | None = None) -> FastAPI:
    """Build the FastAPI app. Pass ``scorer`` to inject one (used in tests)."""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.scorer = scorer or RiskScorer.from_paths(
            model_path=Path(os.environ.get("AML_MODEL_PATH", config.MODEL_PATH)),
            ledger_path=Path(os.environ.get("AML_LEDGER_PATH", config.LEDGER_PATH)),
        )
        yield

    app = FastAPI(
        title="Graph-Based AML Detection API",
        version=__version__,
        description="Real-time scoring of transfers using transactional and graph features.",
        lifespan=lifespan,
    )

    @app.get("/health")
    def health(scorer: RiskScorer = Depends(get_scorer)) -> dict[str, Any]:
        return {
            "status": "ok",
            "version": __version__,
            "backend": scorer.bundle.backend,
            "threshold": scorer.bundle.threshold,
        }

    @app.post("/analyze_transaction", response_model=AnalysisResponse)
    def analyze_transaction(
        payload: TransactionRequest,
        commit: bool = Query(
            False, description="Append the transfer to the live graph after scoring."
        ),
        scorer: RiskScorer = Depends(get_scorer),
    ) -> dict[str, Any]:
        try:
            return scorer.analyze(
                payload.src_account,
                payload.dst_account,
                payload.amount,
                payload.currency,
                payload.timestamp,
                commit=commit,
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.get("/alerts")
    def alerts(
        limit: int = Query(25, ge=1, le=200), scorer: RiskScorer = Depends(get_scorer)
    ) -> list[dict[str, Any]]:
        return scorer.alerts(limit)

    @app.get("/ego_graph/{account_id}")
    def ego_graph(
        account_id: str,
        radius: int = Query(2, ge=1, le=3),
        max_nodes: int = Query(60, ge=5, le=200),
        scorer: RiskScorer = Depends(get_scorer),
    ) -> dict[str, Any]:
        try:
            return scorer.ego_graph(account_id, radius=radius, max_nodes=max_nodes)
        except KeyError as exc:
            raise HTTPException(
                status_code=404, detail=f"Account {account_id!r} not in the live graph"
            ) from exc

    @app.get("/model_info")
    def model_info(scorer: RiskScorer = Depends(get_scorer)) -> dict[str, Any]:
        return scorer.model_info()

    return app


app = create_app()
