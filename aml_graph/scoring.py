"""Real-time risk scoring on top of a trained model and a streaming graph.

The :class:`RiskScorer` is framework-agnostic (no FastAPI import) so it can be
unit-tested and reused from batch jobs. At start-up it

1. replays the most recent two windows of the ledger through a
   :class:`~aml_graph.graph_engine.GraphEngine`. This leaves the engine in the
   live state it would have at the end of the ledger, and scores every replayed
   transfer *as it arrived* (with only earlier history visible),
2. keeps the transfers of the newest window as the alert queue, and
3. derives per-account and per-edge risk for the ego-graph view.

Incoming transfers are scored with :meth:`GraphEngine.features_for`, the same
code path used during training. With ``commit=True`` the transfer is then
appended to the live graph so later requests see it.
"""

from __future__ import annotations

import heapq
import logging
import threading
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from aml_graph import config
from aml_graph.data_pipeline import add_amount_usd, load_ledger
from aml_graph.graph_engine import GraphEngine
from aml_graph.model import ModelBundle, build_tabular_features, load_bundle

LOGGER = logging.getLogger(__name__)

FAN_THRESHOLD = 10  # distinct counterparties that count as a fan-in / fan-out
_EPOCH = pd.Timestamp("1970-01-01")


class RiskScorer:
    """Scores individual transfers and serves graph context for investigators."""

    def __init__(self, bundle: ModelBundle, ledger: pd.DataFrame) -> None:
        self.bundle = bundle
        self._lock = threading.RLock()
        self._commit_count = 0

        ledger = add_amount_usd(ledger).reset_index(drop=True)
        ledger["timestamp"] = pd.to_datetime(ledger["timestamp"])
        window = pd.Timedelta(days=bundle.window_days)
        latest = ledger["timestamp"].max()
        tail = ledger[ledger["timestamp"] > latest - 2 * window].reset_index(drop=True)

        self.engine = GraphEngine(bundle.graph_config)
        graph_features = self.engine.replay(tail)
        features = pd.concat([build_tabular_features(tail), graph_features], axis=1)
        tail["risk_score"] = self._predict(features)

        recent = tail[tail["timestamp"] > latest - window]
        self._alert_rows: list[dict[str, Any]] = [
            {
                "txn_id": str(r.txn_id),
                "timestamp": r.timestamp.isoformat(),
                "src_account": str(r.src_account),
                "dst_account": str(r.dst_account),
                "amount": float(r.amount),
                "currency": str(r.currency),
                "risk_score": float(r.risk_score),
            }
            for r in recent.itertuples(index=False)
        ]
        self._edge_risk: dict[tuple[str, str], float] = {}
        self._node_risk: dict[str, float] = {}
        for row in self._alert_rows:
            self._record_risk(row["src_account"], row["dst_account"], row["risk_score"])
        self.as_of = latest
        LOGGER.info(
            "Scorer ready: %d recent transactions, %d accounts, %d edges, as of %s",
            len(self._alert_rows),
            self.engine.graph.number_of_nodes(),
            self.engine.graph.number_of_edges(),
            latest,
        )

    @classmethod
    def from_paths(
        cls, model_path: Path | None = None, ledger_path: Path | None = None
    ) -> RiskScorer:
        """Load the persisted model and ledger from disk."""
        model_path = Path(model_path or config.MODEL_PATH)
        ledger_path = Path(ledger_path or config.LEDGER_PATH)
        for path in (model_path, ledger_path):
            if not path.exists():
                raise FileNotFoundError(
                    f"{path} not found. Run `python -m aml_graph.data_pipeline` and "
                    "`python -m aml_graph.model` first."
                )
        return cls(load_bundle(model_path), load_ledger(ledger_path))

    # ------------------------------------------------------------------ #
    # Model access
    # ------------------------------------------------------------------ #
    def _predict(self, features: pd.DataFrame) -> np.ndarray:
        ordered = features[self.bundle.feature_names]
        return self.bundle.model.predict_proba(ordered)[:, 1]

    def _tier(self, score: float) -> str:
        if score >= self.bundle.threshold:
            return "HIGH"
        if score >= 0.5 * self.bundle.threshold:
            return "MEDIUM"
        return "LOW"

    def _record_risk(self, src: str, dst: str, score: float) -> None:
        key = (src, dst)
        self._edge_risk[key] = max(self._edge_risk.get(key, 0.0), score)
        for account in key:
            self._node_risk[account] = max(self._node_risk.get(account, 0.0), score)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def analyze(
        self,
        src_account: str,
        dst_account: str,
        amount: float,
        currency: str = "USD",
        timestamp: Any = None,
        commit: bool = False,
    ) -> dict[str, Any]:
        """Score a single incoming transfer.

        Args:
            timestamp: Event time (wall-clock). Defaults to the ledger's latest
                timestamp so that demos on a historical ledger stay in-window.
            commit: Append the transfer to the live graph after scoring.

        Raises:
            ValueError: For self-transfers, non-positive amounts or unsupported
                currencies.
        """
        if src_account == dst_account:
            raise ValueError("Source and destination accounts must differ")
        if not amount > 0:
            raise ValueError("Amount must be positive")
        ts = pd.Timestamp(timestamp) if timestamp is not None else self.as_of
        if ts.tzinfo is not None:  # wall-clock time, consistent with the training data
            ts = ts.tz_localize(None)

        row = pd.DataFrame({"amount": [amount], "currency": [currency], "timestamp": [ts]})
        tabular = build_tabular_features(row).iloc[0].to_dict()
        usd = tabular["amount_usd"]
        with self._lock:
            graph = self.engine.features_for(src_account, dst_account, usd)
            features = {**tabular, **graph}
            score = float(self._predict(pd.DataFrame([features]))[0])
            src_known = self.engine.has_account(src_account)
            dst_known = self.engine.has_account(dst_account)
            if commit:
                self._commit(src_account, dst_account, usd, amount, currency, ts, score)
        return {
            "risk_score": score,
            "risk_tier": self._tier(score),
            "flagged": score >= self.bundle.threshold,
            "threshold": self.bundle.threshold,
            "reasons": self._explain(features),
            "features": {k: float(v) for k, v in features.items()},
            "src_known": src_known,
            "dst_known": dst_known,
            "committed": commit,
        }

    def _commit(
        self,
        src: str,
        dst: str,
        usd: float,
        amount: float,
        currency: str,
        ts: pd.Timestamp,
        score: float,
    ) -> None:
        epoch = float((ts - _EPOCH).total_seconds())
        epoch = max(epoch, self.engine.latest_ts)  # keep the event log ordered
        self.engine.expire(epoch)
        self.engine.add_transaction(src, dst, usd, epoch)
        self.engine.maybe_refresh(epoch)
        self._commit_count += 1
        self._record_risk(src, dst, score)
        self._alert_rows.append(
            {
                "txn_id": f"LIVE{self._commit_count:08d}",
                "timestamp": ts.isoformat(),
                "src_account": src,
                "dst_account": dst,
                "amount": float(amount),
                "currency": currency,
                "risk_score": score,
            }
        )

    def alerts(self, limit: int = 25) -> list[dict[str, Any]]:
        """Highest-risk transactions among the newest window (as scored on arrival)."""
        with self._lock:
            top = heapq.nlargest(limit, self._alert_rows, key=lambda r: r["risk_score"])
        return [{**r, "flagged": r["risk_score"] >= self.bundle.threshold} for r in top]

    def ego_graph(self, account: str, radius: int = 2, max_nodes: int = 60) -> dict[str, Any]:
        """Node-link representation of an account's neighbourhood.

        Raises:
            KeyError: If the account is not in the live graph.
        """
        with self._lock:
            sub = self.engine.ego_graph(account, radius=radius, max_nodes=max_nodes)
            nodes = []
            for node, attrs in sub.nodes(data=True):
                info = self.engine.describe_node(node)
                nodes.append(
                    {
                        "id": node,
                        "is_center": node == account,
                        "distance": int(attrs.get("distance", 0)),
                        "in_degree": int(info["in_degree"]),
                        "out_degree": int(info["out_degree"]),
                        "pagerank": info["pagerank"],
                        "scc_size": int(info["scc_size"]),
                        "in_ring": bool(info["in_ring"]),
                        "risk_score": self._node_risk.get(node, 0.0),
                    }
                )
            edges = [
                {
                    "source": src,
                    "target": dst,
                    "total_usd": data["weight"],
                    "txn_count": data["txn_count"],
                    "risk_score": self._edge_risk.get((src, dst), 0.0),
                }
                for src, dst, data in sub.edges(data=True)
            ]
        return {"center": account, "nodes": nodes, "edges": edges}

    def model_info(self) -> dict[str, Any]:
        """Metrics and importances recorded at training time."""
        with self._lock:
            reference = {
                "as_of": self.as_of.isoformat(),
                "accounts": int(self.engine.graph.number_of_nodes()),
                "edges": int(self.engine.graph.number_of_edges()),
            }
        return {
            "backend": self.bundle.backend,
            "threshold": self.bundle.threshold,
            "window_days": self.bundle.window_days,
            "metrics": self.bundle.metrics,
            "feature_importance": self.bundle.feature_importance,
            "reference_graph": reference,
        }

    # ------------------------------------------------------------------ #
    # Reason codes
    # ------------------------------------------------------------------ #
    @staticmethod
    def _explain(f: dict[str, float]) -> list[str]:
        """Human-readable reason codes derived from the structural features."""
        reasons: list[str] = []
        if f["closes_cycle"] > 0:
            hops = int(f["min_return_hops"])
            reasons.append(
                f"Closes a circular flow: funds can travel from the receiver back to the "
                f"sender through {hops} hops ({hops + 1}-account ring)"
            )
        elif f["src_in_ring"] > 0 or f["dst_in_ring"] > 0:
            size = int(max(f["src_scc_size"], f["dst_scc_size"]))
            reasons.append(f"Account belongs to a {size}-account strongly connected ring")
        if f["src_out_degree"] >= FAN_THRESHOLD:
            reasons.append(
                f"Sender fans out to {int(f['src_out_degree'])} distinct counterparties"
            )
        if f["dst_in_degree"] >= FAN_THRESHOLD:
            reasons.append(
                f"Receiver collects from {int(f['dst_in_degree'])} distinct counterparties"
            )
        if f["near_threshold"] > 0:
            reasons.append(
                f"Amount sits just below the {config.REPORTING_THRESHOLD_USD:,.0f} USD "
                "reporting threshold (possible structuring)"
            )
        if f["src_flow_balance"] >= 0.8 and f["src_out_degree"] >= 3:
            reasons.append(
                f"Sender passes on about {f['src_flow_balance']:.0%} of the funds it "
                "receives (pass-through behaviour)"
            )
        return reasons
