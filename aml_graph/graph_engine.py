"""Graph topology engine built on NetworkX.

Accounts are nodes of a :class:`networkx.DiGraph`; every ordered account pair
with at least one transfer inside the trailing window is a weighted edge
(``weight`` = total USD moved, ``txn_count`` = number of transfers).

The engine is a **streaming** graph: transfers are appended with
:meth:`GraphEngine.add_transaction` and expire after ``window_days``. The
features of a transfer are always computed from the graph *as it was before
that transfer*, whether it is a live API request or a row of the historical
ledger being replayed with :meth:`GraphEngine.replay`. Using one code path for
both is what keeps training and serving consistent and prevents the model
from seeing the future during training.

Structural features
-------------------
Per node (live unless noted):

* ``in_degree`` / ``out_degree`` - distinct counterparties (fan-in / fan-out)
* ``flow_balance`` - ``min(in, out) / max(in, out)`` in USD; near 1 means the
  account forwards what it receives (pass-through / mule behaviour)
* ``pagerank`` (cached) - flow-weighted importance
* ``scc_size`` / ``in_ring`` (cached) - Strongly Connected Component
  membership. An SCC with ``ring_min_size``..``ring_max_size`` accounts is
  flagged as a circular-flow ring. Giant SCCs are *not* flagged: in a dense
  graph almost everybody ends up in one, so the size cap keeps the flag
  meaningful.

Per transfer:

* ``closes_cycle`` / ``min_return_hops`` (live) - is there a path of at least
  two hops from the receiver back to the sender, i.e. would this transfer
  close a ring of three or more accounts? Direct reciprocal transfers
  (A <-> B) are common and deliberately ignored.
* ``same_ring`` - sender and receiver are in the same ring-sized SCC; when the
  transfer closes a cycle the *would-be* merged SCC is used.
* ``pair_txn_count`` / ``pair_total_usd`` - history between the two accounts.

PageRank and SCC labels are O(V + E) to compute, so they are cached and
refreshed every ``refresh_hours``; everything else is incremental.
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass

import networkx as nx
import numpy as np
import pandas as pd

from aml_graph import config
from aml_graph.data_pipeline import add_amount_usd

LOGGER = logging.getLogger(__name__)

GRAPH_FEATURES: list[str] = [
    "src_out_degree",
    "src_in_degree",
    "src_pagerank",
    "src_scc_size",
    "src_in_ring",
    "src_flow_balance",
    "dst_out_degree",
    "dst_in_degree",
    "dst_pagerank",
    "dst_scc_size",
    "dst_in_ring",
    "dst_flow_balance",
    "same_ring",
    "closes_cycle",
    "min_return_hops",
    "pair_txn_count",
    "pair_total_usd",
]

_EPOCH = pd.Timestamp("1970-01-01")


def to_epoch_seconds(timestamps: pd.Series) -> np.ndarray:
    """Convert a datetime Series to float seconds since the Unix epoch."""
    return (pd.to_datetime(timestamps) - _EPOCH).dt.total_seconds().to_numpy(dtype=float)


@dataclass(frozen=True)
class GraphConfig:
    """Parameters for topology feature extraction."""

    window_days: int = config.GRAPH_WINDOW_DAYS
    refresh_hours: float = 6.0
    ring_min_size: int = 3
    ring_max_size: int = 12
    max_cycle_hops: int = 5
    pagerank_alpha: float = 0.85


class GraphEngine:
    """Streaming transaction graph with structural feature extraction."""

    def __init__(self, config: GraphConfig | None = None) -> None:
        self.config = config or GraphConfig()
        self.graph: nx.DiGraph = nx.DiGraph()
        self.latest_ts: float = float("-inf")
        self._events: deque[tuple[float, str, str, float]] = deque()
        self._pagerank: dict[str, float] = {}
        self._scc_id: dict[str, int] = {}
        self._scc_size: dict[str, int] = {}
        self._last_refresh: float = float("-inf")

    # ------------------------------------------------------------------ #
    # Streaming updates
    # ------------------------------------------------------------------ #
    def add_transaction(self, src: str, dst: str, amount_usd: float, ts: float) -> None:
        """Append a transfer to the graph (self-transfers are ignored)."""
        self.latest_ts = max(self.latest_ts, ts)
        if src == dst:
            return
        if self.graph.has_edge(src, dst):
            data = self.graph.edges[src, dst]
            data["weight"] += amount_usd
            data["txn_count"] += 1
        else:
            self.graph.add_edge(src, dst, weight=float(amount_usd), txn_count=1)
        self._events.append((ts, src, dst, float(amount_usd)))

    def expire(self, now: float) -> None:
        """Drop transfers older than the trailing window ending at ``now``."""
        cutoff = now - self.config.window_days * 86_400.0
        graph = self.graph
        while self._events and self._events[0][0] <= cutoff:
            _, src, dst, usd = self._events.popleft()
            data = graph.edges[src, dst]
            data["txn_count"] -= 1
            data["weight"] = max(data["weight"] - usd, 0.0)
            if data["txn_count"] <= 0:
                graph.remove_edge(src, dst)
                for node in (src, dst):
                    if node in graph and graph.degree(node) == 0:
                        graph.remove_node(node)

    def maybe_refresh(self, now: float) -> None:
        """Refresh the cached PageRank / SCC labels if they are stale."""
        if now - self._last_refresh >= self.config.refresh_hours * 3_600.0:
            self.refresh(now)

    def refresh(self, now: float | None = None) -> None:
        """Recompute the cached PageRank and SCC labels from the live graph."""
        graph = self.graph
        self._last_refresh = self.latest_ts if now is None else now
        if graph.number_of_nodes() == 0:
            self._pagerank, self._scc_id, self._scc_size = {}, {}, {}
            return
        self._pagerank = nx.pagerank(
            graph, alpha=self.config.pagerank_alpha, weight="weight"
        )
        scc_id: dict[str, int] = {}
        scc_size: dict[str, int] = {}
        for comp_id, comp in enumerate(nx.strongly_connected_components(graph)):
            for node in comp:
                scc_id[node] = comp_id
                scc_size[node] = len(comp)
        self._scc_id, self._scc_size = scc_id, scc_size

    # ------------------------------------------------------------------ #
    # Feature extraction
    # ------------------------------------------------------------------ #
    def _node_row(self, account: str) -> dict[str, float]:
        graph = self.graph
        if account in graph:
            in_deg = float(graph.in_degree(account))
            out_deg = float(graph.out_degree(account))
            in_str = float(graph.in_degree(account, weight="weight"))
            out_str = float(graph.out_degree(account, weight="weight"))
        else:
            in_deg = out_deg = in_str = out_str = 0.0
        hi, lo = max(in_str, out_str), min(in_str, out_str)
        size = float(self._scc_size.get(account, 1))
        cfg = self.config
        return {
            "in_degree": in_deg,
            "out_degree": out_deg,
            "flow_balance": lo / hi if hi > 0 else 0.0,
            "pagerank": float(self._pagerank.get(account, 0.0)),
            "scc_size": size,
            "in_ring": float(cfg.ring_min_size <= size <= cfg.ring_max_size),
        }

    def _return_path(self, src: str, dst: str) -> list[str] | None:
        """Shortest path ``dst -> ... -> src`` of at least two hops, if any.

        Such a path means a transfer ``src -> dst`` closes a cycle with three
        or more accounts. A direct ``dst -> src`` edge is ignored.
        """
        graph = self.graph
        if src not in graph or dst not in graph:
            return None
        if graph.out_degree(dst) == 0 or graph.in_degree(src) == 0:
            return None
        view = (
            nx.restricted_view(graph, [], [(dst, src)])
            if graph.has_edge(dst, src)
            else graph
        )
        paths = nx.single_source_shortest_path(view, dst, cutoff=self.config.max_cycle_hops)
        path = paths.get(src)
        if path is None or len(path) < 3:
            return None
        return path

    def _merged_scc_size(self, path: list[str]) -> int:
        """Size of the SCC that would exist if ``path`` were closed into a loop."""
        seen: set[int] = set()
        total = 0
        for node in path:
            comp = self._scc_id.get(node)
            if comp is None:  # added since the last refresh: its own component
                total += 1
            elif comp not in seen:
                seen.add(comp)
                total += self._scc_size[node]
        return total

    def features_for(self, src: str, dst: str, amount_usd: float) -> dict[str, float]:
        """:data:`GRAPH_FEATURES` for a transfer *not yet* added to the graph.

        The graph is not modified. Degrees and pair statistics already
        include the transfer being scored, so a first-time pair counts as one
        more counterparty and one transaction.
        """
        cfg = self.config
        s, d = self._node_row(src), self._node_row(dst)
        has_edge = self.graph.has_edge(src, dst)
        existing = self.graph.edges[src, dst] if has_edge else {"weight": 0.0, "txn_count": 0}

        src_scc, dst_scc = s["scc_size"], d["scc_size"]
        src_ring, dst_ring = s["in_ring"], d["in_ring"]
        src_comp, dst_comp = self._scc_id.get(src), self._scc_id.get(dst)
        same_ring = float(src_comp is not None and src_comp == dst_comp and src_ring > 0)

        path = self._return_path(src, dst)
        hops = len(path) - 1 if path else 0
        if path:
            merged = self._merged_scc_size(path)
            src_scc = dst_scc = float(merged)
            src_ring = dst_ring = float(cfg.ring_min_size <= merged <= cfg.ring_max_size)
            same_ring = src_ring

        return {
            "src_out_degree": s["out_degree"] + (0.0 if has_edge else 1.0),
            "src_in_degree": s["in_degree"],
            "src_pagerank": s["pagerank"],
            "src_scc_size": src_scc,
            "src_in_ring": src_ring,
            "src_flow_balance": s["flow_balance"],
            "dst_out_degree": d["out_degree"],
            "dst_in_degree": d["in_degree"] + (0.0 if has_edge else 1.0),
            "dst_pagerank": d["pagerank"],
            "dst_scc_size": dst_scc,
            "dst_in_ring": dst_ring,
            "dst_flow_balance": d["flow_balance"],
            "same_ring": same_ring,
            "closes_cycle": float(hops > 0),
            "min_return_hops": float(hops),
            "pair_txn_count": float(existing["txn_count"] + 1),
            "pair_total_usd": float(existing["weight"] + amount_usd),
        }

    # ------------------------------------------------------------------ #
    # Historical replay (training / alert queue)
    # ------------------------------------------------------------------ #
    def replay(self, ledger: pd.DataFrame) -> pd.DataFrame:
        """Stream a ledger through the graph and return per-row features.

        Rows are processed in timestamp order. Each row's features are read
        *before* the row is added, so they only reflect earlier transfers
        inside the trailing window. The engine is left in the state it has
        after the last transfer.

        Returns:
            A frame with :data:`GRAPH_FEATURES` columns, aligned to
            ``ledger.index``.
        """
        ledger = add_amount_usd(ledger)
        epoch = to_epoch_seconds(ledger["timestamp"])
        order = np.argsort(epoch, kind="stable")
        src = ledger["src_account"].astype(str).to_numpy()
        dst = ledger["dst_account"].astype(str).to_numpy()
        usd = ledger["amount_usd"].to_numpy(dtype=float)

        records: list[dict[str, float]] = []
        for i in order:
            ts = float(epoch[i])
            self.latest_ts = max(self.latest_ts, ts)
            self.expire(ts)
            self.maybe_refresh(ts)
            records.append(self.features_for(src[i], dst[i], usd[i]))
            self.add_transaction(src[i], dst[i], usd[i], ts)
        self.refresh()

        features = pd.DataFrame(records, index=ledger.index[order])
        return features.loc[ledger.index, GRAPH_FEATURES]

    # ------------------------------------------------------------------ #
    # Investigation helpers
    # ------------------------------------------------------------------ #
    def has_account(self, account: str) -> bool:
        """Whether ``account`` is in the live graph."""
        return account in self.graph

    def describe_node(self, account: str) -> dict[str, float]:
        """Structural summary of one account."""
        return self._node_row(account)

    def ego_graph(self, account: str, radius: int = 2, max_nodes: int = 60) -> nx.DiGraph:
        """Neighbourhood of ``account`` for visual inspection.

        Neighbourhoods are explored ignoring edge direction. Hubs (merchants,
        payroll accounts) can have thousands of neighbours, so the result is
        capped at ``max_nodes``, keeping the nearest and then the most
        important nodes.

        Raises:
            KeyError: If the account is not in the graph.
        """
        if account not in self.graph:
            raise KeyError(account)
        undirected = self.graph.to_undirected(as_view=True)
        distances = nx.single_source_shortest_path_length(undirected, account, cutoff=radius)
        ranked = sorted(
            distances, key=lambda n: (distances[n], -self._pagerank.get(n, 0.0))
        )
        keep = ranked[:max_nodes]
        if account not in keep:
            keep[-1] = account
        sub = self.graph.subgraph(keep).copy()
        nx.set_node_attributes(sub, {n: distances[n] for n in sub.nodes}, "distance")
        return sub
