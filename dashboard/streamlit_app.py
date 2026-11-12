"""Streamlit investigation console for compliance analysts.

Start the API first (``uvicorn aml_graph.app:app``), then::

    streamlit run dashboard/streamlit_app.py

The API location can be changed with the ``AML_API_URL`` environment variable.
"""

from __future__ import annotations

import os
from typing import Any

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import requests
import streamlit as st
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

API_URL = os.environ.get("AML_API_URL", "http://localhost:8000").rstrip("/")
TIMEOUT_S = 30
VIEWS = ["Alert queue", "Score a transfer", "Account ego-graph", "Model"]
TIER_RENDERERS = {"HIGH": st.error, "MEDIUM": st.warning, "LOW": st.success}

st.set_page_config(page_title="AML Investigation Console", layout="wide")


def call_api(method: str, path: str, **kwargs: Any) -> tuple[Any, str | None]:
    """Call the backend and return ``(data, error_message)``."""
    try:
        response = requests.request(method, f"{API_URL}{path}", timeout=TIMEOUT_S, **kwargs)
    except requests.RequestException as exc:
        return None, f"Cannot reach the API at {API_URL}: {exc}"
    if response.status_code >= 400:
        try:
            detail = response.json().get("detail", response.text)
        except ValueError:
            detail = response.text
        return None, f"API error {response.status_code}: {detail}"
    return response.json(), None


@st.cache_data(ttl=30, show_spinner=False)
def fetch_alerts(limit: int) -> tuple[Any, str | None]:
    return call_api("GET", "/alerts", params={"limit": limit})


def open_account(account: str) -> None:
    """Callback: jump to the ego-graph view for ``account``."""
    st.session_state["account_input"] = account
    st.session_state["view"] = "Account ego-graph"


def draw_ego_graph(payload: dict[str, Any]) -> plt.Figure:
    """Render an ego-graph payload. Colour = risk, diamond = ring-sized SCC."""
    graph = nx.DiGraph()
    for node in payload["nodes"]:
        graph.add_node(node["id"], **node)
    for edge in payload["edges"]:
        graph.add_edge(edge["source"], edge["target"], **edge)

    pos = nx.spring_layout(graph, seed=7, k=1.5 / np.sqrt(max(len(graph), 1)))
    cmap = plt.get_cmap("RdYlGn_r")
    norm = Normalize(vmin=0.0, vmax=1.0)
    fig, ax = plt.subplots(figsize=(9, 7))

    edges = list(graph.edges(data=True))
    if edges:
        nx.draw_networkx_edges(
            graph,
            pos,
            ax=ax,
            edge_color=[cmap(norm(d["risk_score"])) for _, _, d in edges],
            width=[0.6 + 0.5 * np.log10(d["total_usd"] + 1) for _, _, d in edges],
            arrows=True,
            arrowsize=14,
            connectionstyle="arc3,rad=0.12",
            alpha=0.85,
        )
    for shape, in_ring in (("o", False), ("D", True)):
        nodes = [n for n in graph if graph.nodes[n]["in_ring"] == in_ring]
        if not nodes:
            continue
        nx.draw_networkx_nodes(
            graph,
            pos,
            nodelist=nodes,
            ax=ax,
            node_shape=shape,
            node_color=[graph.nodes[n]["risk_score"] for n in nodes],
            cmap=cmap,
            vmin=0.0,
            vmax=1.0,
            node_size=[900 if graph.nodes[n]["is_center"] else 380 for n in nodes],
            edgecolors=["black" if graph.nodes[n]["is_center"] else "#555" for n in nodes],
            linewidths=[2.5 if graph.nodes[n]["is_center"] else 0.6 for n in nodes],
        )
    nx.draw_networkx_labels(graph, pos, labels={n: n[-6:] for n in graph}, font_size=7, ax=ax)
    fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=ax, label="Risk score", shrink=0.7)
    ax.set_axis_off()
    fig.tight_layout()
    return fig


def view_alerts() -> None:
    st.subheader("Highest-risk recent transactions")
    limit = st.slider("Rows", 5, 100, 25)
    alerts, error = fetch_alerts(limit)
    if error:
        st.error(error)
        return
    frame = pd.DataFrame(alerts)
    st.dataframe(
        frame,
        use_container_width=True,
        hide_index=True,
        column_config={"risk_score": st.column_config.ProgressColumn(
            "risk_score", min_value=0.0, max_value=1.0, format="%.3f"
        )},
    )
    accounts = sorted(set(frame["src_account"]) | set(frame["dst_account"]))
    chosen = st.selectbox("Investigate an account", accounts)
    st.button("Open ego-graph", on_click=open_account, args=(chosen,))


def view_score() -> None:
    st.subheader("Evaluate a transfer")
    with st.form("score"):
        left, right = st.columns(2)
        src = left.text_input("Source account", placeholder="ACC000123")
        dst = right.text_input("Destination account", placeholder="ACC000456")
        amount = left.number_input("Amount", min_value=0.01, value=9500.0, step=100.0)
        currency = right.selectbox("Currency", ["USD", "EUR", "GBP", "INR"])
        commit = st.checkbox("Add to the live graph after scoring")
        submitted = st.form_submit_button("Analyze")
    if not submitted:
        return
    result, error = call_api(
        "POST",
        "/analyze_transaction",
        params={"commit": str(commit).lower()},
        json={
            "src_account": src.strip(),
            "dst_account": dst.strip(),
            "amount": amount,
            "currency": currency,
        },
    )
    if error:
        st.error(error)
        return
    TIER_RENDERERS[result["risk_tier"]](
        f"{result['risk_tier']} risk - score {result['risk_score']:.1%} "
        f"(alert threshold {result['threshold']:.1%})"
    )
    st.progress(min(max(result["risk_score"], 0.0), 1.0))
    if not (result["src_known"] and result["dst_known"]):
        st.caption("At least one account has no activity in the current graph window.")
    st.markdown("**Reason codes**")
    for reason in result["reasons"] or ["No structural red flags detected"]:
        st.write(f"- {reason}")
    with st.expander("Model inputs"):
        st.json(result["features"])


def view_ego() -> None:
    st.subheader("Account ego-graph")
    left, right = st.columns([3, 1])
    account = left.text_input("Account", key="account_input")
    radius = right.slider("Hops", 1, 3, 2)
    if not account:
        st.info("Enter an account ID, or open one from the alert queue.")
        return
    payload, error = call_api(
        "GET", f"/ego_graph/{account.strip()}", params={"radius": radius}
    )
    if error:
        st.error(error)
        return
    centre = next(n for n in payload["nodes"] if n["is_center"])
    cols = st.columns(4)
    cols[0].metric("Max risk", f"{centre['risk_score']:.1%}")
    cols[1].metric("In / out degree", f"{centre['in_degree']} / {centre['out_degree']}")
    cols[2].metric("PageRank", f"{centre['pagerank']:.4f}")
    cols[3].metric("SCC size", centre["scc_size"], "ring" if centre["in_ring"] else None)
    st.pyplot(draw_ego_graph(payload))
    st.caption(
        "Colour = risk score. Diamonds = accounts in a ring-sized strongly connected component."
    )
    st.dataframe(pd.DataFrame(payload["edges"]), use_container_width=True, hide_index=True)


def view_model() -> None:
    st.subheader("Model")
    info, error = call_api("GET", "/model_info")
    if error:
        st.error(error)
        return
    test = info["metrics"]["test"]
    cols = st.columns(4)
    cols[0].metric("Test PR-AUC", f"{test['pr_auc']:.3f}")
    cols[1].metric("Test F1", f"{test['f1']:.3f}")
    cols[2].metric("Precision / recall", f"{test['precision']:.2f} / {test['recall']:.2f}")
    cols[3].metric("Prevalence", f"{test['prevalence']:.2%}")
    st.caption(f"Backend: {info['backend']} | Graph window: {info['window_days']} days")
    st.markdown("**Feature importance**")
    st.bar_chart(pd.Series(info["feature_importance"]).sort_values())
    st.json(info["reference_graph"])


def main() -> None:
    st.title("AML Investigation Console")
    st.sidebar.radio("View", VIEWS, key="view")
    {
        "Alert queue": view_alerts,
        "Score a transfer": view_score,
        "Account ego-graph": view_ego,
        "Model": view_model,
    }[st.session_state["view"]]()


main()
