"""Feature preparation, training and evaluation of the illicit-flow classifier.

Pipeline
--------
1. The ledger is **replayed in time order** through a streaming
   :class:`~aml_graph.graph_engine.GraphEngine`. The topological features of
   each transaction are read from the graph as it was *before* that
   transaction (trailing ``window_days`` window), which is exactly what the
   real-time API sees. Training therefore never peeks at the future.
2. Transactional features (amount, time, currency, structuring flag) are added.
3. The first ``window_days`` are dropped as graph warm-up, and the remainder is
   split **chronologically** into train / validation / test.
4. Class imbalance is handled with ``scale_pos_weight = n_neg / n_pos``
   (XGBoost) or balanced class weights (scikit-learn fallback). The decision
   threshold is chosen on the validation set to maximise F1.
5. Metrics: PR-AUC (average precision) and F1 on the held-out test period,
   plus an ablation with transactional features only, to quantify the lift
   from graph features.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

from aml_graph import config
from aml_graph.data_pipeline import add_amount_usd, load_ledger
from aml_graph.graph_engine import GRAPH_FEATURES, GraphConfig, GraphEngine

LOGGER = logging.getLogger(__name__)

TABULAR_FEATURES: list[str] = [
    "amount_usd",
    "log_amount",
    "hour",
    "day_of_week",
    "is_weekend",
    "is_night",
    "near_threshold",
    "currency_code",
]
FEATURE_COLUMNS: list[str] = TABULAR_FEATURES + GRAPH_FEATURES
BACKENDS = ("xgboost", "sklearn")


# ---------------------------------------------------------------------- #
# Feature preparation
# ---------------------------------------------------------------------- #
def build_tabular_features(ledger: pd.DataFrame) -> pd.DataFrame:
    """Derive transactional features from raw ledger columns."""
    ledger = add_amount_usd(ledger)
    ts = pd.to_datetime(ledger["timestamp"])
    usd = ledger["amount_usd"].astype(float)
    threshold = config.REPORTING_THRESHOLD_USD
    codes = {c: i for i, c in enumerate(config.CURRENCIES)}
    return pd.DataFrame(
        {
            "amount_usd": usd,
            "log_amount": np.log1p(usd),
            "hour": ts.dt.hour.astype(float),
            "day_of_week": ts.dt.dayofweek.astype(float),
            "is_weekend": (ts.dt.dayofweek >= 5).astype(float),
            "is_night": (ts.dt.hour < 6).astype(float),
            "near_threshold": (
                (usd >= config.STRUCTURING_BAND * threshold) & (usd < threshold)
            ).astype(float),
            "currency_code": ledger["currency"].map(codes).astype(float),
        },
        index=ledger.index,
    )[TABULAR_FEATURES]


def build_feature_frame(
    ledger: pd.DataFrame, graph_config: GraphConfig | None = None
) -> pd.DataFrame:
    """Return the full model matrix for a ledger, built by causal replay.

    Extra columns: ``timestamp``, ``is_warm`` (False during the first
    ``window_days`` while the graph fills up) and, if present, ``is_illicit``.
    """
    graph_config = graph_config or GraphConfig()
    ledger = add_amount_usd(ledger).reset_index(drop=True)
    ledger["timestamp"] = pd.to_datetime(ledger["timestamp"])
    graph_features = GraphEngine(graph_config).replay(ledger)
    frame = pd.concat([build_tabular_features(ledger), graph_features], axis=1)
    frame["timestamp"] = ledger["timestamp"]
    warm_start = ledger["timestamp"].min() + pd.Timedelta(days=graph_config.window_days)
    frame["is_warm"] = ledger["timestamp"] >= warm_start
    if "is_illicit" in ledger.columns:
        frame["is_illicit"] = ledger["is_illicit"].to_numpy()
    return frame


def chronological_split(
    frame: pd.DataFrame, val_frac: float = 0.2, test_frac: float = 0.2
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Boolean masks (train, val, test): oldest -> newest, warm-up rows excluded."""
    warm = frame["is_warm"]
    ts = frame["timestamp"]
    val_start = ts[warm].quantile(1.0 - val_frac - test_frac)
    test_start = ts[warm].quantile(1.0 - test_frac)
    train = warm & (ts < val_start)
    val = warm & (ts >= val_start) & (ts < test_start)
    test = warm & (ts >= test_start)
    for name, mask in (("train", train), ("val", val), ("test", test)):
        if frame.loc[mask, "is_illicit"].nunique() < 2:
            raise ValueError(
                f"The {name} split does not contain both classes; use a longer ledger."
            )
    return train, val, test


# ---------------------------------------------------------------------- #
# Model construction and evaluation
# ---------------------------------------------------------------------- #
def make_classifier(backend: str, scale_pos_weight: float, seed: int) -> Any:
    """Instantiate an imbalance-aware gradient-boosted tree classifier."""
    if backend == "xgboost":
        try:
            from xgboost import XGBClassifier
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise RuntimeError(
                "xgboost is not installed. Run `pip install xgboost` or use "
                "`--backend sklearn` for a development fallback."
            ) from exc
        return XGBClassifier(
            n_estimators=600,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=2,
            reg_lambda=1.0,
            scale_pos_weight=scale_pos_weight,
            tree_method="hist",
            eval_metric="aucpr",
            early_stopping_rounds=40,
            random_state=seed,
            n_jobs=-1,
        )
    if backend == "sklearn":
        return HistGradientBoostingClassifier(
            max_iter=250,
            learning_rate=0.06,
            max_depth=5,
            class_weight="balanced",
            early_stopping=False,
            random_state=seed,
        )
    raise ValueError(f"Unknown backend {backend!r}; choose from {BACKENDS}")


def select_threshold(y_val: np.ndarray, proba_val: np.ndarray) -> float:
    """Probability cut-off that maximises F1 on the validation set."""
    precision, recall, thresholds = precision_recall_curve(y_val, proba_val)
    denom = np.clip(precision[:-1] + recall[:-1], 1e-12, None)
    f1 = 2 * precision[:-1] * recall[:-1] / denom
    return float(thresholds[int(np.argmax(f1))])


def evaluate(y_true: np.ndarray, proba: np.ndarray, threshold: float) -> dict[str, Any]:
    """PR-AUC, F1 and companion metrics at a fixed decision threshold."""
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "pr_auc": float(average_precision_score(y_true, proba)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, proba)),
        "threshold": float(threshold),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "n": int(len(y_true)),
        "positives": int(np.sum(y_true)),
        "prevalence": float(np.mean(y_true)),
    }


@dataclass
class ModelBundle:
    """Everything the serving layer needs, persisted as a single artifact."""

    model: Any
    backend: str
    feature_names: list[str]
    threshold: float
    graph_config: GraphConfig
    metrics: dict[str, Any] = field(default_factory=dict)
    feature_importance: dict[str, float] = field(default_factory=dict)

    @property
    def window_days(self) -> int:
        """Length of the trailing graph window the model was trained with."""
        return self.graph_config.window_days


def _fit(
    backend: str,
    features: list[str],
    frame: pd.DataFrame,
    masks: tuple[pd.Series, pd.Series, pd.Series],
    seed: int,
) -> tuple[Any, float, dict[str, Any]]:
    train, val, test = masks
    x_train, y_train = frame.loc[train, features], frame.loc[train, "is_illicit"].to_numpy()
    x_val, y_val = frame.loc[val, features], frame.loc[val, "is_illicit"].to_numpy()
    x_test, y_test = frame.loc[test, features], frame.loc[test, "is_illicit"].to_numpy()

    pos = max(int(y_train.sum()), 1)
    scale_pos_weight = float((len(y_train) - pos) / pos)
    clf = make_classifier(backend, scale_pos_weight, seed)
    if backend == "xgboost":
        clf.fit(x_train, y_train, eval_set=[(x_val, y_val)], verbose=False)
    else:
        clf.fit(x_train, y_train)

    threshold = select_threshold(y_val, clf.predict_proba(x_val)[:, 1])
    metrics = {
        "train_scale_pos_weight": scale_pos_weight,
        "validation": evaluate(y_val, clf.predict_proba(x_val)[:, 1], threshold),
        "test": evaluate(y_test, clf.predict_proba(x_test)[:, 1], threshold),
    }
    return clf, threshold, metrics


def _importance(
    clf: Any, backend: str, features: list[str], x_val: pd.DataFrame, y_val: np.ndarray, seed: int
) -> dict[str, float]:
    if backend == "xgboost":
        values = clf.feature_importances_
    else:
        values = permutation_importance(
            clf, x_val, y_val, scoring="average_precision", n_repeats=3, random_state=seed
        ).importances_mean
    ranked = sorted(zip(features, values, strict=True), key=lambda kv: kv[1], reverse=True)
    return {name: float(value) for name, value in ranked}


def train_model(
    ledger: pd.DataFrame,
    backend: str = "xgboost",
    seed: int = config.RANDOM_SEED,
    run_ablation: bool = True,
    graph_config: GraphConfig | None = None,
) -> ModelBundle:
    """Train, evaluate and package the classifier."""
    graph_config = graph_config or GraphConfig()
    LOGGER.info(
        "Replaying ledger through a %d-day streaming graph...", graph_config.window_days
    )
    frame = build_feature_frame(ledger, graph_config)
    masks = chronological_split(frame)
    train, val, test = masks
    LOGGER.info(
        "Rows: train=%d val=%d test=%d | illicit rate: train=%.2f%% test=%.2f%%",
        train.sum(),
        val.sum(),
        test.sum(),
        100 * frame.loc[train, "is_illicit"].mean(),
        100 * frame.loc[test, "is_illicit"].mean(),
    )

    clf, threshold, metrics = _fit(backend, FEATURE_COLUMNS, frame, masks, seed)
    metrics["backend"] = backend
    metrics["window_days"] = graph_config.window_days

    if run_ablation:
        _, _, tab_metrics = _fit(backend, TABULAR_FEATURES, frame, masks, seed)
        metrics["ablation_tabular_only"] = tab_metrics["test"]
        metrics["graph_feature_pr_auc_lift"] = (
            metrics["test"]["pr_auc"] - tab_metrics["test"]["pr_auc"]
        )

    importance = _importance(
        clf,
        backend,
        FEATURE_COLUMNS,
        frame.loc[val, FEATURE_COLUMNS],
        frame.loc[val, "is_illicit"].to_numpy(),
        seed,
    )
    return ModelBundle(
        model=clf,
        backend=backend,
        feature_names=list(FEATURE_COLUMNS),
        threshold=threshold,
        graph_config=graph_config,
        metrics=metrics,
        feature_importance=importance,
    )


def save_bundle(bundle: ModelBundle, model_path: Path, metrics_path: Path | None = None) -> None:
    """Persist the bundle (joblib) and, optionally, its metrics (JSON)."""
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, model_path)
    if metrics_path is not None:
        payload = {**bundle.metrics, "feature_importance": bundle.feature_importance}
        metrics_path.write_text(json.dumps(payload, indent=2))


def load_bundle(model_path: Path) -> ModelBundle:
    """Load a bundle written by :func:`save_bundle`."""
    bundle = joblib.load(model_path)
    if not isinstance(bundle, ModelBundle):
        raise TypeError(f"{model_path} does not contain a ModelBundle")
    return bundle


def _format_report(metrics: dict[str, Any]) -> str:
    test = metrics["test"]
    lines = [
        f"Backend            : {metrics['backend']}",
        f"Test PR-AUC        : {test['pr_auc']:.4f}  (random baseline = {test['prevalence']:.4f})",
        f"Test F1            : {test['f1']:.4f}  "
        f"(precision {test['precision']:.3f}, recall {test['recall']:.3f})",
        f"Test ROC-AUC       : {test['roc_auc']:.4f}",
        f"Threshold (val-F1) : {test['threshold']:.4f}",
        f"Confusion matrix   : {test['confusion_matrix']}",
    ]
    if "ablation_tabular_only" in metrics:
        tab = metrics["ablation_tabular_only"]
        lines.append(
            f"Tabular-only       : PR-AUC {tab['pr_auc']:.4f}, F1 {tab['f1']:.4f}  "
            f"-> graph features add {metrics['graph_feature_pr_auc_lift']:+.4f} PR-AUC"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    """CLI entry point: ``python -m aml_graph.model``."""
    parser = argparse.ArgumentParser(description="Train the AML illicit-flow classifier.")
    parser.add_argument("--ledger", type=Path, default=config.LEDGER_PATH)
    parser.add_argument("--out", type=Path, default=config.MODEL_PATH)
    parser.add_argument("--metrics-out", type=Path, default=config.METRICS_PATH)
    parser.add_argument("--backend", choices=BACKENDS, default="xgboost")
    parser.add_argument("--window-days", type=int, default=config.GRAPH_WINDOW_DAYS)
    parser.add_argument("--seed", type=int, default=config.RANDOM_SEED)
    parser.add_argument("--no-ablation", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ledger = load_ledger(args.ledger)
    bundle = train_model(
        ledger,
        backend=args.backend,
        graph_config=GraphConfig(window_days=args.window_days),
        seed=args.seed,
        run_ablation=not args.no_ablation,
    )
    save_bundle(bundle, args.out, args.metrics_out)
    print(_format_report(bundle.metrics))
    print(f"\nSaved model to {args.out} and metrics to {args.metrics_out}")


if __name__ == "__main__":
    # Re-import under the real module path. Running with ``python -m`` executes
    # this file as ``__main__``, which would make joblib pickle the bundle as
    # ``__main__.ModelBundle`` and break loading from the API process.
    from aml_graph.model import main as _main

    _main()
