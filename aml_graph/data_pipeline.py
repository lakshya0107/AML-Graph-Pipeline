"""Synthetic ledger generator with injected AML typologies.

The generator produces a realistic-looking transfer ledger made of:

* **Normal behaviour** - customer-to-merchant payments (Zipf-distributed
  merchant popularity), peer-to-peer transfers inside small contact lists,
  monthly payroll runs (a *legitimate* fan-out) and occasional large
  legitimate transfers such as rent or tuition (legitimate large amounts,
  some just under the reporting threshold).
* **Fan-out** (placement / structuring) - a hub is funded and immediately
  splits the money across many recipients, usually just below the
  reporting threshold.
* **Fan-in** (aggregation) - many mule accounts pay one collector, which
  forwards the pooled funds to a single exit account.
* **Circular flows** - value travels ``A -> B -> C -> A`` (rings of 3-6
  accounts, 1-3 laps, small "fee" shrinkage per hop).

Typology participants are drawn from the same pool as normal customers, so
illicit accounts also carry ordinary activity and labels are per
transaction, not per account.
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from aml_graph import config

LOGGER = logging.getLogger(__name__)

SECONDS_PER_DAY = 86_400
LEDGER_COLUMNS = [
    "txn_id",
    "timestamp",
    "src_account",
    "dst_account",
    "amount",
    "currency",
    "is_illicit",
    "typology",
]

# (offset_seconds, src, dst, amount_usd, typology)
Row = tuple[float, str, str, float, str]
_ROW_COLUMNS = ["offset", "src", "dst", "usd", "typology"]

_HOUR_WEIGHTS = np.array(
    [1, 1, 1, 1, 1, 2, 4, 6, 8, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 8, 7, 5, 3, 2],
    dtype=float,
)
_HOUR_WEIGHTS /= _HOUR_WEIGHTS.sum()
_CURRENCY_WEIGHTS = np.array([0.45, 0.20, 0.10, 0.25])


@dataclass(frozen=True)
class GeneratorConfig:
    """Knobs controlling the size and shape of the synthetic ledger."""

    n_customers: int = 2_500
    n_merchants: int = 150
    n_employers: int = 10
    n_normal_txns: int = 40_000
    typology_count: int = 30  # number of injected cases *per* typology
    start: str = "2026-01-01"
    days: int = 90
    seed: int = config.RANDOM_SEED


def add_amount_usd(ledger: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of ``ledger`` with an ``amount_usd`` column.

    Raises:
        ValueError: If a currency has no configured FX rate.
    """
    out = ledger.copy()
    if "amount_usd" not in out.columns:
        rates = out["currency"].map(config.FX_TO_USD)
        if rates.isna().any():
            bad = sorted(out.loc[rates.isna(), "currency"].unique())
            raise ValueError(f"Unsupported currencies: {bad}")
        out["amount_usd"] = out["amount"].astype(float) * rates.astype(float)
    return out


class LedgerGenerator:
    """Builds a labelled synthetic transaction ledger."""

    def __init__(self, cfg: GeneratorConfig | None = None) -> None:
        self.cfg = cfg or GeneratorConfig()
        self.rng = np.random.default_rng(self.cfg.seed)
        self._init_accounts()

    # ------------------------------------------------------------------ #
    # Accounts
    # ------------------------------------------------------------------ #
    def _init_accounts(self) -> None:
        cfg = self.cfg
        total = cfg.n_customers + cfg.n_merchants + cfg.n_employers
        # Neutral IDs, shuffled so an ID never reveals the account's role.
        ids = self.rng.permutation([f"ACC{i:06d}" for i in range(1, total + 1)])
        self.customers = ids[: cfg.n_customers]
        self.merchants = ids[cfg.n_customers : cfg.n_customers + cfg.n_merchants]
        self.employers = ids[cfg.n_customers + cfg.n_merchants :]
        home = self.rng.choice(config.CURRENCIES, size=total, p=_CURRENCY_WEIGHTS)
        self.home_currency: dict[str, str] = dict(zip(ids.tolist(), home.tolist(), strict=True))
        activity = self.rng.lognormal(0.0, 1.0, cfg.n_customers)
        self._activity = activity / activity.sum()

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _daytime_offsets(self, n: int) -> np.ndarray:
        day = self.rng.integers(0, self.cfg.days, size=n)
        hour = self.rng.choice(24, size=n, p=_HOUR_WEIGHTS)
        second = self.rng.integers(0, 3_600, size=n)
        return day * SECONDS_PER_DAY + hour * 3_600 + second

    @staticmethod
    def _frame(
        offsets: np.ndarray,
        src: np.ndarray,
        dst: np.ndarray,
        usd: np.ndarray,
        typology: str = "normal",
    ) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "offset": offsets.astype(float),
                "src": src,
                "dst": dst,
                "usd": usd,
                "typology": typology,
            }
        )

    # ------------------------------------------------------------------ #
    # Normal behaviour
    # ------------------------------------------------------------------ #
    def _merchant_payments(self, n: int) -> pd.DataFrame:
        rng = self.rng
        popularity = 1.0 / np.arange(1, len(self.merchants) + 1)
        popularity /= popularity.sum()
        src = rng.choice(self.customers, size=n, p=self._activity)
        dst = rng.choice(self.merchants, size=n, p=popularity)
        usd = np.clip(rng.lognormal(np.log(35), 0.9, n), 1.0, 3_000.0)
        return self._frame(self._daytime_offsets(n), src, dst, usd)

    def _p2p_transfers(self, n: int) -> pd.DataFrame:
        rng = self.rng
        n_cust = len(self.customers)
        n_contacts = rng.integers(1, 5, size=n_cust)
        shifts = rng.integers(1, n_cust, size=(n_cust, 4))
        contacts = (np.arange(n_cust)[:, None] + shifts) % n_cust  # never self
        sender = rng.choice(n_cust, size=n, p=self._activity)
        slot = (rng.random(n) * n_contacts[sender]).astype(int)
        receiver = contacts[sender, slot]
        usd = np.clip(rng.lognormal(np.log(80), 1.0, n), 1.0, 5_000.0)
        return self._frame(
            self._daytime_offsets(n),
            self.customers[sender],
            self.customers[receiver],
            usd,
        )

    def _large_legit_transfers(self, n: int) -> pd.DataFrame:
        """Rent, tuition, property deposits: large but legitimate."""
        rng = self.rng
        n_cust = len(self.customers)
        sender = rng.choice(n_cust, size=n, p=self._activity)
        to_merchant = rng.random(n) < 0.5
        peer = (sender + rng.integers(1, n_cust, size=n)) % n_cust
        dst = np.where(
            to_merchant,
            rng.choice(self.merchants, size=n),
            self.customers[peer],
        )
        usd = np.clip(rng.lognormal(np.log(6_000), 0.8, n), 1_000.0, 60_000.0)
        return self._frame(self._daytime_offsets(n), self.customers[sender], dst, usd)

    def _payroll(self) -> pd.DataFrame:
        """Monthly payroll: a legitimate high-out-degree pattern."""
        rng = self.rng
        pay_days = [d for d in (30, 58, 89) if d < self.cfg.days]
        frames: list[pd.DataFrame] = []
        for employer in self.employers:
            n_staff = int(rng.integers(25, 60))
            staff = rng.choice(self.customers, size=n_staff, replace=False)
            base = np.clip(rng.lognormal(np.log(2_500), 0.5, n_staff), 800, 12_000)
            for day in pay_days:
                offsets = (
                    day * SECONDS_PER_DAY
                    + 9 * 3_600
                    + rng.integers(0, 3 * 3_600, size=n_staff)
                )
                usd = base * rng.uniform(0.98, 1.02, n_staff)
                src = np.full(n_staff, employer)
                frames.append(self._frame(offsets, src, staff, usd))
        if not frames:
            return pd.DataFrame(columns=_ROW_COLUMNS)
        return pd.concat(frames, ignore_index=True)

    # ------------------------------------------------------------------ #
    # AML typologies
    # ------------------------------------------------------------------ #
    def _fan_out(self) -> list[Row]:
        """source -> hub -> many recipients, usually just under the threshold."""
        rng = self.rng
        n_out = int(rng.integers(10, 26))
        members = rng.choice(self.customers, size=n_out + 2, replace=False)
        source, hub, recipients = members[0], members[1], members[2:]
        structured = rng.random() < 0.7
        if structured:
            splits = rng.uniform(9_000, 9_950, n_out)
        else:
            splits = rng.uniform(500, 5_000, n_out)
        t0 = rng.uniform(0, (self.cfg.days - 4) * SECONDS_PER_DAY)
        funding = splits.sum() * rng.uniform(1.0, 1.02)
        rows: list[Row] = [(t0, str(source), str(hub), float(funding), "fan_out")]
        start = t0 + rng.uniform(1_800, 6 * 3_600)
        offsets = start + np.sort(rng.uniform(0, 3 * 3_600, n_out))
        for off, dst, usd in zip(offsets, recipients, splits, strict=True):
            rows.append((float(off), str(hub), str(dst), float(usd), "fan_out"))
        return rows

    def _fan_in(self) -> list[Row]:
        """many mules -> collector -> exit account."""
        rng = self.rng
        n_in = int(rng.integers(10, 26))
        members = rng.choice(self.customers, size=n_in + 2, replace=False)
        collector, exit_acc, mules = members[0], members[1], members[2:]
        low_value = rng.random() < 0.3  # camouflage: normal-looking amounts
        if low_value:
            deposits = rng.uniform(20, 250, n_in)
        else:
            deposits = rng.uniform(1_000, 4_500, n_in)
        t0 = rng.uniform(0, (self.cfg.days - 5) * SECONDS_PER_DAY)
        offsets = t0 + np.sort(rng.uniform(0, 2 * SECONDS_PER_DAY, n_in))
        rows: list[Row] = [
            (float(off), str(mule), str(collector), float(usd), "fan_in")
            for off, mule, usd in zip(offsets, mules, deposits, strict=True)
        ]
        exit_time = offsets[-1] + rng.uniform(3_600, 12 * 3_600)
        payout = float(deposits.sum() * 0.97)
        rows.append((float(exit_time), str(collector), str(exit_acc), payout, "fan_in"))
        return rows

    def _circular(self) -> list[Row]:
        """A -> B -> C -> A with a small fee deducted at every hop."""
        rng = self.rng
        ring_size = int(rng.integers(3, 7))
        laps = int(rng.integers(1, 4))
        ring = rng.choice(self.customers, size=ring_size, replace=False)
        if rng.random() < 0.25:  # camouflage: low-value ring
            amount = float(rng.uniform(800, 6_000))
        else:
            amount = float(np.clip(rng.lognormal(np.log(40_000), 0.6), 10_000, 250_000))
        t = float(rng.uniform(0, (self.cfg.days - 5) * SECONDS_PER_DAY))
        rows: list[Row] = []
        for _ in range(laps):
            for i in range(ring_size):
                src, dst = str(ring[i]), str(ring[(i + 1) % ring_size])
                rows.append((t, src, dst, amount, "circular"))
                amount *= 1.0 - float(rng.uniform(0.005, 0.03))
                t += float(rng.uniform(300, 4 * 3_600))
        return rows

    # ------------------------------------------------------------------ #
    # Assembly
    # ------------------------------------------------------------------ #
    def generate(self) -> pd.DataFrame:
        """Generate the full labelled ledger, sorted by timestamp."""
        cfg = self.cfg
        n = cfg.n_normal_txns
        frames = [
            self._merchant_payments(int(n * 0.60)),
            self._p2p_transfers(int(n * 0.35)),
            self._large_legit_transfers(int(n * 0.05)),
            self._payroll(),
        ]
        typology_rows: list[Row] = []
        for _ in range(cfg.typology_count):
            typology_rows += self._fan_out()
            typology_rows += self._fan_in()
            typology_rows += self._circular()
        frames.append(pd.DataFrame(typology_rows, columns=_ROW_COLUMNS))

        raw = pd.concat(frames, ignore_index=True)
        raw = raw.sort_values("offset", kind="stable").reset_index(drop=True)

        currency = raw["src"].map(self.home_currency)
        fx = currency.map(config.FX_TO_USD).astype(float)
        ledger = pd.DataFrame(
            {
                "txn_id": [f"TXN{i:08d}" for i in range(1, len(raw) + 1)],
                "timestamp": (
                    pd.Timestamp(cfg.start) + pd.to_timedelta(raw["offset"], unit="s")
                ).dt.floor("s"),
                "src_account": raw["src"],
                "dst_account": raw["dst"],
                "amount": (raw["usd"] / fx).round(2),
                "currency": currency,
                "is_illicit": (raw["typology"] != "normal").astype(int),
                "typology": raw["typology"],
            }
        )[LEDGER_COLUMNS]
        if (ledger["src_account"] == ledger["dst_account"]).any():
            raise RuntimeError("Generator produced a self-transfer")
        return ledger


def generate_ledger(cfg: GeneratorConfig | None = None) -> pd.DataFrame:
    """Convenience wrapper around :class:`LedgerGenerator`."""
    return LedgerGenerator(cfg).generate()


def save_ledger(ledger: pd.DataFrame, path: Path) -> None:
    """Write the ledger to CSV, creating parent directories."""
    path.parent.mkdir(parents=True, exist_ok=True)
    ledger.to_csv(path, index=False)


def load_ledger(path: Path) -> pd.DataFrame:
    """Read a ledger CSV with parsed timestamps."""
    return pd.read_csv(path, parse_dates=["timestamp"])


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a synthetic AML ledger.")
    parser.add_argument("--out", type=Path, default=config.LEDGER_PATH)
    parser.add_argument("--customers", type=int, default=GeneratorConfig.n_customers)
    parser.add_argument("--normal-txns", type=int, default=GeneratorConfig.n_normal_txns)
    parser.add_argument(
        "--typology-count",
        type=int,
        default=GeneratorConfig.typology_count,
        help="Injected cases per typology (fan-out, fan-in, circular).",
    )
    parser.add_argument("--days", type=int, default=GeneratorConfig.days)
    parser.add_argument("--seed", type=int, default=config.RANDOM_SEED)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """CLI entry point: ``python -m aml_graph.data_pipeline``."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = _parse_args(argv)
    cfg = GeneratorConfig(
        n_customers=args.customers,
        n_normal_txns=args.normal_txns,
        typology_count=args.typology_count,
        days=args.days,
        seed=args.seed,
    )
    ledger = generate_ledger(cfg)
    save_ledger(ledger, args.out)
    LOGGER.info(
        "Wrote %d transactions to %s (illicit rate %.2f%%)",
        len(ledger),
        args.out,
        100 * ledger["is_illicit"].mean(),
    )
    LOGGER.info("Typology counts:\n%s", ledger["typology"].value_counts().to_string())


if __name__ == "__main__":
    main()
