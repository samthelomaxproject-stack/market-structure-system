from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class BacktestResult:
    r_multiples: List[float]

    @property
    def total_trades(self) -> int:
        return len(self.r_multiples)

    @property
    def wins(self) -> int:
        return sum(1 for r in self.r_multiples if r > 0)

    @property
    def losses(self) -> int:
        return sum(1 for r in self.r_multiples if r < 0)

    @property
    def win_rate(self) -> float:
        return self.wins / self.total_trades if self.total_trades else 0.0

    @property
    def average_win(self) -> float:
        vals = [r for r in self.r_multiples if r > 0]
        return float(np.mean(vals)) if vals else 0.0

    @property
    def average_loss(self) -> float:
        vals = [r for r in self.r_multiples if r < 0]
        return float(np.mean(vals)) if vals else 0.0

    @property
    def expectancy(self) -> float:
        return float(np.mean(self.r_multiples)) if self.r_multiples else 0.0


def _base_log_record(setup: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "setup_time": setup["time"],
        "entry_time": None,
        "exit_time": None,
        "side": setup["side"],
        "entry": float(setup["entry"]),
        "stop": float(setup["stop"]),
        "target": float(setup["target"]),
        "outcome": "not_triggered",
        "r_multiple": 0.0,
    }


def backtest(
    df: pd.DataFrame,
    setups: List[Dict],
    close_unresolved_at_end: bool = False,
) -> Tuple[List[float], List[Dict[str, Any]]]:
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be DataFrame")
    if not isinstance(setups, list):
        raise TypeError("setups must be list")

    data = df.reset_index(drop=True).copy()
    if "time" in data.columns:
        data["time"] = pd.to_datetime(data["time"], errors="coerce")
    else:
        data["time"] = data.index

    valid_setups: List[Dict[str, Any]] = []
    for setup in setups:
        required = {"time", "side", "entry", "stop", "target"}
        if not isinstance(setup, dict) or not required.issubset(setup.keys()):
            continue

        side = str(setup["side"]).lower()
        if side not in {"long", "short"}:
            continue

        setup_time = pd.to_datetime(setup["time"], errors="coerce")
        if pd.isna(setup_time):
            continue

        valid_setups.append(
            {
                "time": setup_time,
                "side": side,
                "entry": float(setup["entry"]),
                "stop": float(setup["stop"]),
                "target": float(setup["target"]),
            }
        )

    valid_setups.sort(key=lambda x: x["time"])

    r_multiples: List[float] = []
    trade_logs: List[Dict[str, Any]] = []
    active_until_idx = -1

    for setup in valid_setups:
        setup_log = _base_log_record(setup)

        idx = data.index[data["time"] >= setup["time"]]
        if len(idx) == 0:
            setup_log["outcome"] = "not_triggered"
            trade_logs.append(setup_log)
            continue

        signal_idx = int(idx[0])

        # While trade is active, ignore new setups for realism.
        if signal_idx <= active_until_idx:
            setup_log["outcome"] = "not_triggered"
            trade_logs.append(setup_log)
            continue

        active = False
        resolved = False
        entry_idx: Optional[int] = None
        entry = setup["entry"]
        stop = setup["stop"]
        target = setup["target"]
        side = setup["side"]

        for j in range(signal_idx + 1, len(data)):
            high_j = float(data.at[j, "high"])
            low_j = float(data.at[j, "low"])

            if not active:
                if low_j <= entry <= high_j:
                    active = True
                    entry_idx = j
                    setup_log["entry_time"] = data.at[j, "time"]
                else:
                    continue

            # Stop is checked before target by requirement.
            if side == "long":
                if low_j <= stop:
                    setup_log["exit_time"] = data.at[j, "time"]
                    setup_log["outcome"] = "loss"
                    setup_log["r_multiple"] = -1.0
                    r_multiples.append(-1.0)
                    resolved = True
                    active_until_idx = j
                    break
                if high_j >= target:
                    setup_log["exit_time"] = data.at[j, "time"]
                    setup_log["outcome"] = "win"
                    setup_log["r_multiple"] = 2.0
                    r_multiples.append(2.0)
                    resolved = True
                    active_until_idx = j
                    break
            else:
                if high_j >= stop:
                    setup_log["exit_time"] = data.at[j, "time"]
                    setup_log["outcome"] = "loss"
                    setup_log["r_multiple"] = -1.0
                    r_multiples.append(-1.0)
                    resolved = True
                    active_until_idx = j
                    break
                if low_j <= target:
                    setup_log["exit_time"] = data.at[j, "time"]
                    setup_log["outcome"] = "win"
                    setup_log["r_multiple"] = 2.0
                    r_multiples.append(2.0)
                    resolved = True
                    active_until_idx = j
                    break

        if not active:
            setup_log["outcome"] = "not_triggered"
        elif not resolved:
            if close_unresolved_at_end and entry_idx is not None and len(data) > 0:
                final_close = float(data.at[len(data) - 1, "close"])
                risk = (entry - stop) if side == "long" else (stop - entry)
                if risk > 0:
                    pnl = (final_close - entry) if side == "long" else (entry - final_close)
                    r_mult = pnl / risk
                else:
                    r_mult = 0.0
                setup_log["exit_time"] = data.at[len(data) - 1, "time"]
                setup_log["outcome"] = "unresolved"
                setup_log["r_multiple"] = float(r_mult)
                r_multiples.append(float(r_mult))
                active_until_idx = len(data) - 1
            else:
                setup_log["outcome"] = "unresolved"
                setup_log["r_multiple"] = 0.0
                active_until_idx = len(data) - 1

        trade_logs.append(setup_log)

    return r_multiples, trade_logs
