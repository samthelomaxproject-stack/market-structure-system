from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class ICCBacktestResult:
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


def _base_log(setup: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "setup_time": setup["setup_time"],
        "entry_time": None,
        "exit_time": None,
        "side": setup["side"],
        "entry": float(setup["entry"]),
        "stop": float(setup["stop"]),
        "target": float(setup["target"]),
        "outcome": "not_triggered",
        "r_multiple": 0.0,
        "bars_to_entry": None,
        "bars_in_trade": None,
    }


def backtest_icc(
    df: pd.DataFrame,
    setups: List[Dict[str, Any]],
    allow_overlapping_trades: bool = False,
    force_close_on_final_bar: bool = False,
) -> Tuple[List[float], List[Dict[str, Any]]]:
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a DataFrame")
    if not isinstance(setups, list):
        raise TypeError("setups must be a list")

    data = df.copy().reset_index(drop=True)
    if "time" in data.columns:
        data["time"] = pd.to_datetime(data["time"], errors="coerce")
    else:
        raise ValueError("df must contain a time column")

    valid_setups: List[Dict[str, Any]] = []
    required = {
        "setup_time",
        "side",
        "entry",
        "stop",
        "target",
        "entry_mode",
        "max_bars_for_retest",
    }

    for s in setups:
        if not isinstance(s, dict) or not required.issubset(s.keys()):
            continue
        st = pd.to_datetime(s["setup_time"], errors="coerce")
        if pd.isna(st):
            continue
        side = str(s["side"]).lower()
        if side not in {"long", "short"}:
            continue
        valid_setups.append(
            {
                **s,
                "setup_time": st,
                "side": side,
                "entry": float(s["entry"]),
                "stop": float(s["stop"]),
                "target": float(s["target"]),
                "max_bars_for_retest": int(s.get("max_bars_for_retest", 0) or 0),
            }
        )

    valid_setups.sort(key=lambda x: x["setup_time"])

    r_values: List[float] = []
    trade_logs: List[Dict[str, Any]] = []
    active_until_idx = -1

    for setup in valid_setups:
        log = _base_log(setup)
        idx = data.index[data["time"] >= setup["setup_time"]]
        if len(idx) == 0:
            trade_logs.append(log)
            continue

        setup_idx = int(idx[0])

        if (not allow_overlapping_trades) and setup_idx <= active_until_idx:
            trade_logs.append(log)
            continue

        entry_idx: Optional[int] = None
        mode = str(setup["entry_mode"]).lower()

        if mode == "breakout":
            for j in range(setup_idx, len(data)):
                high_j = float(data.at[j, "high"])
                low_j = float(data.at[j, "low"])
                if low_j <= setup["entry"] <= high_j:
                    entry_idx = j
                    break
        elif mode == "retest":
            max_bars = max(0, setup["max_bars_for_retest"])
            end_idx = min(len(data) - 1, setup_idx + max_bars)
            for j in range(setup_idx + 1, end_idx + 1):
                high_j = float(data.at[j, "high"])
                low_j = float(data.at[j, "low"])
                if low_j <= setup["entry"] <= high_j:
                    entry_idx = j
                    break
        else:
            trade_logs.append(log)
            continue

        if entry_idx is None:
            trade_logs.append(log)
            continue

        log["entry_time"] = data.at[entry_idx, "time"]
        log["bars_to_entry"] = int(entry_idx - setup_idx)

        resolved = False
        for k in range(entry_idx, len(data)):
            high_k = float(data.at[k, "high"])
            low_k = float(data.at[k, "low"])

            if setup["side"] == "long":
                if low_k <= setup["stop"]:
                    log["exit_time"] = data.at[k, "time"]
                    log["outcome"] = "loss"
                    log["r_multiple"] = -1.0
                    log["bars_in_trade"] = int(k - entry_idx)
                    r_values.append(-1.0)
                    resolved = True
                    active_until_idx = k
                    break
                if high_k >= setup["target"]:
                    log["exit_time"] = data.at[k, "time"]
                    log["outcome"] = "win"
                    log["r_multiple"] = 2.0
                    log["bars_in_trade"] = int(k - entry_idx)
                    r_values.append(2.0)
                    resolved = True
                    active_until_idx = k
                    break
            else:
                if high_k >= setup["stop"]:
                    log["exit_time"] = data.at[k, "time"]
                    log["outcome"] = "loss"
                    log["r_multiple"] = -1.0
                    log["bars_in_trade"] = int(k - entry_idx)
                    r_values.append(-1.0)
                    resolved = True
                    active_until_idx = k
                    break
                if low_k <= setup["target"]:
                    log["exit_time"] = data.at[k, "time"]
                    log["outcome"] = "win"
                    log["r_multiple"] = 2.0
                    log["bars_in_trade"] = int(k - entry_idx)
                    r_values.append(2.0)
                    resolved = True
                    active_until_idx = k
                    break

        if not resolved:
            if force_close_on_final_bar and len(data) > 0:
                final_idx = len(data) - 1
                final_close = float(data.at[final_idx, "close"])
                if setup["side"] == "long":
                    risk = setup["entry"] - setup["stop"]
                    pnl = final_close - setup["entry"]
                else:
                    risk = setup["stop"] - setup["entry"]
                    pnl = setup["entry"] - final_close

                r_mult = float(pnl / risk) if risk > 0 else 0.0
                log["exit_time"] = data.at[final_idx, "time"]
                log["outcome"] = "unresolved"
                log["r_multiple"] = r_mult
                log["bars_in_trade"] = int(final_idx - entry_idx)
                r_values.append(r_mult)
                active_until_idx = final_idx
            else:
                log["outcome"] = "unresolved"
                log["r_multiple"] = 0.0
                if len(data) > 0:
                    active_until_idx = len(data) - 1

        trade_logs.append(log)

    return r_values, trade_logs
