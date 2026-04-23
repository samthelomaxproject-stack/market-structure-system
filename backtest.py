from dataclasses import dataclass
from typing import Dict, List

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


def backtest(df: pd.DataFrame, setups: List[Dict]) -> List[float]:
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be DataFrame")
    if not isinstance(setups, list):
        raise TypeError("setups must be list")

    data = df.reset_index(drop=True).copy()
    if "time" in data.columns:
        data["time"] = pd.to_datetime(data["time"], errors="coerce")
    else:
        data["time"] = data.index

    r_multiples: List[float] = []

    for s in setups:
        required = {"time", "side", "entry", "stop", "target"}
        if not isinstance(s, dict) or not required.issubset(s.keys()):
            continue

        side = str(s["side"]).lower()
        if side not in {"long", "short"}:
            continue

        setup_time = pd.to_datetime(s["time"], errors="coerce")
        if pd.isna(setup_time):
            continue

        entry = float(s["entry"])
        stop = float(s["stop"])
        target = float(s["target"])

        idx = data.index[data["time"] >= setup_time]
        if len(idx) == 0:
            continue
        signal_idx = int(idx[0])

        active = False
        resolved = False

        for j in range(signal_idx + 1, len(data)):
            high_j = float(data.at[j, "high"])
            low_j = float(data.at[j, "low"])

            if not active:
                if low_j <= entry <= high_j:
                    active = True
                else:
                    continue

            if side == "long":
                if low_j <= stop:
                    r_multiples.append(-1.0)
                    resolved = True
                    break
                if high_j >= target:
                    r_multiples.append(2.0)
                    resolved = True
                    break
            else:
                if high_j >= stop:
                    r_multiples.append(-1.0)
                    resolved = True
                    break
                if low_j <= target:
                    r_multiples.append(2.0)
                    resolved = True
                    break

        if not resolved:
            continue

    return r_multiples
