from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class StrategyConfig:
    atr_period: int = 14
    pivot_left: int = 3
    pivot_right: int = 3
    sweep_recent_bars: int = 5
    rr_multiple: float = 2.0
    require_displacement: bool = True


class MarketStructureStrategy:
    REQUIRED_COLUMNS = {"open", "high", "low", "close", "volume"}

    def __init__(self, config: Optional[StrategyConfig] = None):
        self.config = config or StrategyConfig()

    @staticmethod
    def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
        rename_map = {}
        for c in df.columns:
            lc = c.lower()
            if lc in {"timestamp", "time", "datetime", "date"}:
                rename_map[c] = "time"
            elif lc in {"open", "high", "low", "close", "volume"}:
                rename_map[c] = lc
        return df.rename(columns=rename_map).copy()

    def validate_df(self, df: pd.DataFrame) -> None:
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame")
        missing = self.REQUIRED_COLUMNS - set(c.lower() for c in df.columns)
        if missing:
            raise ValueError(f"Missing required OHLCV columns: {sorted(missing)}")

    def atr(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        prev_close = out["close"].shift(1)
        tr1 = out["high"] - out["low"]
        tr2 = (out["high"] - prev_close).abs()
        tr3 = (out["low"] - prev_close).abs()
        out["tr"] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        out["atr"] = out["tr"].ewm(alpha=1 / self.config.atr_period, adjust=False).mean()
        return out

    def detect_pivots(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out["pivot_high"] = np.nan
        out["pivot_low"] = np.nan
        highs = out["high"].to_numpy()
        lows = out["low"].to_numpy()

        left = self.config.pivot_left
        right = self.config.pivot_right

        for i in range(left, len(out) - right):
            h_window = highs[i - left : i + right + 1]
            l_window = lows[i - left : i + right + 1]

            if np.argmax(h_window) == left and highs[i] == np.max(h_window):
                out.iat[i, out.columns.get_loc("pivot_high")] = highs[i]

            if np.argmin(l_window) == left and lows[i] == np.min(l_window):
                out.iat[i, out.columns.get_loc("pivot_low")] = lows[i]

        return out

    def detect_structure(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out["last_pivot_high"] = np.nan
        out["last_pivot_low"] = np.nan
        out["bos_up"] = False
        out["bos_down"] = False
        out["structure"] = "neutral"
        out["choch_up"] = False
        out["choch_down"] = False

        last_high: Optional[float] = None
        last_low: Optional[float] = None
        structure_state = "neutral"

        for i in range(len(out)):
            idx = out.index[i]
            ph = out.at[idx, "pivot_high"]
            pl = out.at[idx, "pivot_low"]

            if pd.notna(ph):
                last_high = float(ph)
            if pd.notna(pl):
                last_low = float(pl)

            out.at[idx, "last_pivot_high"] = last_high
            out.at[idx, "last_pivot_low"] = last_low

            close_i = float(out.at[idx, "close"])
            bos_up = last_high is not None and close_i > last_high
            bos_down = last_low is not None and close_i < last_low

            out.at[idx, "bos_up"] = bool(bos_up)
            out.at[idx, "bos_down"] = bool(bos_down)

            previous_structure = structure_state
            if bos_up and not bos_down:
                structure_state = "bullish"
            elif bos_down and not bos_up:
                structure_state = "bearish"

            out.at[idx, "structure"] = structure_state
            out.at[idx, "choch_up"] = previous_structure == "bearish" and structure_state == "bullish"
            out.at[idx, "choch_down"] = previous_structure == "bullish" and structure_state == "bearish"

        return out

    def detect_sweeps(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out["sweep_high"] = False
        out["sweep_low"] = False

        last_high: Optional[float] = None
        last_low: Optional[float] = None

        for i in range(len(out)):
            idx = out.index[i]
            high_i = float(out.at[idx, "high"])
            low_i = float(out.at[idx, "low"])
            close_i = float(out.at[idx, "close"])

            if last_high is not None:
                out.at[idx, "sweep_high"] = bool(high_i > last_high and close_i < last_high)
            if last_low is not None:
                out.at[idx, "sweep_low"] = bool(low_i < last_low and close_i > last_low)

            ph = out.at[idx, "pivot_high"]
            pl = out.at[idx, "pivot_low"]
            if pd.notna(ph):
                last_high = float(ph)
            if pd.notna(pl):
                last_low = float(pl)

        out["recent_sweep_high"] = (
            out["sweep_high"].astype(int).rolling(self.config.sweep_recent_bars, min_periods=1).max().astype(bool)
        )
        out["recent_sweep_low"] = (
            out["sweep_low"].astype(int).rolling(self.config.sweep_recent_bars, min_periods=1).max().astype(bool)
        )
        return out

    def detect_displacement(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        if "atr" not in out.columns:
            out = self.atr(out)

        out["body_size"] = (out["close"] - out["open"]).abs()
        out["displacement"] = out["body_size"] > out["atr"]
        # Directional displacement so setup quality can require momentum alignment.
        out["bullish_displacement"] = out["displacement"] & (out["close"] > out["open"])
        out["bearish_displacement"] = out["displacement"] & (out["close"] < out["open"])
        return out

    def detect_fvg(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out["bullish_fvg"] = False
        out["bearish_fvg"] = False
        out["bullish_fvg_top"] = np.nan
        out["bullish_fvg_bottom"] = np.nan
        out["bearish_fvg_top"] = np.nan
        out["bearish_fvg_bottom"] = np.nan

        for i in range(2, len(out)):
            idx = out.index[i]
            low_i = float(out.at[idx, "low"])
            high_i = float(out.at[idx, "high"])
            high_im2 = float(out.at[out.index[i - 2], "high"])
            low_im2 = float(out.at[out.index[i - 2], "low"])

            if low_i > high_im2:
                out.at[idx, "bullish_fvg"] = True
                out.at[idx, "bullish_fvg_bottom"] = high_im2
                out.at[idx, "bullish_fvg_top"] = low_i

            if high_i < low_im2:
                out.at[idx, "bearish_fvg"] = True
                out.at[idx, "bearish_fvg_bottom"] = high_i
                out.at[idx, "bearish_fvg_top"] = low_im2

        return out

    def generate_setups(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[Dict]]:
        out = df.copy()
        setups: List[Dict] = []
        out["setup_long"] = False
        out["setup_short"] = False

        for i in range(len(out)):
            idx = out.index[i]
            row = out.iloc[i]

            long_conditions = [
                bool(row.get("choch_up", False)),
                bool(row.get("recent_sweep_low", False)),
                bool(row.get("bullish_fvg", False)),
            ]
            short_conditions = [
                bool(row.get("choch_down", False)),
                bool(row.get("recent_sweep_high", False)),
                bool(row.get("bearish_fvg", False)),
            ]

            if self.config.require_displacement:
                long_conditions.append(bool(row.get("bullish_displacement", False)))
                short_conditions.append(bool(row.get("bearish_displacement", False)))

            if all(long_conditions):
                fvg_bottom = float(row["bullish_fvg_bottom"])
                fvg_top = float(row["bullish_fvg_top"])
                entry = (fvg_top + fvg_bottom) / 2.0
                stop = fvg_bottom
                risk = entry - stop

                # Skip invalid trade geometries.
                if risk > 0 and entry != stop:
                    target = entry + self.config.rr_multiple * risk
                    setups.append(
                        {
                            "time": row.get("time", idx),
                            "side": "long",
                            "entry": float(entry),
                            "stop": float(stop),
                            "target": float(target),
                        }
                    )
                    out.at[idx, "setup_long"] = True

            if all(short_conditions):
                fvg_bottom = float(row["bearish_fvg_bottom"])
                fvg_top = float(row["bearish_fvg_top"])
                entry = (fvg_top + fvg_bottom) / 2.0
                stop = fvg_top
                risk = stop - entry

                # Skip invalid trade geometries.
                if risk > 0 and entry != stop:
                    target = entry - self.config.rr_multiple * risk
                    setups.append(
                        {
                            "time": row.get("time", idx),
                            "side": "short",
                            "entry": float(entry),
                            "stop": float(stop),
                            "target": float(target),
                        }
                    )
                    out.at[idx, "setup_short"] = True

        return out, setups

    def run(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[Dict]]:
        self.validate_df(df)
        out = self.standardize_columns(df)

        if "time" in out.columns:
            out["time"] = pd.to_datetime(out["time"], errors="coerce")
            out = out.sort_values("time").reset_index(drop=True).set_index("time", drop=False)
        else:
            out = out.reset_index(drop=True)

        out = self.atr(out)
        out = self.detect_pivots(out)
        out = self.detect_structure(out)
        out = self.detect_sweeps(out)
        out = self.detect_displacement(out)
        out = self.detect_fvg(out)
        out, setups = self.generate_setups(out)
        return out, setups
