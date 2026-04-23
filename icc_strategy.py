from dataclasses import asdict, dataclass
from typing import Dict, List, Literal, Optional, Tuple

import numpy as np
import pandas as pd


BiasMode = Literal["ma", "structure"]
EntryMode = Literal["breakout", "retest"]


@dataclass
class ICCConfig:
    symbol: str = "FUTURES"
    timeframe: str = "15min"
    higher_timeframe: str = "1H"
    pivot_left: int = 3
    pivot_right: int = 3
    moving_average_period: int = 50
    rr_multiple: float = 2.0
    entry_mode: EntryMode = "breakout"
    bias_mode: BiasMode = "ma"
    require_htf_alignment: bool = True
    max_bars_for_correction: int = 20
    max_bars_for_retest: int = 10
    allow_overlapping_trades: bool = False
    break_buffer_pct: float = 0.0005


class ICCStrategy:
    REQUIRED_COLUMNS = {"time", "open", "high", "low", "close", "volume"}

    def __init__(self, config: Optional[ICCConfig] = None):
        self.config = config or ICCConfig()

    @staticmethod
    def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
        rename_map: Dict[str, str] = {}
        for col in df.columns:
            low = col.lower()
            if low in {"timestamp", "datetime", "date", "time"}:
                rename_map[col] = "time"
            elif low in {"open", "high", "low", "close", "volume"}:
                rename_map[col] = low
        out = df.rename(columns=rename_map).copy()
        return out

    def _validate(self, df: pd.DataFrame) -> None:
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame")
        missing = self.REQUIRED_COLUMNS - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {sorted(missing)}")

    def _detect_pivots(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out["pivot_high"] = np.nan
        out["pivot_low"] = np.nan

        highs = out["high"].to_numpy()
        lows = out["low"].to_numpy()
        left = self.config.pivot_left
        right = self.config.pivot_right

        for i in range(left, len(out) - right):
            h_win = highs[i - left : i + right + 1]
            l_win = lows[i - left : i + right + 1]
            if np.argmax(h_win) == left and highs[i] == np.max(h_win):
                out.iat[i, out.columns.get_loc("pivot_high")] = highs[i]
            if np.argmin(l_win) == left and lows[i] == np.min(l_win):
                out.iat[i, out.columns.get_loc("pivot_low")] = lows[i]

        out["last_swing_high"] = out["pivot_high"].ffill()
        out["last_swing_low"] = out["pivot_low"].ffill()
        return out

    @staticmethod
    def _compute_structure_bias(df: pd.DataFrame) -> pd.Series:
        last_high = np.nan
        last_low = np.nan
        state = "neutral"
        out: List[str] = []

        for _, row in df.iterrows():
            if pd.notna(row.get("pivot_high", np.nan)):
                last_high = float(row["pivot_high"])
            if pd.notna(row.get("pivot_low", np.nan)):
                last_low = float(row["pivot_low"])

            close = float(row["close"])
            if pd.notna(last_high) and close > last_high:
                state = "bullish"
            elif pd.notna(last_low) and close < last_low:
                state = "bearish"
            out.append(state)

        return pd.Series(out, index=df.index, name="structure_bias")

    def _build_htf_bias(self, df: pd.DataFrame) -> pd.DataFrame:
        ohlc = {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }
        htf = (
            df.set_index("time")
            .resample(self.config.higher_timeframe)
            .agg(ohlc)
            .dropna(subset=["open", "high", "low", "close"])
            .reset_index()
        )

        htf["htf_ma"] = htf["close"].rolling(self.config.moving_average_period, min_periods=1).mean()
        htf = self._detect_pivots(htf)
        htf["htf_structure_bias"] = self._compute_structure_bias(htf)

        htf["htf_bias"] = "neutral"
        if self.config.bias_mode == "ma":
            htf.loc[htf["close"] > htf["htf_ma"], "htf_bias"] = "bullish"
            htf.loc[htf["close"] < htf["htf_ma"], "htf_bias"] = "bearish"
        else:
            htf.loc[htf["htf_structure_bias"] == "bullish", "htf_bias"] = "bullish"
            htf.loc[htf["htf_structure_bias"] == "bearish", "htf_bias"] = "bearish"

        aligned = pd.merge_asof(
            df.sort_values("time"),
            htf[["time", "htf_bias", "htf_ma", "htf_structure_bias"]].sort_values("time"),
            on="time",
            direction="backward",
        )
        aligned["htf_bias"] = aligned["htf_bias"].fillna("neutral")
        return aligned

    def _break_above(self, price: float, level: float) -> bool:
        if pd.isna(level):
            return False
        return price > float(level) * (1.0 + self.config.break_buffer_pct)

    def _break_below(self, price: float, level: float) -> bool:
        if pd.isna(level):
            return False
        return price < float(level) * (1.0 - self.config.break_buffer_pct)

    def _can_trade_side(self, htf_bias: str, side: str) -> bool:
        if not self.config.require_htf_alignment:
            return True
        if side == "long":
            return htf_bias == "bullish"
        return htf_bias == "bearish"

    def _state_template(self) -> Dict[str, object]:
        return {
            "state": "idle",
            "indication_idx": None,
            "indication_level": np.nan,
            "origin_invalid_level": np.nan,
            "correction_high": np.nan,
            "correction_low": np.nan,
            "continuation_level": np.nan,
            "correction_bars": 0,
            "pullback_seen": False,
            "continuation_idx": None,
        }

    def _reset_state(self, state: Dict[str, object]) -> None:
        state.clear()
        state.update(self._state_template())

    def generate_setups(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[Dict[str, object]]]:
        out = df.copy()
        setups: List[Dict[str, object]] = []

        out["icc_state_long"] = "idle"
        out["icc_state_short"] = "idle"

        long_state = self._state_template()
        short_state = self._state_template()

        for i in range(len(out)):
            idx = out.index[i]
            row = out.iloc[i]
            htf_bias = str(row.get("htf_bias", "neutral"))
            high = float(row["high"])
            low = float(row["low"])
            close = float(row["close"])
            last_swing_high = row.get("last_swing_high", np.nan)
            last_swing_low = row.get("last_swing_low", np.nan)

            # ---- LONG ICC STATE MACHINE ----
            if long_state["state"] == "idle":
                if self._can_trade_side(htf_bias, "long") and self._break_above(close, last_swing_high):
                    long_state["state"] = "indication_found"
                    long_state["indication_idx"] = i
                    long_state["indication_level"] = float(last_swing_high)
                    long_state["origin_invalid_level"] = float(last_swing_low) if pd.notna(last_swing_low) else np.nan
                    long_state["correction_high"] = high
                    long_state["correction_low"] = low
                    long_state["correction_bars"] = 0
                    long_state["pullback_seen"] = False
                out.at[idx, "icc_state_long"] = long_state["state"]
            else:
                if pd.notna(long_state["origin_invalid_level"]) and low <= float(long_state["origin_invalid_level"]):
                    long_state["state"] = "invalidated"

                if long_state["state"] in {"indication_found", "correction_active"}:
                    long_state["state"] = "correction_active"
                    long_state["correction_bars"] = int(long_state["correction_bars"]) + 1
                    long_state["correction_high"] = max(float(long_state["correction_high"]), high)
                    long_state["correction_low"] = min(float(long_state["correction_low"]), low)

                    if i > int(long_state["indication_idx"]):
                        prev_low = float(out.iloc[i - 1]["low"])
                        if low < prev_low:
                            long_state["pullback_seen"] = True

                    if int(long_state["correction_bars"]) > self.config.max_bars_for_correction:
                        long_state["state"] = "invalidated"

                    if long_state["state"] != "invalidated" and bool(long_state["pullback_seen"]):
                        cont_level = float(long_state["correction_high"])
                        if self._break_above(close, cont_level):
                            long_state["state"] = "continuation_confirmed"
                            long_state["continuation_level"] = cont_level
                            long_state["continuation_idx"] = i

                if long_state["state"] == "continuation_confirmed":
                    entry = float(long_state["continuation_level"])
                    stop = float(long_state["correction_low"])
                    risk = entry - stop
                    target = entry + self.config.rr_multiple * risk

                    valid = (
                        risk > 0
                        and entry != stop
                        and np.isfinite(target)
                        and target > entry
                        and self._can_trade_side(htf_bias, "long")
                    )
                    if valid:
                        setups.append(
                            {
                                "setup_time": row["time"],
                                "time": row["time"],
                                "side": "long",
                                "indication_level": float(long_state["indication_level"]),
                                "correction_high": float(long_state["correction_high"]),
                                "correction_low": float(long_state["correction_low"]),
                                "continuation_level": float(long_state["continuation_level"]),
                                "entry": entry,
                                "stop": stop,
                                "target": float(target),
                                "risk": float(risk),
                                "entry_mode": self.config.entry_mode,
                                "htf_bias": htf_bias,
                                "max_bars_for_retest": self.config.max_bars_for_retest,
                            }
                        )
                        long_state["state"] = "setup_ready"
                    else:
                        long_state["state"] = "invalidated"

                if long_state["state"] in {"setup_ready", "invalidated", "completed"}:
                    self._reset_state(long_state)

                out.at[idx, "icc_state_long"] = long_state["state"]

            # ---- SHORT ICC STATE MACHINE ----
            if short_state["state"] == "idle":
                if self._can_trade_side(htf_bias, "short") and self._break_below(close, last_swing_low):
                    short_state["state"] = "indication_found"
                    short_state["indication_idx"] = i
                    short_state["indication_level"] = float(last_swing_low)
                    short_state["origin_invalid_level"] = float(last_swing_high) if pd.notna(last_swing_high) else np.nan
                    short_state["correction_high"] = high
                    short_state["correction_low"] = low
                    short_state["correction_bars"] = 0
                    short_state["pullback_seen"] = False
                out.at[idx, "icc_state_short"] = short_state["state"]
            else:
                if pd.notna(short_state["origin_invalid_level"]) and high >= float(short_state["origin_invalid_level"]):
                    short_state["state"] = "invalidated"

                if short_state["state"] in {"indication_found", "correction_active"}:
                    short_state["state"] = "correction_active"
                    short_state["correction_bars"] = int(short_state["correction_bars"]) + 1
                    short_state["correction_high"] = max(float(short_state["correction_high"]), high)
                    short_state["correction_low"] = min(float(short_state["correction_low"]), low)

                    if i > int(short_state["indication_idx"]):
                        prev_high = float(out.iloc[i - 1]["high"])
                        if high > prev_high:
                            short_state["pullback_seen"] = True

                    if int(short_state["correction_bars"]) > self.config.max_bars_for_correction:
                        short_state["state"] = "invalidated"

                    if short_state["state"] != "invalidated" and bool(short_state["pullback_seen"]):
                        cont_level = float(short_state["correction_low"])
                        if self._break_below(close, cont_level):
                            short_state["state"] = "continuation_confirmed"
                            short_state["continuation_level"] = cont_level
                            short_state["continuation_idx"] = i

                if short_state["state"] == "continuation_confirmed":
                    entry = float(short_state["continuation_level"])
                    stop = float(short_state["correction_high"])
                    risk = stop - entry
                    target = entry - self.config.rr_multiple * risk

                    valid = (
                        risk > 0
                        and entry != stop
                        and np.isfinite(target)
                        and target < entry
                        and self._can_trade_side(htf_bias, "short")
                    )
                    if valid:
                        setups.append(
                            {
                                "setup_time": row["time"],
                                "time": row["time"],
                                "side": "short",
                                "indication_level": float(short_state["indication_level"]),
                                "correction_high": float(short_state["correction_high"]),
                                "correction_low": float(short_state["correction_low"]),
                                "continuation_level": float(short_state["continuation_level"]),
                                "entry": entry,
                                "stop": stop,
                                "target": float(target),
                                "risk": float(risk),
                                "entry_mode": self.config.entry_mode,
                                "htf_bias": htf_bias,
                                "max_bars_for_retest": self.config.max_bars_for_retest,
                            }
                        )
                        short_state["state"] = "setup_ready"
                    else:
                        short_state["state"] = "invalidated"

                if short_state["state"] in {"setup_ready", "invalidated", "completed"}:
                    self._reset_state(short_state)

                out.at[idx, "icc_state_short"] = short_state["state"]

        return out, setups

    def run(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[Dict[str, object]], ICCConfig]:
        out = self._standardize_columns(df)
        out["time"] = pd.to_datetime(out["time"], errors="coerce")
        out = out.dropna(subset=["time"]).sort_values("time").reset_index(drop=True)

        self._validate(out)

        out = self._detect_pivots(out)
        out = self._build_htf_bias(out)
        out, setups = self.generate_setups(out)
        return out, setups, self.config


def config_to_dict(config: ICCConfig) -> Dict[str, object]:
    return asdict(config)
