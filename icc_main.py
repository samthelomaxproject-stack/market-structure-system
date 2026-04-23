import argparse
import os
from typing import Tuple

import pandas as pd

from icc_backtest import ICCBacktestResult, backtest_icc
from icc_strategy import ICCConfig, ICCStrategy


def load_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"CSV not found: {path}")
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError("CSV is empty")
    return df


def summarize(result: ICCBacktestResult, setups_count: int, trade_logs_df: pd.DataFrame) -> None:
    executed = int((trade_logs_df["outcome"].isin(["win", "loss", "unresolved"])).sum()) if not trade_logs_df.empty else 0
    wins = int((trade_logs_df["outcome"] == "win").sum()) if not trade_logs_df.empty else 0
    losses = int((trade_logs_df["outcome"] == "loss").sum()) if not trade_logs_df.empty else 0
    unresolved = int((trade_logs_df["outcome"] == "unresolved").sum()) if not trade_logs_df.empty else 0
    not_triggered = int((trade_logs_df["outcome"] == "not_triggered").sum()) if not trade_logs_df.empty else 0

    print(f"setups found: {setups_count}")
    print(f"trades executed: {executed}")
    print(f"wins: {wins}")
    print(f"losses: {losses}")
    print(f"unresolved: {unresolved}")
    print(f"not triggered: {not_triggered}")
    print(f"win rate: {result.win_rate:.2%}")
    print(f"average win: {result.average_win:.4f}")
    print(f"average loss: {result.average_loss:.4f}")
    print(f"expectancy: {result.expectancy:.4f}")


def build_config(args: argparse.Namespace) -> ICCConfig:
    return ICCConfig(
        symbol=args.symbol,
        timeframe=args.timeframe,
        higher_timeframe=args.higher_timeframe,
        pivot_left=args.pivot_left,
        pivot_right=args.pivot_right,
        moving_average_period=args.moving_average_period,
        rr_multiple=args.rr_multiple,
        entry_mode=args.entry_mode,
        bias_mode=args.bias_mode,
        require_htf_alignment=not args.disable_htf_alignment,
        max_bars_for_correction=args.max_bars_for_correction,
        max_bars_for_retest=args.max_bars_for_retest,
        allow_overlapping_trades=args.allow_overlapping_trades,
        break_buffer_pct=args.break_buffer_pct,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ICC Futures Continuation Strategy")
    parser.add_argument("--csv", required=True, help="Path to OHLCV CSV")
    parser.add_argument("--symbol", default="FUTURES")
    parser.add_argument("--timeframe", default="15min")
    parser.add_argument("--higher-timeframe", default="1H")
    parser.add_argument("--pivot-left", type=int, default=3)
    parser.add_argument("--pivot-right", type=int, default=3)
    parser.add_argument("--moving-average-period", type=int, default=50)
    parser.add_argument("--rr-multiple", type=float, default=2.0)
    parser.add_argument("--entry-mode", choices=["breakout", "retest"], default="breakout")
    parser.add_argument("--bias-mode", choices=["ma", "structure"], default="ma")
    parser.add_argument("--disable-htf-alignment", action="store_true")
    parser.add_argument("--max-bars-for-correction", type=int, default=20)
    parser.add_argument("--max-bars-for-retest", type=int, default=10)
    parser.add_argument("--allow-overlapping-trades", action="store_true")
    parser.add_argument("--break-buffer-pct", type=float, default=0.0005)
    parser.add_argument("--force-close-on-final-bar", action="store_true")

    args = parser.parse_args()

    raw_df = load_csv(args.csv)
    config = build_config(args)
    strategy = ICCStrategy(config)

    processed_df, setups, used_config = strategy.run(raw_df)

    r_values, trade_logs = backtest_icc(
        processed_df,
        setups,
        allow_overlapping_trades=used_config.allow_overlapping_trades,
        force_close_on_final_bar=args.force_close_on_final_bar,
    )

    result = ICCBacktestResult(r_values)
    trade_logs_df = pd.DataFrame(trade_logs)

    processed_df.to_csv("icc_processed_signals.csv", index=False)
    trade_logs_df.to_csv("icc_trade_logs.csv", index=False)

    summarize(result, len(setups), trade_logs_df)
