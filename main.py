import argparse
import os

import pandas as pd

from backtest import BacktestResult, backtest
from strategy import MarketStructureStrategy


def load_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"CSV not found: {path}")
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError("CSV is empty")
    return df


def print_summary(result: BacktestResult, setups_count: int, trade_logs: pd.DataFrame) -> None:
    wins = int((trade_logs["outcome"] == "win").sum()) if not trade_logs.empty else 0
    losses = int((trade_logs["outcome"] == "loss").sum()) if not trade_logs.empty else 0
    unresolved = int((trade_logs["outcome"] == "unresolved").sum()) if not trade_logs.empty else 0

    print(f"setups found: {setups_count}")
    print(f"trades executed: {result.total_trades}")
    print(f"wins: {wins}")
    print(f"losses: {losses}")
    print(f"unresolved trades: {unresolved}")
    print(f"win rate: {result.win_rate:.2%}")
    print(f"average win: {result.average_win:.4f}")
    print(f"average loss: {result.average_loss:.4f}")
    print(f"expectancy: {result.expectancy:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Market Structure Bot (modular)")
    parser.add_argument("--csv", required=True, help="Path to OHLCV CSV")
    parser.add_argument(
        "--close-unresolved-at-end",
        action="store_true",
        help="Close unresolved trades at final bar close and assign fractional R",
    )
    args = parser.parse_args()

    raw = load_csv(args.csv)
    strategy = MarketStructureStrategy()
    processed_df, setups = strategy.run(raw)

    r_vals, trade_logs = backtest(
        processed_df,
        setups,
        close_unresolved_at_end=args.close_unresolved_at_end,
    )

    summary = BacktestResult(r_vals)
    trade_logs_df = pd.DataFrame(trade_logs)

    processed_df.to_csv("processed_signals.csv", index=False)
    trade_logs_df.to_csv("trade_logs.csv", index=False)

    print_summary(summary, len(setups), trade_logs_df)
