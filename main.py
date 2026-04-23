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


def print_summary(result: BacktestResult) -> None:
    print(f"total trades: {result.total_trades}")
    print(f"win rate: {result.win_rate:.2%}")
    print(f"average win: {result.average_win:.4f}")
    print(f"average loss: {result.average_loss:.4f}")
    print(f"expectancy: {result.expectancy:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Market Structure Bot (modular)")
    parser.add_argument("--csv", required=True, help="Path to OHLCV CSV")
    args = parser.parse_args()

    raw = load_csv(args.csv)
    strategy = MarketStructureStrategy()
    processed_df, setups = strategy.run(raw)

    r_vals = backtest(processed_df, setups)
    summary = BacktestResult(r_vals)
    print_summary(summary)
