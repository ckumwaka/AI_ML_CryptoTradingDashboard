import argparse
import datetime as dt
import os
from typing import List

import numpy as np
import pandas as pd
import torch

# FinRL‑Meta core imports -----------------------------------------------------
from agents.stablebaselines3_models import DRLAgent as DRLAgentSB3
from meta.data_processors.binance import BinanceProcessor
from meta.env_crypto_trading.env_multiple_crypto import CryptoEnv


# ----------------------------------------------------------------------
# Utility helpers
# ----------------------------------------------------------------------

def get_binance_data(
    tickers: List[str],
    start_date: str,
    end_date: str,
    interval: str = "1d",
):
    """Download & transform historical OHLCV from Binance public API."""

    processor = BinanceProcessor(
        data_source="binance",
        start_date=start_date,
        end_date=end_date,
        time_interval=interval
    )

    processor.download_data(
        ticker_list=tickers,
        save_path="./data/binance_data.csv"
    )

    # Since clean_data is not implemented, just drop NaNs here
    cleaned_df = processor.dataframe.dropna()

    # Define your own technical indicators list to match your data
    # (adjust as per your use case or your own indicators)
    tech_indicator_list = ['open', 'high', 'low', 'close', 'volume']

    # Manually implement df_to_array here (similar to your commented code in binance.py)
    unique_tickers = cleaned_df.tic.unique()
    price_array = np.column_stack([cleaned_df[cleaned_df.tic == tic].close.values for tic in unique_tickers])
    tech_array = np.hstack([
        cleaned_df.loc[cleaned_df.tic == tic, tech_indicator_list].values for tic in unique_tickers
    ])
    turb_array = np.array([])  # no turbulence for now

    return price_array, tech_array, turb_array, tech_indicator_list


# ----------------------------------------------------------------------
# Main Training Function
# ----------------------------------------------------------------------

def run_train(args):
    print("[INFO] ⏳  Fetching Binance data …")
    today = dt.date.today()
    start = (today - dt.timedelta(days=args.lookback_days)).strftime("%Y-%m-%d")
    end = today.strftime("%Y-%m-%d")

    price_array, tech_array, turb_array, tech_indicators = get_binance_data(
        tickers=args.tickers,
        start_date=start,
        end_date=end,
        interval=args.interval,
    )

    env_config = {
        "price_array": price_array,
        "tech_array": tech_array,
        "turbulence_array": turb_array,
        "if_vix": False,
        "initial_amount": args.initial_cash,
        "reward_scaling": 1e-4,
        "state_space": tech_array.shape[1],
        "action_space": len(args.tickers),
        "tech_indicator_list": tech_indicators,
    }

    env_train = CryptoEnv(config=env_config)

    print("[INFO] 🏗️  Building DRL agent …")
    agent = DRLAgentSB3(env=env_train)
    model = agent.get_model(args.algo)

    print("[INFO] 🚀  Training starts …")
    trained_model = agent.train_model(
        model=model,
        tb_log_name=args.algo,
        total_timesteps=args.total_timesteps,
    )

    save_path = os.path.join("trained_models", f"{args.algo}_{dt.datetime.now():%Y%m%d_%H%M%S}")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    trained_model.save(save_path)
    print(f"[INFO] ✅  Model saved to {save_path}.")


# ----------------------------------------------------------------------
# CLI argument parser
# ----------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="FinRL‑Meta crypto training script using live Binance data")
    p.add_argument("--mode", choices=["train", "test", "trade"], default="train")
    p.add_argument("--tickers", nargs="*", default=["BTCUSDT", "ETHUSDT"], help="Crypto symbols to trade")
    p.add_argument("--interval", default="1d", help="Binance kline interval (e.g. 1d, 4h, 1h, 15m)")
    p.add_argument("--lookback-days", type=int, default=365, help="How many days of history to fetch")
