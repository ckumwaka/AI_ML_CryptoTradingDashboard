import numpy as np
from data_ingestion import get_spot_price, get_orderbook, get_tradingview_signal

def compute_features():
    """Return a 1D array of current feature values."""
    price = get_spot_price()["bitcoin"]["usd"]
    ob = get_orderbook()
    bid_ask_spread = ob['asks'][0][0] - ob['bids'][0][0]
    tv = get_tradingview_signal()
    tv_signal = 1 if tv == 'bullish' else (-1 if tv == 'bearish' else 0)

    return np.array([price, bid_ask_spread, tv_signal], dtype=float)
