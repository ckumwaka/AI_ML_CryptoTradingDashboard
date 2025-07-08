# data_ingestion.py
import requests
from pycoingecko import CoinGeckoAPI
import ccxt
import pandas as pd
# For TradingView signals, via unofficial API or webhook

cg = CoinGeckoAPI()
exchange = ccxt.binance()

def get_spot_price(symbol="bitcoin", vs_currency="usd"):
    """Fetch current spot price."""
    return cg.get_price(ids=symbol, vs_currencies=vs_currency)

def get_onchain_tvl(contract_address):
    """Fetch on-chain TVL for given contract."""
    info = cg.get_token_info_by_contract_address_platform(
        id="ethereum", contract_address=contract_address
    )
    return info["market_data"]["total_value_locked"]

def get_orderbook(symbol="BTC/USDT", limit=50):
    """Fetch current orderbook."""
    return exchange.fetch_order_book(symbol, limit=limit)

def get_tradingview_signal(symbol="BINANCE:BTCUSDT"):
    """Fetch trading signal (bullish/bearish)."""
    try:
        url = f"https://api.tradingview.com/v1/signals/{symbol}"
        resp = requests.get(url)
        return resp.json().get("signal", "neutral")
    except:
        return "neutral"
