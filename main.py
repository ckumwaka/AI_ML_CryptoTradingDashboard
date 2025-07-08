import torch, numpy as np # type: ignore
from krypto_pls import KryptoPLS
from finrl_env import FuturesCryptoEnv
from finrl_agent import FinRLAgent
from jesse_integration import JesseExecutor
from felix_opt import FelixOptimizer
import ccxt
import pandas as pd
import numpy as np
from krypto_pls import KryptoPLS
from feature_engineering import compute_features

def load_ohlcv(symbol='BTC/USDT', timeframe='5m', since=None, limit=1000):
    ex = ccxt.binance()
    from_ts = ex.parse8601(since) if since else None
    all_bars = []
    while True:
        bars = ex.fetch_ohlcv(symbol, timeframe, since=from_ts, limit=limit)
        if not bars:
            break
        all_bars += bars
        if len(bars) < limit:
            break
        from_ts = bars[-1][0]
    df = pd.DataFrame(all_bars, columns=['timestamp','open','high','low','close','volume'])
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('datetime', inplace=True)
    return df

if __name__ == "__main__":
    df = load_ohlcv(since='2023-01-01T00:00:00Z')
    X_base = df[['open','high','low','close','volume']].values
    y = df['close'].shift(-1).fillna(method='ffill').values

    live_feat = compute_features()
    print("Live features:", live_feat)

    X = np.column_stack([X_base, np.tile(live_feat, (len(X_base), 1))])

    n_train = int(len(X) * 0.8)
    X_train, y_train = X[:n_train], y[:n_train]
    X_val = X[n_train:]

    pls = KryptoPLS(n_components=3, init_lookback=20)
    preds = pls.fit_predict(X_train, y_train, X_val)

    print("Sample Predictions:", preds[:5])
    print("Learnable lookback:", pls.lookback.item())

def main():
    print("🚀 Started main()")
    live_feat = compute_features()
    print("Live features:", live_feat)
    # … your pipeline logic …
    print("✅ Completed main()")

if __name__ == "__main__":
    print("Script is running as __main__")
    main()





