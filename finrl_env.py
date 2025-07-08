


import numpy as np

from feature_engineering import compute_features
# Import from local FinRL-Meta installation

from meta.env_crypto_trading.env_multiple_crypto import CryptoEnv


class FuturesCryptoEnv(CryptoEnv):
    def __init__(self, df, additional_signals=None):
        super().__init__(df)
        self.additional_signals = additional_signals

    def _get_observation(self):
        base = super()._get_observation()
        live_feats = compute_features()
        return np.concatenate([base, live_feats])

    def step(self, action):
        obs, reward, done, info = super().step(action)
        return obs, reward, done, info
