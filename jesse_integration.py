from jesse import research, backtest_mode, live_trading
import numpy as np
from feature_engineering import compute_features

class JesseExecutor:
    def __init__(self, pls, agent, symbol):
        self.pls = pls
        self.agent = agent
        self.symbol = symbol

    def run_live(self):
        @research
        def strategy():
            feats = compute_features().reshape(1, -1)
            pred = self.pls.fit_predict(np.zeros((1, feats.shape[1])), np.zeros(1), feats)[0]
            rl_act = self.agent.predict_action(feats)[0]

            if pred > 0 and rl_act == 1:
                return 'Long'
            elif pred < 0 and rl_act == -1:
                return 'Short'
            else:
                return 'Neutral'

        live_trading.run(strategy, symbol=self.symbol)

