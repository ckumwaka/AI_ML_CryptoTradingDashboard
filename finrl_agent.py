from finrl import config, train
class FinRLAgent:
    def __init__(self, env, algorithm='PPO', use_cpcv=False):
        self.env = env
        self.alg = algorithm
        self.use_cpcv = use_cpcv
        self.trainer = None
    def train(self):
        self.trainer = train(env=self.env, model_name=self.alg, use_cpcv=self.use_cpcv)
    def update_env_signals(self, new_signals):
        self.env.additional_signals = new_signals
