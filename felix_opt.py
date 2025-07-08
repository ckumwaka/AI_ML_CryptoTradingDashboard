class FelixOptimizer:
    def __init__(self, models, data, aim_loss):
        self.models = models
        self.data = data
        self.aim_loss = aim_loss
    def optimize(self, n_iter, lr):
        # jointly backpropagate loss across PLS mse and negative RL return
        ...
