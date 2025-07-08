import torch
import numpy as np
from sklearn.cross_decomposition import PLSRegression

def window_features(X: np.ndarray, lookback: int) -> np.ndarray:
    n_samples, n_feats = X.shape
    if lookback <= 1:
        return X
    windows = [X[i-lookback:i].flatten() for i in range(lookback, n_samples)]
    return np.stack(windows)

class KryptoPLS(torch.nn.Module):
    def __init__(self, n_components=3, init_lookback=20):
        super().__init__()
        self.n_components = n_components
        self.lookback = torch.nn.Parameter(torch.tensor(init_lookback, dtype=torch.float32))
        self.pls = PLSRegression(n_components=n_components, scale=True)
    
    def forward(self, X_train: np.ndarray, y_train: np.ndarray, X_pred: np.ndarray):
        lb = max(1, int(self.lookback.item()))
        Xw = window_features(X_train, lb)
        yw = y_train[lb:]
        self.pls.fit(Xw, yw)
        Xp = window_features(X_pred, lb)
        return self.pls.predict(Xp)
    
    def fit_predict(self, X_train, y_train, X_pred):
        return self.forward(X_train, y_train, X_pred)
