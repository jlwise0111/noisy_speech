"""Per-frequency-bin Ridge regression from streamed sufficient statistics.

For every bin f we accumulate n, Σx, Σxxᵀ, Σy, Σxy, Σy² over all frames. The statistics add
together, so files can be processed in parallel and summed. Ridge fits for any α, and the exact
MSE on any other set of statistics, follow in closed form without another pass over the audio.

Features are standardized by the training mean and std, and the intercept is not penalized.
α is a per-frame penalty: the fit equals sklearn's `StandardScaler` + `Ridge(alpha=α·n,
fit_intercept=True)` per bin, where n is the number of training frames. That keeps one α grid
meaningful whether training on 28spk, 56spk or both (n differs by millions of frames).
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class Stats:
    n: np.ndarray    # (F,)
    sx: np.ndarray   # (F, D)
    sxx: np.ndarray  # (F, D, D)
    sy: np.ndarray   # (F,)
    sxy: np.ndarray  # (F, D)
    syy: np.ndarray  # (F,)

    @classmethod
    def zeros(cls, n_bins, n_features):
        F, D = n_bins, n_features
        return cls(np.zeros(F), np.zeros((F, D)), np.zeros((F, D, D)),
                   np.zeros(F), np.zeros((F, D)), np.zeros(F))

    @classmethod
    def from_arrays(cls, X, y):
        """X: (T, F, D) features, y: (T, F) targets."""
        X = X.astype(np.float64, copy=False)
        y = y.astype(np.float64, copy=False)
        return cls(
            n=np.full(X.shape[1], float(X.shape[0])),
            sx=X.sum(axis=0),
            sxx=np.einsum("tfi,tfj->fij", X, X, optimize=True),
            sy=y.sum(axis=0),
            sxy=np.einsum("tfi,tf->fi", X, y, optimize=True),
            syy=(y * y).sum(axis=0),
        )

    def __add__(self, other):
        return Stats(*(a + b for a, b in zip(self._fields(), other._fields())))

    def _fields(self):
        return (self.n, self.sx, self.sxx, self.sy, self.sxy, self.syy)

    def to_dict(self, prefix=""):
        names = ("n", "sx", "sxx", "sy", "sxy", "syy")
        return {prefix + k: v for k, v in zip(names, self._fields())}


@dataclass
class Model:
    W: np.ndarray      # (F, D) weights in original (unstandardized) feature space
    b: np.ndarray      # (F,)
    alpha: np.ndarray  # (F,) penalty used per bin

    def predict(self, X):
        """X: (T, F, D) -> (T, F)."""
        return np.einsum("tfd,fd->tf", X, self.W) + self.b


def fit(stats, alpha):
    """Closed-form ridge per bin. `alpha` (per-frame penalty) is a scalar or an (F,) array."""
    n = stats.n
    mu = stats.sx / n[:, None]
    ybar = stats.sy / n
    # Centered Gram and cross-moment: Σ(x-μ)(x-μ)ᵀ and Σ(x-μ)(y-ȳ)
    G = stats.sxx - n[:, None, None] * mu[:, :, None] * mu[:, None, :]
    c = stats.sxy - n[:, None] * mu * ybar[:, None]
    std = np.sqrt(np.maximum(np.einsum("fii->fi", G) / n[:, None], 0.0))
    std[std < 1e-12] = 1.0
    Gs = G / (std[:, :, None] * std[:, None, :])
    cs = c / std
    a = np.broadcast_to(np.asarray(alpha, dtype=np.float64), n.shape)
    D = Gs.shape[1]
    beta = np.linalg.solve(Gs + (a * n)[:, None, None] * np.eye(D), cs[:, :, None])[:, :, 0]
    W = beta / std
    b = ybar - np.einsum("fd,fd->f", W, mu)
    return Model(W=W, b=b, alpha=np.array(a))


def sse(model, stats):
    """Exact per-bin sum of squared errors of `model` on the data summarized by `stats`."""
    W, b = model.W, model.b
    return (stats.syy
            - 2 * np.einsum("fd,fd->f", W, stats.sxy)
            - 2 * b * stats.sy
            + np.einsum("fi,fij,fj->f", W, stats.sxx, W)
            + 2 * b * np.einsum("fd,fd->f", W, stats.sx)
            + stats.n * b * b)


def sweep(train, val, alphas):
    """Fit on `train` for each α; return per-bin train and val MSE, each shaped (A, F)."""
    tr = np.empty((len(alphas), len(train.n)))
    va = np.empty_like(tr)
    for i, a in enumerate(alphas):
        m = fit(train, a)
        tr[i] = sse(m, train) / train.n
        va[i] = sse(m, val) / val.n
    return tr, va
