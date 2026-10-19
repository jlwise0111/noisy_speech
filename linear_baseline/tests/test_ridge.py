import numpy as np

from lrb.ridge import Stats, fit, sse, sweep


def make_data(T=400, F=5, D=9, noise=0.01, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(T, F, D)) * rng.uniform(0.5, 3, size=D) + rng.normal(size=D)
    W = rng.normal(size=(F, D))
    b = rng.normal(size=F)
    y = np.einsum("tfd,fd->tf", X, W) + b + noise * rng.normal(size=(T, F))
    return X, y, W, b


def test_recovers_weights_with_small_alpha():
    X, y, W, b = make_data()
    m = fit(Stats.from_arrays(X, y), 1e-8)
    np.testing.assert_allclose(m.W, W, atol=5e-3)
    np.testing.assert_allclose(m.b, b, atol=5e-3)


def test_matches_explicit_standardized_ridge():
    X, y, _, _ = make_data(noise=1.0)
    alpha = 0.1
    m = fit(Stats.from_arrays(X, y), alpha)
    for f in range(X.shape[1]):
        Xf, yf = X[:, f], y[:, f]
        mu, sd = Xf.mean(0), Xf.std(0)
        Z = (Xf - mu) / sd
        beta = np.linalg.solve(Z.T @ Z + alpha * len(Z) * np.eye(Z.shape[1]), Z.T @ (yf - yf.mean()))
        np.testing.assert_allclose(m.W[f], beta / sd, rtol=1e-8, atol=1e-10)


def test_sse_from_stats_matches_direct():
    X, y, _, _ = make_data(noise=0.5)
    Xv, yv, _, _ = make_data(T=150, noise=0.5, seed=1)
    m = fit(Stats.from_arrays(X, y), 0.01)
    direct = ((m.predict(Xv) - yv) ** 2).sum(axis=0)
    np.testing.assert_allclose(sse(m, Stats.from_arrays(Xv, yv)), direct, rtol=1e-8)


def test_stats_are_additive():
    X, y, _, _ = make_data()
    whole = Stats.from_arrays(X, y)
    halves = Stats.from_arrays(X[:150], y[:150]) + Stats.from_arrays(X[150:], y[150:])
    for a, b in zip(whole._fields(), halves._fields()):
        np.testing.assert_allclose(a, b)


def test_larger_alpha_shrinks_weights_and_sweep_shapes():
    X, y, _, _ = make_data(noise=1.0)
    s = Stats.from_arrays(X, y)
    norms = [np.linalg.norm(fit(s, a).W) for a in (1e-5, 1e-2, 1.0, 1e2)]
    assert all(n1 > n2 for n1, n2 in zip(norms, norms[1:]))
    tr, va = sweep(s, s, [1e-3, 1e-1, 1.0])
    assert tr.shape == va.shape == (3, X.shape[1])
    np.testing.assert_allclose(tr, va)
