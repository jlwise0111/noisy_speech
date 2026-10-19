import numpy as np

from lrb.features import patches


def test_patch_shape_and_centre():
    L = np.random.default_rng(0).normal(size=(20, 11))
    P = patches(L)
    assert P.shape == (20, 11, 9)
    np.testing.assert_array_equal(P[:, :, 4], L)


def test_patch_neighbours_and_edges():
    L = np.arange(5 * 4, dtype=float).reshape(5, 4)
    P = patches(L)
    # Interior: feature 3*dt+df is L[t-1+dt, f-1+df]
    np.testing.assert_array_equal(P[2, 1].reshape(3, 3), L[1:4, 0:3])
    # Corner (0, 0) uses edge padding
    expected = np.array([[L[0, 0], L[0, 0], L[0, 1]],
                         [L[0, 0], L[0, 0], L[0, 1]],
                         [L[1, 0], L[1, 0], L[1, 1]]])
    np.testing.assert_array_equal(P[0, 0].reshape(3, 3), expected)
