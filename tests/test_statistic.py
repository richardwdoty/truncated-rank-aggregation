import numpy as np
import pytest

import tra


def test_statistic_basic_range():
    rng = np.random.default_rng(0)
    p = rng.uniform(0, 1, size=50)
    t = tra.statistic(p, k=5)
    assert 0.0 <= t <= 1.0


def test_statistic_rejects_nan():
    p = [0.1, np.nan, 0.2]
    with pytest.raises(ValueError, match="NaN"):
        tra.statistic(p, k=2)


def test_statistic_rejects_out_of_bounds():
    p = [0.1, -1e-6, 0.2]
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        tra.statistic(p, k=2)


def test_k_must_be_at_most_n():
    p = [0.2, 0.3, 0.4]
    with pytest.raises(ValueError, match="k must be <= n"):
        tra.statistic(p, k=4)


def test_accepts_column_vector_shape():
    rng = np.random.default_rng(1)
    p = rng.uniform(0, 1, size=(20, 1))
    t = tra.statistic(p, k=5)
    assert 0.0 <= t <= 1.0


def test_rejects_matrix_input():
    rng = np.random.default_rng(2)
    p = rng.uniform(0, 1, size=(10, 2))
    with pytest.raises(ValueError, match="1D"):
        tra.statistic(p, k=5)