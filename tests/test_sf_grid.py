import numpy as np

import tra


def test_sf_grid_shape_exact():
    cs = np.linspace(0.0, 1.0, 51)
    vals = tra.sf_grid(cs, n=40, k=5, method="exact")
    assert vals.shape == cs.shape


def test_sf_grid_matches_scalar_exact():
    cs = np.linspace(0.01, 0.99, 25)
    vals_grid = tra.sf_grid(cs, n=30, k=6, method="exact")
    vals_scalar = np.array([tra.sf(float(c), n=30, k=6, method="exact") for c in cs])
    assert np.allclose(vals_grid, vals_scalar, rtol=1e-12, atol=1e-14)


def test_sf_grid_monotone_exact():
    cs = np.linspace(0.0, 1.0, 51)
    vals = tra.sf_grid(cs, n=50, k=8, method="exact")
    assert np.all(np.diff(vals) <= 1e-12)


def test_sf_grid_matches_scalar_simplex():
    cs = np.linspace(0.01, 0.99, 21)
    vals_grid = tra.sf_grid(cs, n=20, k=5, method="simplex")
    vals_scalar = np.array([tra.sf(float(c), n=20, k=5, method="simplex") for c in cs])
    assert np.allclose(vals_grid, vals_scalar, rtol=1e-12, atol=1e-14)


def test_sf_grid_endpoints():
    cs = np.array([0.0, 0.25, 0.75, 1.0])
    vals = tra.sf_grid(cs, n=25, k=4, method="exact")
    assert vals[0] == 1.0
    assert vals[-1] == 0.0
