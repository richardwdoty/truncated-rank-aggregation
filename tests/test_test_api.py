import numpy as np

import tra


def test_test_matches_statistic_and_pvalue():
    rng = np.random.default_rng(0)
    p = rng.uniform(0, 1, size=40)
    k = 5

    res = tra.test(p, k)
    t = tra.statistic(p, k)
    pv = tra.pvalue(p, k)

    assert res.statistic == t
    assert res.pvalue == pv
    assert res.n == 40
    assert res.k == 5


def test_test_result_fields_in_range():
    rng = np.random.default_rng(1)
    p = rng.uniform(0, 1, size=25)

    res = tra.test(p, k=4)

    assert 0.0 <= res.statistic <= 1.0
    assert 0.0 <= res.pvalue <= 1.0