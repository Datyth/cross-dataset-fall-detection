import numpy as np

from src.domain_analysis import holm, kernel, mmd, permutation, prepare, c2st


def test_mmd_identity_and_shift():
    rng = np.random.default_rng(9)
    x = rng.normal(size=(25, 3))
    assert abs(mmd(kernel(np.vstack((x, x)), 1.), len(x))) < 1e-12
    value, p = permutation(np.vstack((x, x+5)), len(x), 1., rng)
    assert value > 0 and p < .01


def test_holm_and_missing_features():
    np.testing.assert_allclose(holm([.01, .04, .03]), [.03, .06, .06])
    x = prepare([[1, np.nan, 9], [2, 4, 9], [3, 6, 9]])
    assert x.shape == (3, 3)
    assert np.isfinite(x).all()


def test_c2st_equivalent_domains_grouped():
    rng = np.random.default_rng(111)
    rows = []
    for d in ("a", "b"):
        for label in ("0", "1"):
            for i in range(40):
                rows.append({"dataset": d, "label": label,
                             "group": f"{d}/{label}/{i}", "x": rng.normal()})
    result = c2st(rows, ["x"], rng)
    assert result["status"] == "ok"
    assert .35 < result["auc"] < .65
