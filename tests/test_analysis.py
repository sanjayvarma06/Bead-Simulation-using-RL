"""Statistics helpers agree with reference implementations."""
import numpy as np
import pytest
from scipy.stats import wilcoxon
from analyze import wilcoxon_exact, holm


@pytest.mark.parametrize("seed", range(10))
def test_exact_wilcoxon_matches_scipy(seed):
    d = np.random.default_rng(seed).normal(0.3, 1.0, 8)
    p, _ = wilcoxon_exact(d)
    assert p == pytest.approx(wilcoxon(d, method="exact").pvalue, rel=1e-9)


def test_smallest_p_for_n8():
    assert wilcoxon_exact(np.arange(1, 9))[0] == pytest.approx(2 / 256)


def test_holm():
    adj = holm(np.array([0.01, 0.04, 0.03, 0.5]))
    assert np.allclose(adj, [0.04, 0.09, 0.09, 0.5])
