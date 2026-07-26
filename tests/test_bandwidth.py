"""
Tests for bandwidth.py module.
"""

import numpy as np
import pytest
from PyGALAX.bandwidth import check_class_sizes, search_bw_lw_ISA, search_bandwidth


class TestCheckClassSizes:
    """Test cases for check_class_sizes function."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.y_regression = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        self.y_classification = np.array([0, 1, 0, 1, 0])
        self.weights = np.array([
            [1.0, 0.8, 0.0, 0.0, 0.0],
            [0.8, 1.0, 0.8, 0.0, 0.0],
            [0.0, 0.8, 1.0, 0.8, 0.0],
            [0.0, 0.0, 0.8, 1.0, 0.8],
            [0.0, 0.0, 0.0, 0.8, 1.0],
        ])
    
    def test_check_class_sizes_classification(self):
        """Test class size checking for classification task."""
        result = check_class_sizes(self.weights, self.y_classification, min_samples=2)
        assert isinstance(result, bool)
    
    def test_check_class_sizes_with_invalid_weights(self):
        """Test with weights that may have problematic locations."""
        weights_sparse = np.array([
            [1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.8, 0.0, 0.0],
            [0.0, 0.8, 1.0, 0.8, 0.0],
            [0.0, 0.0, 0.8, 1.0, 0.8],
            [0.0, 0.0, 0.0, 0.8, 1.0],
        ])
        result = check_class_sizes(weights_sparse, self.y_classification, min_samples=2)
        assert isinstance(result, bool)



class TestSearchBandwidth:
    """Bandwidth search: correct signatures, return types, and kernel weighting."""

    def setup_method(self):
        gx, gy = np.meshgrid(np.linspace(0, 9, 6), np.linspace(0, 9, 6))
        self.coords = np.column_stack([gx.ravel(), gy.ravel()])
        self.n = len(self.coords)
        self.y = self.coords[:, 0].astype(float)
        self.X = np.random.RandomState(0).normal(0, 1, (self.n, 3))

    def test_isa_returns_tuple(self):
        bw, moran_i, p_value = search_bw_lw_ISA(
            self.X, self.y, self.coords, kernel='bisquare', task='regression', spherical=False)
        assert bw > 0
        assert np.isfinite(moran_i)
        assert 0.0 <= p_value <= 1.0

    def test_search_bandwidth_returns_dict(self, fake_automl):
        settings = {"time_budget": 1, "estimator_list": ['rf'], "task": 'regression',
                    "metric": 'r2', "seed": 42, "verbose": 0}
        out = search_bandwidth(self.X, self.y, self.coords, settings, kernel='bisquare',
                               fixed=False, n_jobs=1, bw_min=8, bw_max=10, step=1,
                               task='regression', spherical=False)
        assert isinstance(out, dict)
        assert out['best_bandwidth'] > 0

    def test_f7_adaptive_uses_kernel_weights(self, fake_automl):
        settings = {"time_budget": 1, "estimator_list": ['rf'], "task": 'regression',
                    "metric": 'r2', "seed": 42, "verbose": 0}
        search_bandwidth(self.X, self.y, self.coords, settings, kernel='bisquare',
                         fixed=False, n_jobs=1, bw_min=8, bw_max=8, step=1,
                         task='regression', spherical=False)
        weights = [a.sample_weight for a in fake_automl.instances if a.sample_weight is not None]
        assert weights
        assert all(np.all(np.isfinite(w)) for w in weights)
        assert any(np.unique(w[w > 0]).size > 1 for w in weights)


def test_f2_fixed_isa_moran_gets_diagonal_free_weights(monkeypatch):
    import PyGALAX.bandwidth as bwmod
    captured = {}

    class FakeMoran:
        def __init__(self, y, w):
            captured['w'] = w
            self.I, self.z_norm, self.p_norm = 0.5, 3.0, 0.001

    monkeypatch.setattr(bwmod, 'Moran', FakeMoran)
    coords = np.column_stack([np.linspace(0, 10, 12), np.zeros(12)])
    y = coords[:, 0].astype(float)
    X = np.zeros((12, 2))
    bwmod.search_bw_lw_ISA(X, y, coords, fixed=True, bw_min=3, bw_max=5, step=1, task='regression', spherical=False)
    w = captured['w']
    assert sum(len(nbrs) for nbrs in w.neighbors.values()) > 0
    for i, neighbors in w.neighbors.items():
        assert i not in neighbors
