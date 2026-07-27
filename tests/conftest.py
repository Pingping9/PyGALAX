"""Shared test fixtures for PyGALAX."""

import types
import numpy as np
import pytest


class FakeAutoML:
    """Deterministic stand-in for flaml.AutoML.
    """
    instances = []

    def __init__(self):
        self.model = types.SimpleNamespace(estimator=object())
        self.best_estimator = 'rf'
        self.X_train = None
        self.sample_weight = None
        self.fit_calls = 0
        self.predict_inputs = []
        FakeAutoML.instances.append(self)

    def fit(self, X, y, sample_weight=None, **kwargs):
        self.fit_calls += 1
        self.X_train = np.asarray(X)
        self.sample_weight = None if sample_weight is None else np.asarray(sample_weight)

    def predict(self, X):
        X = np.asarray(X)
        self.predict_inputs.append(X.copy())
        return X[:, 0]

    def score(self, X, y, sample_weight=None):
        return 0.0


@pytest.fixture
def fake_automl(monkeypatch):
    """Patch AutoML in model and bandwidth with the deterministic fake."""
    FakeAutoML.instances = []
    monkeypatch.setattr('PyGALAX.model.AutoML', FakeAutoML)
    monkeypatch.setattr('PyGALAX.bandwidth.AutoML', FakeAutoML)
    return FakeAutoML


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "slow: runs real FLAML (deselect with -m 'not slow')")
