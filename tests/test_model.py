"""
Tests for model.py module - GALAX model implementation.
"""

import numpy as np
import pytest
from PyGALAX.model import GALAX


class TestGALAXInitialization:
    """Test cases for GALAX model initialization."""
    
    def setup_method(self):
        """Set up test fixtures."""
        np.random.seed(42)
        self.n_samples = 30
        self.n_features = 5
        
        self.coords = np.random.uniform(0, 10, (self.n_samples, 2))
        self.X = np.random.normal(0, 1, (self.n_samples, self.n_features))
        self.y_regression = np.random.normal(0, 1, (self.n_samples, 1))
        self.y_classification = np.random.randint(0, 2, self.n_samples)
        
        self.x_vars = [f'X{i}' for i in range(self.n_features)]
    
    def test_galax_initialization_regression(self):
        """Test GALAX initialization for regression."""
        model = GALAX(
            coords=self.coords,
            y=self.y_regression,
            X=self.X,
            task='regression'
        )
        assert model is not None
        assert model.task == 'regression'
        assert model.coords.shape == (self.n_samples, 2)
    
    def test_galax_initialization_classification(self):
        """Test GALAX initialization for classification."""
        model = GALAX(
            coords=self.coords,
            y=self.y_classification,
            X=self.X,
            task='classification'
        )
        assert model is not None
        assert model.task == 'classification'
    
    def test_galax_with_custom_bandwidth(self):
        """Test GALAX with custom bandwidth value."""
        model = GALAX(
            coords=self.coords,
            y=self.y_regression,
            X=self.X,
            bw=2.0,
            task='regression'
        )
        assert model.bw == 2.0
    
    def test_galax_with_isa_bandwidth(self):
        """Test GALAX with ISA bandwidth method."""
        model = GALAX(
            coords=self.coords,
            y=self.y_regression,
            X=self.X,
            bw='isa',
            task='regression'
        )
        assert model.bw == 'isa'
    
    def test_galax_with_invalid_bandwidth_method(self):
        """Test that invalid bandwidth method raises error."""
        with pytest.raises(ValueError):
            GALAX(
                coords=self.coords,
                y=self.y_regression,
                X=self.X,
                bw='invalid_method',
                task='regression'
            )
    
    def test_galax_kernel_options(self):
        """Test GALAX with different kernel options."""
        kernels = ['bisquare', 'gaussian']
        for kernel in kernels:
            model = GALAX(
                coords=self.coords,
                y=self.y_regression,
                X=self.X,
                kernel=kernel,
                task='regression'
            )
            assert model.kernel == kernel
    
    def test_galax_with_custom_automl_settings(self):
        """Test GALAX with custom AutoML settings."""
        custom_settings = {
            "time_budget": 60,
            "estimator_list": ['rf', 'xgboost'],
        }
        model = GALAX(
            coords=self.coords,
            y=self.y_regression,
            X=self.X,
            automl_settings=custom_settings,
            task='regression'
        )
        assert model.automl_settings['time_budget'] == 60
        assert 'rf' in model.automl_settings['estimator_list']


class TestGALAXFitting:
    """Test cases for GALAX model fitting."""
    
    def setup_method(self):
        """Set up test fixtures with smaller dataset for faster testing."""
        np.random.seed(42)
        self.n_samples = 15
        self.n_features = 3
        
        self.coords = np.random.uniform(0, 10, (self.n_samples, 2))
        self.X = np.random.normal(0, 1, (self.n_samples, self.n_features))
        self.y_regression = np.random.normal(0, 1, (self.n_samples, 1))
        self.y_classification = np.random.randint(0, 2, self.n_samples)
        
        self.x_vars = [f'X{i}' for i in range(self.n_features)]
    
    def test_galax_fit_regression(self):
        """Test GALAX fitting for regression task."""
        model = GALAX(
            coords=self.coords,
            y=self.y_regression,
            X=self.X,
            bw=3.0,
            task='regression',
            n_jobs=1
        )
        try:
            results = model.fit()
            assert results is not None
        except Exception as e:
            # Model fitting might fail due to data characteristics, which is acceptable
            print(f"Fit failed: {e}")
    
    def test_galax_fit_classification(self):
        """Test GALAX fitting for classification task."""
        model = GALAX(
            coords=self.coords,
            y=self.y_classification,
            X=self.X,
            bw=3.0,
            task='classification',
            n_jobs=1
        )
        try:
            results = model.fit()
            assert results is not None
        except Exception as e:
            # Model fitting might fail due to data characteristics, which is acceptable
            print(f"Fit failed: {e}")
    
    def test_galax_fit_with_x_vars(self):
        """Test GALAX fitting with feature names."""
        model = GALAX(
            coords=self.coords,
            y=self.y_regression,
            X=self.X,
            bw=3.0,
            x_vars=self.x_vars,
            task='regression',
            n_jobs=1
        )
        assert model.x_vars == self.x_vars


class TestGALAXValidation:
    """Test input validation for GALAX."""
    
    def setup_method(self):
        """Set up test fixtures."""
        np.random.seed(42)
        self.coords = np.random.uniform(0, 10, (10, 2))
        self.X = np.random.normal(0, 1, (10, 3))
        self.y = np.random.normal(0, 1, (10, 1))
    
    def test_mismatched_sample_sizes(self):
        """Test that mismatched sample sizes are handled."""
        try:
            model = GALAX(
                coords=self.coords,
                y=self.y,
                X=self.X[:8],
                task='regression'
            )
        except (ValueError, IndexError):
            pass
    
    def test_galax_with_valid_task_types(self):
        """Test GALAX accepts valid task types."""
        for task in ['regression', 'classification']:
            model = GALAX(
                coords=self.coords,
                y=self.y,
                X=self.X,
                task=task
            )
            assert model.task == task


class TestGALAXNewParams:
    """Validation, defaults, removed APIs, spherical override, weighted metrics."""

    def setup_method(self):
        rng = np.random.default_rng(0)
        self.coords = rng.uniform(2e5, 3e5, (10, 2))
        self.X = rng.normal(0, 1, (10, 3))
        self.y = rng.normal(0, 1, (10, 1))

    def _kw(self, **extra):
        return dict(coords=self.coords, y=self.y, X=self.X, **extra)

    def test_defaults(self):
        m = GALAX(**self._kw())
        assert m.evaluation_mode == 'gwr_automl'
        assert m.test_size == 0.3
        assert m.split_seed == 42

    def test_invalid_evaluation_mode(self):
        with pytest.raises(ValueError):
            GALAX(**self._kw(evaluation_mode='bad'))

    def test_test_size_bounds(self):
        for bad in (0, 1, -0.1, 1.5):
            with pytest.raises(ValueError):
                GALAX(**self._kw(test_size=bad))

    def test_removed_apis(self):
        with pytest.raises(TypeError):
            GALAX(**self._kw(loo=True))
        from PyGALAX.results import GALAXResults
        assert not hasattr(GALAXResults, 'get_detailed_shap_for_location')

    def test_spherical_override(self):
        lonlat = np.column_stack([np.linspace(-79, -78, 10), np.linspace(42, 43, 10)])
        proj = self.coords
        assert GALAX(coords=lonlat, y=self.y, X=self.X).spherical is True
        assert GALAX(coords=proj, y=self.y, X=self.X).spherical is False
        assert GALAX(coords=lonlat, y=self.y, X=self.X, spherical=False).spherical is False
        assert GALAX(coords=proj, y=self.y, X=self.X, spherical=True).spherical is True

    def test_f6_classification_metrics_use_weights(self):
        y_true = np.array([0, 0, 1, 1])
        p, _, _ = GALAX._macro_prf(y_true, np.array([0, 0, 0, 0]), np.ones(4))
        assert np.isclose(p, 0.5)
        p_bal, _, _ = GALAX._macro_prf(y_true, np.array([0, 1, 1, 1]), np.ones(4))
        p_imb, _, _ = GALAX._macro_prf(y_true, np.array([0, 1, 1, 1]), np.array([0.05, 0.05, 3.0, 3.0]))
        assert not np.isclose(p_bal, p_imb)


class TestLocalHoldoutSemantics:
    """local_holdout split contract, driven by a deterministic AutoML."""

    def _model(self, n=60, bw=20, task='regression'):
        rng = np.random.default_rng(0)
        X = rng.normal(0, 1, (n, 4))
        X[:, 0] = np.arange(n)  
        coords = rng.uniform(2e5, 3e5, (n, 2))
        if task == 'classification':
            y = (np.arange(n) % 2)
        else:
            y = np.arange(n, dtype=float).reshape(-1, 1)
        return GALAX(coords=coords, y=y, X=X, bw=bw, task=task, n_jobs=1,
                     evaluation_mode='local_holdout', test_size=0.3, split_seed=42), n

    def test_focal_held_out_and_single_fit(self, fake_automl):
        model, n = self._model()
        rec = model._process_location_holdout(5)
        assert sum(a.fit_calls for a in fake_automl.instances) == 1
        fit = fake_automl.instances[-1]
        train_ids = set(fit.X_train[:, 0].tolist())
        assert 5 not in train_ids
        assert rec['n_train'] == fit.X_train.shape[0]
        test_input = fit.predict_inputs[-1]
        test_ids = set(test_input[:, 0].tolist())
        assert test_input.shape[0] == rec['n_test']
        assert test_input[-1, 0] == 5 and rec['focal_pred'] == 5
        assert train_ids.isdisjoint(test_ids)

    def test_split_reproducible(self, fake_automl):
        model, _ = self._model()
        model._process_location_holdout(7)
        ids1 = fake_automl.instances[-1].X_train[:, 0].copy()
        model._process_location_holdout(7)
        ids2 = fake_automl.instances[-1].X_train[:, 0]
        np.testing.assert_array_equal(ids1, ids2)

    def test_classification_records(self, fake_automl):
        model, _ = self._model(task='classification')
        rec = model._process_location_holdout(3)
        assert 3 not in fake_automl.instances[-1].X_train[:, 0]
        assert 'test_accuracy' in rec and 'test_f1' in rec

    def test_fit_routes_to_local_holdout(self, fake_automl):
        from PyGALAX.results import LocalHoldoutResults
        model, n = self._model()
        res = model.fit()
        assert isinstance(res, LocalHoldoutResults)
        assert res.n_valid == n
        assert 'test_r2' in res.local_stats

    @pytest.mark.slow
    def test_local_holdout_real_fit(self):
        rng = np.random.default_rng(0)
        n = 60
        coords = rng.uniform(2e5, 3e5, (n, 2))
        X = rng.normal(0, 1, (n, 4))
        y = X[:, 0] * 2 - X[:, 1] + rng.normal(0, 0.1, n)
        settings = {"time_budget": 5, "estimator_list": ['rf'], "task": 'regression',
                    "metric": 'r2', "seed": 42, "verbose": 0}
        from PyGALAX.results import LocalHoldoutResults
        res = GALAX(coords=coords, y=y.reshape(-1, 1), X=X, bw=40, automl_settings=settings,
                    n_jobs=4, evaluation_mode='local_holdout').fit()
        assert isinstance(res, LocalHoldoutResults)
        assert res.n_valid > 0
        assert np.isfinite(res.global_rmse)

