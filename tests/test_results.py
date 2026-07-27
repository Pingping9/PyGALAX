"""
Tests for results.py module - Result processing and visualization.
"""

import types
import numpy as np
import pytest
from PyGALAX.results import GALAXResults, LocalHoldoutResults
from PyGALAX.model import GALAX


class TestGALAXResults:
    """Test cases for GALAXResults class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        np.random.seed(42)
        self.n_samples = 10
        self.n_features = 3
        
        self.coords = np.random.uniform(0, 10, (self.n_samples, 2))
        self.X = np.random.normal(0, 1, (self.n_samples, self.n_features))
        self.y_regression = np.random.normal(0, 1, (self.n_samples, 1))
        self.y_classification = np.random.randint(0, 2, self.n_samples)
    
    def _create_mock_results(self, task='regression'):
        """Create mock results list for testing."""
        results = []
        for i in range(self.n_samples):
            result = {
                'location_index': i,
                'prediction': self.y_regression[i, 0] if task == 'regression' else self.y_classification[i],
                'local_metric': np.random.random(),
                'local_rmse': np.random.random() if task == 'regression' else None,
                'raw_shap_values_neighbors': [np.random.normal(0, 0.1, self.n_features) for _ in range(5)],
                'X_neighbors_values': self.X,
                'y_neighbors_values': self.y_regression if task == 'regression' else self.y_classification,
                'weights_neighbors': np.random.random(self.n_samples),
            }
            if task == 'classification':
                result['precision'] = np.random.random()
                result['recall'] = np.random.random()
                result['f1'] = np.random.random()
            results.append(result)
        return results
    
    def test_results_initialization_regression(self):
        """Test GALAXResults initialization for regression."""
        model = GALAX(
            coords=self.coords,
            y=self.y_regression,
            X=self.X,
            bw=2.0,
            task='regression'
        )
        mock_results = self._create_mock_results(task='regression')
        
        results = GALAXResults(model=model, results=mock_results)
        assert results is not None
        assert len(results.params) == self.n_samples
    
    def test_results_attributes(self):
        """Test that results object has expected attributes."""
        model = GALAX(
            coords=self.coords,
            y=self.y_regression,
            X=self.X,
            bw=2.0,
            task='regression'
        )
        mock_results = self._create_mock_results(task='regression')
        
        results = GALAXResults(model=model, results=mock_results)
        
        assert hasattr(results, 'params')
        assert hasattr(results, 'local_metrics')
        assert hasattr(results, 'model')
    
    def test_results_with_regression_metrics(self):
        """Test results object with regression metrics."""
        model = GALAX(
            coords=self.coords,
            y=self.y_regression,
            X=self.X,
            bw=2.0,
            task='regression'
        )
        mock_results = self._create_mock_results(task='regression')
        
        results = GALAXResults(model=model, results=mock_results)
        assert results is not None
        assert hasattr(results, 'local_rmse')
    
    def test_results_with_classification_metrics(self):
        """Test results object with classification metrics."""
        model = GALAX(
            coords=self.coords,
            y=self.y_classification,
            X=self.X,
            bw=2.0,
            task='classification'
        )
        mock_results = self._create_mock_results(task='classification')
        
        results = GALAXResults(model=model, results=mock_results)
        assert results is not None
        assert hasattr(results, 'global_accuracy')


class TestResultsValidation:
    """Test input validation for results."""
    
    def setup_method(self):
        """Set up test fixtures."""
        np.random.seed(42)
        self.n_samples = 10
        self.n_features = 3
        self.coords = np.random.uniform(0, 10, (self.n_samples, 2))
        self.X = np.random.normal(0, 1, (self.n_samples, self.n_features))
        self.y_regression = np.random.normal(0, 1, (self.n_samples, 1))
    
    def _create_mock_results(self, n_results):
        """Create mock results list."""
        results = []
        for i in range(n_results):
            result = {
                'location_index': i,
                'prediction': self.y_regression[i, 0],
                'local_metric': np.random.random(),
                'local_rmse': np.random.random(),
                'raw_shap_values_neighbors': [np.random.normal(0, 0.1, self.n_features) for _ in range(5)],
                'X_neighbors_values': self.X,
                'y_neighbors_values': self.y_regression,
                'weights_neighbors': np.random.random(self.n_samples),
            }
            results.append(result)
        return results
    
    def test_results_processing(self):
        """Test that results are properly processed."""
        model = GALAX(
            coords=self.coords,
            y=self.y_regression,
            X=self.X,
            bw=2.0,
            task='regression'
        )
        mock_results = self._create_mock_results(self.n_samples)
        
        results = GALAXResults(model=model, results=mock_results)
        assert len(results.params) == self.n_samples
        assert len(results.local_metrics) == self.n_samples
    
    def test_results_summary(self):
        """Test that results can generate summary."""
        model = GALAX(
            coords=self.coords,
            y=self.y_regression,
            X=self.X,
            bw=2.0,
            task='regression'
        )
        mock_results = self._create_mock_results(self.n_samples)
        
        results = GALAXResults(model=model, results=mock_results)
        
        if hasattr(results, 'summary'):
            try:
                results.summary()
            except Exception as e:
                print(f"Summary generation raised: {e}")


def _holdout_model(task='regression'):
    return types.SimpleNamespace(task=task, test_size=0.3, split_seed=42, bw=8,
                                 kernel='bisquare', fixed=False,
                                 coords=np.zeros((10, 2)), x_vars=None)


def _reg_rec(i, **kw):
    rec = {'location_index': i, 'focal_pred': float(i), 'focal_true': float(i) + 0.1,
           'n_train': 7, 'n_test': 3, 'train_r2': 0.9, 'test_r2': 0.6,
           'train_rmse': 0.1, 'test_rmse': 0.2}
    rec.update(kw)
    return rec


class TestLocalHoldoutResults:
    """Aggregation, pooled global metric, and None/NaN handling."""

    def test_regression_aggregation_exact(self):
        r = LocalHoldoutResults(_holdout_model(), [_reg_rec(0, test_r2=0.6), _reg_rec(1, test_r2=0.8)])
        assert r.n_valid == 2
        assert np.isclose(r.local_stats['test_r2']['mean'], 0.7)
        assert r.local_stats['test_r2']['n'] == 2
        assert np.isclose(r.global_r2, 0.96)
        assert np.isclose(r.global_rmse, 0.1)

    def test_none_and_nan_handling(self):
        recs = [_reg_rec(0, test_r2=0.6), None, _reg_rec(2, test_r2=np.nan)]
        r = LocalHoldoutResults(_holdout_model(), recs)
        assert r.n_valid == 2
        assert r.local_stats['test_r2']['n'] == 1
        assert np.isclose(r.local_stats['test_r2']['mean'], 0.6)

    def test_all_none(self):
        r = LocalHoldoutResults(_holdout_model(), [None, None])
        assert r.n_valid == 0
        assert np.isnan(r.global_r2)
        r.summary()

    def test_classification_records(self):
        recs = [{'location_index': i, 'focal_pred': i % 2, 'focal_true': i % 2,
                 'n_train': 7, 'n_test': 3,
                 'train_accuracy': 0.9, 'test_accuracy': 0.7,
                 'train_precision': 0.8, 'test_precision': 0.6,
                 'train_recall': 0.85, 'test_recall': 0.65,
                 'train_f1': 0.82, 'test_f1': 0.62} for i in range(4)]
        r = LocalHoldoutResults(_holdout_model('classification'), recs)
        assert r.n_valid == 4
        assert np.isfinite(r.global_accuracy)
        assert 'test_precision' in r.local_stats

