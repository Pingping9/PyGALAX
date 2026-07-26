"""
Results handling for GALAX models.
"""

import numpy as np
np.float = float
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, r2_score, mean_squared_error
from joblib import dump


class GALAXResults:
    """
    Results class for GALAX models.
    
    Parameters
    ----------
    model : GALAX
        The fitted GALAX model instance
    results : list
        List of location-specific results from model fitting
        
    Attributes
    ----------
    model : GALAX
        The original GALAX model
    results : list
        Successful location results
    params : array
        Global R² score (regression only)
    local_metrics : array
        Local performance metrics (R² for regression, accuracy for classification)
    global_r2 : float
        Global R² score (regression only)
    global_rmse : float
        Global RMSE (regression only)
    global_accuracy : float
        Global accuracy (classification only)
    global_precision : float
        Global precision (classification only)
    global_recall : float
        Global recall (classification only)
    global_f1 : float
        Global F1 score (classification only)
    """
    def __init__(self, model, results):
        self.model = model
        self.results = results
        self._process_results()

        if self.model.task == 'regression':
            self.local_rmse = np.array([r['local_rmse'] for r in self.results])

    def _process_results(self):
        """Process and aggregate results"""
        self.params = np.array([r['prediction'] for r in self.results])
        self.local_metrics = np.array([r['local_metric'] for r in self.results])

        self.raw_shap_values_neighbors = []
        self.X_neighbors_values = []
        self.y_neighbors_values = []
        self.weights_neighbors = []
        self.location_original_indices = [r['location_index'] for r in self.results]

        for r in self.results:
            if isinstance(r['raw_shap_values_neighbors'], list) and all(isinstance(item, list) for item in r['raw_shap_values_neighbors']):
                self.raw_shap_values_neighbors.append([np.array(s) for s in r['raw_shap_values_neighbors']])
            else:
                self.raw_shap_values_neighbors.append(np.array(r['raw_shap_values_neighbors']))

            self.X_neighbors_values.append(np.array(r['X_neighbors_values']))
            self.y_neighbors_values.append(np.array(r['y_neighbors_values']))
            self.weights_neighbors.append(np.array(r['weights_neighbors']))

        if self.model.task == 'classification':
            self.local_precision = np.array([r['precision'] for r in self.results])
            self.local_recall = np.array([r['recall'] for r in self.results])
            self.local_f1 = np.array([r['f1'] for r in self.results])

            y_pred = self.params
            y_true = self.model.y[self.location_original_indices]

            if len(y_true) == 0 or len(y_pred) == 0:
                self.global_accuracy = np.nan
                self.global_precision = np.nan
                self.global_recall = np.nan
                self.global_f1 = np.nan
            else:
                self.global_accuracy = accuracy_score(y_true, y_pred)
                self.global_precision = precision_score(y_true, y_pred, average='weighted', zero_division=np.nan)
                self.global_recall = recall_score(y_true, y_pred, average='weighted', zero_division=np.nan)
                self.global_f1 = f1_score(y_true, y_pred, average='weighted', zero_division=np.nan)
        else:
            y_pred = self.params.reshape(-1, 1)
            y_true = self.model.y[self.location_original_indices].reshape(-1, 1)

            if len(y_true) == 0 or len(y_pred) == 0:
                self.global_r2 = np.nan
                self.global_rmse = np.nan
            else:
                self.global_r2 = r2_score(y_true, y_pred)
                self.global_rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    def summary(self):
        """Print summary statistics"""
        print(f"GALAX Model Results Summary")
        print("-" * 50)
        print(f"Task: {self.model.task}")
        print(f"Bandwidth: {self.model.bw}")
        print(f"Kernel function: {self.model.kernel}")
        print(f"Bandwidth type: {'Fixed' if self.model.fixed else 'Adaptive'}")

        if self.model.task == 'classification':
            print(f"Global Accuracy: {self.global_accuracy:.4f}")
            print(f"Global Precision: {self.global_precision:.4f}")
            print(f"Global Recall: {self.global_recall:.4f}")
            print(f"Global F1 Score: {self.global_f1:.4f}")
            print(f"\nLocal Precision Statistics:")
            print(f"  - Mean: {np.mean(self.local_precision):.4f}")
            print(f"  - Min: {np.min(self.local_precision):.4f}")
            print(f"  - Max: {np.max(self.local_precision):.4f}")
            print(f"  - Std: {np.std(self.local_precision):.4f}")
            print(f"\nLocal Recall Statistics:")
            print(f"  - Mean: {np.mean(self.local_recall):.4f}")
            print(f"  - Min: {np.min(self.local_recall):.4f}")
            print(f"  - Max: {np.max(self.local_recall):.4f}")
            print(f"  - Std: {np.std(self.local_recall):.4f}")
            print(f"\nLocal F1 Statistics:")
            print(f"  - Mean: {np.mean(self.local_f1):.4f}")
            print(f"  - Min: {np.min(self.local_f1):.4f}")
            print(f"  - Max: {np.max(self.local_f1):.4f}")
            print(f"  - Std: {np.std(self.local_f1):.4f}")
        else:
            print(f"Global R²: {self.global_r2:.4f}")
            print(f"Global RMSE: {self.global_rmse:.4f}")
            print(f"\nLocal R² Statistics:")
            print(f"  - Mean: {np.mean(self.local_metrics):.4f}")
            print(f"  - Min: {np.min(self.local_metrics):.4f}")
            print(f"  - Max: {np.max(self.local_metrics):.4f}")
            print(f"  - Std: {np.std(self.local_metrics):.4f}")
            print(f"\nLocal RMSE Statistics:")
            print(f"  - Mean: {np.mean(self.local_rmse):.4f}")
            print(f"  - Min: {np.min(self.local_rmse):.4f}")
            print(f"  - Max: {np.max(self.local_rmse):.4f}")
            print(f"  - Std: {np.std(self.local_rmse):.4f}")

    def save_results(self, filename):
        """
        Save results to file.
        
        Parameters
        ----------
        filename : str
            Path to save results (should end with .joblib)
        """
        successful_results = [r for r in self.results if r is not None]
        total_locations = len(self.model.coords)
        successful_locations = len(successful_results)
        results_dict = {
            'task': self.model.task,
            'bandwidth': self.model.bw,
            'kernel': self.model.kernel,
            'fixed': self.model.fixed,
            'predictions': self.params.tolist(),
            'coords': self.model.coords.tolist(),
            'location_results': successful_results,
            'x_variables': self.model.x_vars if self.model.x_vars else [],
            'total_locations': total_locations,
            'successful_locations': successful_locations
        }

        if self.model.task == 'classification':
            results_dict.update({
                'global_accuracy': self.global_accuracy,
                'global_precision': self.global_precision,
                'global_recall': self.global_recall,
                'global_f1': self.global_f1,
                'local_accuracy': self.local_metrics.tolist(),
                'local_precision': self.local_precision.tolist(),
                'local_recall': self.local_recall.tolist(),
                'local_f1': self.local_f1.tolist()
            })
        else:
            results_dict.update({
                'global_r2': self.global_r2,
                'global_rmse': self.global_rmse,
                'local_r2': self.local_metrics.tolist(),
                'local_rmse': self.local_rmse.tolist()
            })
        dump(results_dict, filename)
        print(f"Results saved to {filename}")


class LocalHoldoutResults:
    """
    Results for GALAX fitted with evaluation_mode='local_holdout'.

    Parameters
    ----------
    model : GALAX
        The fitted GALAX model instance
    results : list
        List of per-location holdout results from model fitting
    """

    def __init__(self, model, results):
        self.model = model
        self.n_attempted = len(results)
        self.results = [r for r in results if r is not None]
        self.n_valid = len(self.results)
        self.test_size = model.test_size
        self.split_seed = model.split_seed
        self._aggregate()

    @staticmethod
    def _stats(values):
        vals = np.asarray([v for v in values if v is not None and np.isfinite(v)], dtype=float)
        if vals.size == 0:
            return {'mean': np.nan, 'min': np.nan, 'max': np.nan, 'std': np.nan, 'n': 0}
        return {'mean': float(np.mean(vals)), 'min': float(np.min(vals)),
                'max': float(np.max(vals)), 'std': float(np.std(vals)), 'n': int(vals.size)}

    def _aggregate(self):
        task = self.model.task
        if self.n_valid == 0:
            self.local_stats = {}
            if task == 'classification':
                self.global_accuracy = self.global_precision = self.global_recall = self.global_f1 = np.nan
            else:
                self.global_r2 = self.global_rmse = np.nan
            return

        focal_pred = np.array([r['focal_pred'] for r in self.results])
        focal_true = np.array([r['focal_true'] for r in self.results])

        if task == 'classification':
            keys = ['train_accuracy', 'test_accuracy', 'train_precision', 'test_precision',
                    'train_recall', 'test_recall', 'train_f1', 'test_f1']
            self.local_stats = {k: self._stats([r[k] for r in self.results]) for k in keys}
            self.global_accuracy = float(accuracy_score(focal_true, focal_pred))
            self.global_precision = float(precision_score(focal_true, focal_pred, average='weighted', zero_division=np.nan))
            self.global_recall = float(recall_score(focal_true, focal_pred, average='weighted', zero_division=np.nan))
            self.global_f1 = float(f1_score(focal_true, focal_pred, average='weighted', zero_division=np.nan))
        else:
            keys = ['train_r2', 'test_r2', 'train_rmse', 'test_rmse']
            self.local_stats = {k: self._stats([r[k] for r in self.results]) for k in keys}
            yt = focal_true.astype(float)
            yp = focal_pred.astype(float)
            self.global_r2 = float(r2_score(yt, yp)) if yt.size >= 2 else np.nan
            self.global_rmse = float(np.sqrt(mean_squared_error(yt, yp))) if yt.size >= 1 else np.nan

    def summary(self):
        print("GALAX Local-Holdout Results Summary")
        print("-" * 50)
        print(f"Task: {self.model.task}")
        print(f"Bandwidth: {self.model.bw}")
        print(f"Kernel function: {self.model.kernel}")
        print(f"Bandwidth type: {'Fixed' if self.model.fixed else 'Adaptive'}")
        print(f"Evaluation mode: local_holdout (test_size={self.test_size}, split_seed={self.split_seed})")
        print(f"Valid locations: {self.n_valid} / {self.n_attempted}")
        if self.n_valid == 0:
            print("No valid locations; all holdout metrics are undefined.")
            return

        def _pstats(title, s):
            print(f"\n{title} (valid: {s['n']} / {self.n_valid}):")
            print(f"  - Mean: {s['mean']:.4f}")
            print(f"  - Min: {s['min']:.4f}")
            print(f"  - Max: {s['max']:.4f}")
            print(f"  - Std: {s['std']:.4f}")

        print(f"\nGlobal focal-holdout metrics (pooled, out-of-sample):")
        if self.model.task == 'classification':
            print(f"  Accuracy:  {self.global_accuracy:.4f}")
            print(f"  Precision: {self.global_precision:.4f}")
            print(f"  Recall:    {self.global_recall:.4f}")
            print(f"  F1 Score:  {self.global_f1:.4f}")
        else:
            print(f"  R²:   {self.global_r2:.4f}")
            print(f"  RMSE: {self.global_rmse:.4f}")

        for key, s in self.local_stats.items():
            _pstats("Local " + key.replace('_', ' '), s)

    def save_results(self, filename):
        """Save local-holdout results to a joblib file."""
        results_dict = {
            'evaluation_mode': 'local_holdout',
            'task': self.model.task,
            'bandwidth': self.model.bw,
            'kernel': self.model.kernel,
            'fixed': self.model.fixed,
            'test_size': self.test_size,
            'split_seed': self.split_seed,
            'n_attempted': self.n_attempted,
            'n_valid': self.n_valid,
            'coords': self.model.coords.tolist(),
            'x_variables': self.model.x_vars if self.model.x_vars else [],
            'location_results': self.results,
            'local_stats': self.local_stats,
        }
        if self.model.task == 'classification':
            results_dict.update({
                'global_accuracy': self.global_accuracy,
                'global_precision': self.global_precision,
                'global_recall': self.global_recall,
                'global_f1': self.global_f1,
            })
        else:
            results_dict.update({
                'global_r2': self.global_r2,
                'global_rmse': self.global_rmse,
            })
        dump(results_dict, filename)
        print(f"Results saved to {filename}")
