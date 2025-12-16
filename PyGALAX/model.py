"""
Core GALAX implementation.
"""

import os
import sys
import numpy as np
np.float = float
import pandas as pd
from joblib import Parallel, delayed
from flaml import AutoML
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from scipy import stats
import shap

from .kernel import Kernel
from .bandwidth import search_bw_lw_ISA, search_bandwidth
from .results import GALAXResults


class GALAX:
    """
    Geographically Weighted Regression/Classification with AutoML and eXplainable AI.
    
    Parameters
    ----------
    coords : array
        Spatial coordinates of observations
    y : array
        Dependent variable
    X : array
        Independent variables
    bw : int, float, str, or None, optional
        Bandwidth specification:
        - int/float: Use this specific bandwidth value
        - None: Use ISA method first, fall back to performance-based if ISA fails
        - 'isa': Use ISA method only
        - 'performance': Use performance-based optimization method only
    kernel : str, optional
        Kernel function type ('bisquare', 'gaussian', 'exponential', etc.)
    fixed : bool, optional
        Whether to use fixed (True) or adaptive (False) bandwidth
    automl_settings : dict, optional
        Settings for AutoML model
    n_jobs : int, optional
        Number of parallel jobs
    x_vars : list, optional
        Names of independent variables
    task : str, optional
        Type of task: 'regression' or 'classification'
    """
    def __init__(self, coords, y, X, bw=None, kernel='bisquare', fixed=False,
                 automl_settings=None, n_jobs=None, x_vars=None, task='regression'):
        self.coords = np.array(coords)
        self.y = np.array(y)
        self.X = np.array(X)
        self.bw = bw
        self.kernel = kernel
        self.fixed = fixed
        self.x_vars = x_vars
        self.task = task

        if isinstance(bw, str) and bw not in ['isa', 'performance']:
            raise ValueError(f"Invalid bandwidth method: '{bw}'. Must be 'isa' or 'performance'.")

        default_settings = {
            "time_budget": 180,
            "estimator_list": ['rf', 'xgboost', 'xgb_limitdepth', 'extra_tree'],
            "task": task,
            "metric": 'accuracy' if task == 'classification' else 'r2',
            "seed": 42,
            "verbose": 0,
        }
        self.automl_settings = {**default_settings, **(automl_settings or {})}
        total_cpus = os.cpu_count()
        default_jobs = int(total_cpus * 0.5)
        self.n_jobs = n_jobs if n_jobs is not None else default_jobs

    def _build_wi(self, i):
        """
        Build weight matrix for location i.
        
        Parameters
        ----------
        i : int
            Index of location
            
        Returns
        -------
        array
            Weight vector for location i
        """
        kernel_obj = Kernel(self.coords[i], self.coords, self.bw, fixed=self.fixed, function=self.kernel)
        return kernel_obj.kernel

    def fit(self):
        """
        Fit the GALAX model.
        
        Returns
        -------
        GALAXResults
            Results object containing model outputs and statistics
        """
        if isinstance(self.bw, (int, float)):
            print(f"Using provided bandwidth: {self.bw}")
        elif self.bw is None:
            print("No bandwidth provided. Starting bandwidth selection...")
            try:
                print("Attempting ISA bandwidth selection...")
                self.bw, moran_i, p_val = search_bw_lw_ISA(
                    X=self.X,
                    y=self.y,
                    coords=self.coords,
                    kernel=self.kernel,
                    fixed=self.fixed,
                    task=self.task,
                    min_samples_per_class=5
                )
                print("ISA bandwidth selection successful:")
                print(f"- Optimal bandwidth: {self.bw}")
                print(f"- Moran's I: {moran_i:.4f}")
                print(f"- p-value: {p_val:.4f}")
            except Exception as e:
                print(f"ISA bandwidth search failed: {str(e)}")
                print("Falling back to performance-based bandwidth search...")
                search_result = search_bandwidth(self.X, self.y, self.coords,
                                              self.automl_settings,
                                              kernel=self.kernel,
                                              fixed=self.fixed,
                                              n_jobs=self.n_jobs,
                                              task=self.task)
                self.bw = search_result['best_bandwidth']
                print("Performance-based bandwidth selection successful:")
                print(f"- Optimal bandwidth: {self.bw}")
                print(f"- Optimization metric: {search_result['metric']}")
        elif self.bw == 'isa':
            try:
                print("Starting ISA bandwidth selection...")
                self.bw, moran_i, p_val = search_bw_lw_ISA(
                    X=self.X,
                    y=self.y,
                    coords=self.coords,
                    kernel=self.kernel,
                    fixed=self.fixed,
                    task=self.task,
                    min_samples_per_class=5
                )
                print("ISA bandwidth selection successful:")
                print(f"- Optimal bandwidth: {self.bw}")
                print(f"- Moran's I: {moran_i:.4f}")
                print(f"- p-value: {p_val:.4f}")
            except Exception as e:
                raise ValueError(f"ISA bandwidth search failed: {str(e)}")
        elif self.bw == 'performance':
            try:
                print("Starting performance-based bandwidth selection...")
                search_result = search_bandwidth(self.X, self.y, self.coords,
                                              self.automl_settings,
                                              kernel=self.kernel,
                                              fixed=self.fixed,
                                              n_jobs=self.n_jobs,
                                              task=self.task)
                self.bw = search_result['best_bandwidth']
                print("Performance-based bandwidth selection successful:")
                print(f"- Optimal bandwidth: {self.bw}")
                print(f"- Optimization metric: {search_result['metric']}")
            except Exception as e:
                raise ValueError(f"Performance-based bandwidth search failed: {str(e)}")

        # Process all locations in parallel
        results = Parallel(n_jobs=self.n_jobs)(
            delayed(self._process_location)(i)
            for i in range(len(self.y))
        )

        successful_results = [r for r in results if r is not None]
        total_locations = self.X.shape[0]
        print(f"Successfully processed locations: {len(successful_results)} / {total_locations}")

        return GALAXResults(self, results)

    def _process_location(self, i):
        """Process a single location"""
        try:
            weights_i = self._build_wi(i)
            neighbors_indices = np.where(weights_i > 0)[0]

            X_neighbors = self.X[neighbors_indices]
            y_neighbors = self.y[neighbors_indices]
            weights_neighbors = weights_i[neighbors_indices]

            automl = AutoML()
            automl.fit(X_neighbors, y_neighbors.ravel(), sample_weight=weights_neighbors, **self.automl_settings)

            y_pred_neighbors = automl.predict(X_neighbors)

            explainer = shap.TreeExplainer(automl.model.estimator)
            raw_shap_values = explainer.shap_values(X_neighbors)

            if isinstance(raw_shap_values, list):
                raw_shap_values_serializable = [s.tolist() for s in raw_shap_values]
            else:
                raw_shap_values_serializable = raw_shap_values.tolist()

            X_neighbors_serializable = X_neighbors.tolist()

            if self.task == 'classification':
                weighted_acc = np.sum(weights_neighbors * (y_neighbors.ravel() == y_pred_neighbors)) / np.sum(weights_neighbors)

                labels = np.unique(np.concatenate([y_neighbors, y_pred_neighbors]))
                labels = labels[~pd.isna(labels)]

                precision_per_class = precision_score(y_neighbors, y_pred_neighbors,
                                                      average=None, labels=labels, zero_division=np.nan)
                recall_per_class = recall_score(y_neighbors, y_pred_neighbors,
                                                average=None, labels=labels, zero_division=np.nan)
                f1_per_class = f1_score(y_neighbors, y_pred_neighbors,
                                        average=None, labels=labels, zero_division=np.nan)

                precision = np.nanmean(precision_per_class)
                recall = np.nanmean(recall_per_class)
                f1 = np.nanmean(f1_per_class)

                local_metric = weighted_acc
                additional_metrics = {
                    'precision': precision,
                    'recall': recall,
                    'f1': f1,
                    'precision_per_class': precision_per_class.tolist(),
                    'recall_per_class': recall_per_class.tolist(),
                    'f1_per_class': f1_per_class.tolist(),
                    'classes_present': labels.tolist()
                }
            else:
                y_bar_i = np.sum(weights_neighbors * y_neighbors.ravel()) / np.sum(weights_neighbors)
                TSS_i = np.sum(weights_neighbors * (y_neighbors.ravel() - y_bar_i) ** 2)
                RSS_i = np.sum(weights_neighbors * (y_neighbors.ravel() - y_pred_neighbors) ** 2)
                local_r2_i = 1 - (RSS_i / TSS_i) if TSS_i != 0 else 0
                local_rmse_i = np.sqrt(
                    np.sum(weights_neighbors * (y_neighbors.ravel() - y_pred_neighbors) ** 2) /
                    np.sum(weights_neighbors)
                )
                local_metric = local_r2_i
                additional_metrics = {
                    'local_rmse': local_rmse_i
                }

            pred_i = automl.predict(self.X[i].reshape(1, -1))[0]

            location_results = {
                'location_index': i,
                'model': automl.model.estimator,
                'estimator_name': automl.best_estimator,
                'local_metric': local_metric,
                'prediction': pred_i,
                'raw_shap_values_neighbors': raw_shap_values_serializable,
                'X_neighbors_values': X_neighbors_serializable,
                'y_neighbors_values': y_neighbors.tolist(),
                'weights_neighbors': weights_neighbors.tolist(),
            }
            location_results.update(additional_metrics)
            print(f"Location {i}/{self.X.shape[0]} successfully trained ML model")

            return location_results

        except Exception as e:
            print(f"Error at location {i}: {str(e)}", file=sys.stderr)
            return None
