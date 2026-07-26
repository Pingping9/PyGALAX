"""
Core GALAX implementation.
"""

import sys
import numpy as np
np.float = float
import pandas as pd
from joblib import Parallel, delayed
from flaml import AutoML
from sklearn.metrics import r2_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
import shap

from .kernel import Kernel, is_geographic
from .bandwidth import search_bw_lw_ISA, search_bandwidth
from .results import GALAXResults, LocalHoldoutResults

TREE_ESTIMATORS = {'rf', 'extra_tree', 'xgboost', 'xgb_limitdepth', 'lgbm', 'catboost', 'histgb'}


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
    evaluation_mode : str, optional
        'gwr_automl' (default) fits each location's full-neighborhood model and
        reports the usual local and global metrics.
        'local_holdout' splits every neighborhood into train/test, fits
        one model per location on the train neighbors, and reports held-out
        local metrics plus a pooled out-of-sample focal-holdout global metric.
    test_size : float, optional
        Fraction of each neighborhood held out for testing in 'local_holdout' mode (default 0.3). 
        The focal observation is always placed in the test set.
    split_seed : int, optional
        Random seed for the per-neighborhood train/test split (default 42).
    spherical : bool or None, optional
        Distance metric override. None (default) auto-detects geographic (longitude/latitude) coordinates 
        and uses great-circle distance for them, Euclidean otherwise. 
        Pass True to force great-circle distance or False to force Euclidean.
    """
    def __init__(self, coords, y, X, bw=None, kernel='bisquare', fixed=False, automl_settings=None, n_jobs=None, x_vars=None, task='regression',
                 evaluation_mode='gwr_automl', test_size=0.3, split_seed=42, spherical=None):
        self.coords = np.array(coords)
        self.y = np.array(y)
        self.X = np.array(X)
        self.bw = bw
        self.kernel = kernel
        self.fixed = fixed
        self.x_vars = x_vars
        self.task = task
        self.spherical = is_geographic(self.coords) if spherical is None else bool(spherical)

        if isinstance(bw, str) and bw not in ['isa', 'performance']:
            raise ValueError(f"Invalid bandwidth method: '{bw}'. Must be 'isa' or 'performance'.")
        if evaluation_mode not in ('gwr_automl', 'local_holdout'):
            raise ValueError(f"Invalid evaluation_mode: '{evaluation_mode}'. Must be 'gwr_automl' or 'local_holdout'.")
        if not 0 < test_size < 1:
            raise ValueError(f"test_size must be between 0 and 1, got {test_size}.")
        self.evaluation_mode = evaluation_mode
        self.test_size = test_size
        self.split_seed = split_seed

        default_settings = {
            "time_budget": 180,
            "estimator_list": ['rf', 'xgboost', 'xgb_limitdepth', 'extra_tree'],
            "task": task,
            "metric": 'accuracy' if task == 'classification' else 'r2',
            "seed": 42,
            "verbose": 0,
        }
        self.automl_settings = {**default_settings, **(automl_settings or {})}
        default_jobs = 4
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
        kernel_obj = Kernel(self.coords[i], self.coords, self.bw, fixed=self.fixed, function=self.kernel, spherical=self.spherical)
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
                    min_samples_per_class=5,
                    spherical=self.spherical
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
                                              task=self.task,
                                              spherical=self.spherical)
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
                    min_samples_per_class=5,
                    spherical=self.spherical
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
                                              task=self.task,
                                              spherical=self.spherical)
                self.bw = search_result['best_bandwidth']
                print("Performance-based bandwidth selection successful:")
                print(f"- Optimal bandwidth: {self.bw}")
                print(f"- Optimization metric: {search_result['metric']}")
            except Exception as e:
                raise ValueError(f"Performance-based bandwidth search failed: {str(e)}")

        if self.evaluation_mode == 'local_holdout':
            return self._fit_local_holdout()

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
        """
        Process a single location.

        Trains a full-neighborhood AutoML model for SHAP interpretation and local
        goodness-of-fit metrics, and predicts the focal location.
        """
        try:
            weights_i = self._build_wi(i)
            neighbors_indices = np.where(weights_i > 0)[0]

            X_neighbors = self.X[neighbors_indices]
            y_neighbors = self.y[neighbors_indices]
            weights_neighbors = weights_i[neighbors_indices]

            automl = AutoML()
            automl.fit(X_neighbors, y_neighbors.ravel(), sample_weight=weights_neighbors, **self.automl_settings)

            y_pred_neighbors = automl.predict(X_neighbors)

            if automl.best_estimator in TREE_ESTIMATORS:
                explainer = shap.TreeExplainer(automl.model.estimator)
                raw_shap_values = explainer.shap_values(X_neighbors)
            else:
                # model-agnostic fallback for non-tree estimators
                estimator = automl.model.estimator
                predict_fn = estimator.predict
                if self.task == 'classification' and hasattr(estimator, 'predict_proba'):
                    predict_fn = estimator.predict_proba
                explainer = shap.Explainer(predict_fn, X_neighbors)
                raw_shap_values = explainer(X_neighbors).values

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
                                                      average=None, labels=labels,
                                                      sample_weight=weights_neighbors, zero_division=np.nan)
                recall_per_class = recall_score(y_neighbors, y_pred_neighbors,
                                                average=None, labels=labels,
                                                sample_weight=weights_neighbors, zero_division=np.nan)
                f1_per_class = f1_score(y_neighbors, y_pred_neighbors,
                                        average=None, labels=labels,
                                        sample_weight=weights_neighbors, zero_division=np.nan)

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

    # ------------------------------------------------------------------
    # local_holdout evaluation
    # ------------------------------------------------------------------
    def _fit_local_holdout(self):
        """Fit one model per location on a per-neighborhood train/test split."""
        results = Parallel(n_jobs=self.n_jobs)(
            delayed(self._process_location_holdout)(i)
            for i in range(len(self.y))
        )
        valid = [r for r in results if r is not None]
        print(f"local_holdout: valid locations {len(valid)} / {self.X.shape[0]}")
        return LocalHoldoutResults(self, results)

    @staticmethod
    def _weighted_rmse(y_true, y_pred, w):
        return float(np.sqrt(np.average((y_true - y_pred) ** 2, weights=w)))

    def _process_location_holdout(self, i):
        """
        Split location i's neighborhood into train/test (focal forced into test),
        fit one AutoML model on the train neighbors, and return held-out local
        metrics plus the out-of-sample focal prediction.
        """
        try:
            weights_i = self._build_wi(i)
            neighbors_indices = np.where(weights_i > 0)[0]

            focal_arr = np.where(neighbors_indices == i)[0]
            if len(focal_arr) == 0:
                return None
            focal_pos = int(focal_arr[0])

            X_n = self.X[neighbors_indices]
            y_n = self.y[neighbors_indices]
            w_n = weights_i[neighbors_indices]
            n = len(neighbors_indices)

            non_focal = np.array([p for p in range(n) if p != focal_pos])
            if len(non_focal) < 3:
                return None

            n_test_total = max(2, int(round(self.test_size * n)))
            n_test_nonfocal = min(max(n_test_total - 1, 1), len(non_focal) - 2)
            if n_test_nonfocal < 1:
                return None

            stratify = y_n[non_focal].ravel() if self.task == 'classification' else None
            try:
                train_pos, test_nf_pos = train_test_split(
                    non_focal, test_size=n_test_nonfocal,
                    random_state=self.split_seed, stratify=stratify)
            except ValueError:
                try:
                    train_pos, test_nf_pos = train_test_split(
                        non_focal, test_size=n_test_nonfocal,
                        random_state=self.split_seed, stratify=None)
                except ValueError:
                    return None
            test_pos = np.append(test_nf_pos, focal_pos)

            if len(train_pos) < 2 or len(test_pos) < 2:
                return None

            X_tr, y_tr, w_tr = X_n[train_pos], y_n[train_pos], w_n[train_pos]
            X_te, y_te, w_te = X_n[test_pos], y_n[test_pos], w_n[test_pos]
            if np.sum(w_tr) <= 0 or np.sum(w_te) <= 0:
                return None
            if self.task == 'classification' and len(np.unique(y_tr)) < 2:
                return None

            automl = AutoML()
            automl.fit(X_tr, y_tr.ravel(), sample_weight=w_tr, **self.automl_settings)
            if getattr(automl, 'model', None) is None:
                return None

            yp_tr = np.asarray(automl.predict(X_tr)).ravel()
            yp_te = np.asarray(automl.predict(X_te)).ravel()
            focal_pred = yp_te[-1]

            rec = {
                'location_index': i,
                'focal_pred': focal_pred,
                'focal_true': np.asarray(self.y[i]).ravel()[0],
                'n_train': int(len(train_pos)),
                'n_test': int(len(test_pos)),
            }
            if self.task == 'classification':
                yt_tr, yt_te = y_tr.ravel(), y_te.ravel()
                rec['train_accuracy'] = float(np.sum(w_tr * (yt_tr == yp_tr)) / np.sum(w_tr))
                rec['test_accuracy'] = float(np.sum(w_te * (yt_te == yp_te)) / np.sum(w_te))
                rec['train_precision'], rec['train_recall'], rec['train_f1'] = self._macro_prf(yt_tr, yp_tr, w_tr)
                rec['test_precision'], rec['test_recall'], rec['test_f1'] = self._macro_prf(yt_te, yp_te, w_te)
            else:
                yt_tr, yt_te = y_tr.ravel().astype(float), y_te.ravel().astype(float)
                rec['train_r2'] = float(r2_score(yt_tr, yp_tr, sample_weight=w_tr, force_finite=False))
                rec['test_r2'] = float(r2_score(yt_te, yp_te, sample_weight=w_te, force_finite=False))
                rec['train_rmse'] = self._weighted_rmse(yt_tr, yp_tr, w_tr)
                rec['test_rmse'] = self._weighted_rmse(yt_te, yp_te, w_te)
            print(f"local_holdout: location {i}/{self.X.shape[0]} done")
            return rec

        except Exception as e:
            print(f"Error at location {i}: {str(e)}", file=sys.stderr)
            return None

    @staticmethod
    def _macro_prf(y_true, y_pred, w):
        labels = np.unique(np.concatenate([y_true, y_pred]))
        p = np.nanmean(precision_score(y_true, y_pred, labels=labels, average=None,
                                       sample_weight=w, zero_division=np.nan))
        r = np.nanmean(recall_score(y_true, y_pred, labels=labels, average=None,
                                    sample_weight=w, zero_division=np.nan))
        f = np.nanmean(f1_score(y_true, y_pred, labels=labels, average=None,
                                sample_weight=w, zero_division=np.nan))
        return float(p), float(r), float(f)
