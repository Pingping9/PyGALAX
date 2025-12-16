"""
Results handling for GALAX models.
"""

import numpy as np
np.float = float
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, r2_score, mean_squared_error
from joblib import dump
from scipy import stats
import sys

from .kernel import Kernel


class GALAXResults:
    """
    Results class for GALAX models.
    
    This class handles the processing, analysis, and storage of GALAX model results,
    including local and global performance metrics, SHAP interpretations, and 
    detailed location-specific outputs.
    
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
        Model predictions for each location
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

    def get_detailed_shap_for_location(self, location_idx):
        """
        Get detailed SHAP analysis for a specific location.
        
        Parameters
        ----------
        location_idx : int
            Original index of the location to analyze
            
        Returns
        -------
        pd.DataFrame or None
            Detailed SHAP analysis or None if location not found
        """
        try:
            result_internal_idx = self.location_original_indices.index(location_idx)
            loc_result = self.results[result_internal_idx]
        except ValueError:
            print(f"Location {location_idx} was not successfully processed.", file=sys.stderr)
            return None

        raw_shap_values = self.raw_shap_values_neighbors[result_internal_idx]
        X_neighbors = self.X_neighbors_values[result_internal_idx]

        feature_names = self.model.x_vars if self.model.x_vars else [f'Feature_{j}' for j in range(X_neighbors.shape[1])]

        kernel_obj = Kernel(self.model.coords[location_idx], self.model.coords, self.model.bw,
                            fixed=self.model.fixed, function=self.model.kernel)
        weights_i_full = kernel_obj.kernel
        neighbor_original_indices = np.where(weights_i_full > 0)[0]

        detailed_data = []

        if self.model.task == 'classification':
            class_labels = loc_result.get('classes_present', None)
            if isinstance(raw_shap_values, list):
                num_classes = len(raw_shap_values)
                if class_labels is None or len(class_labels) != num_classes:
                    class_labels = list(range(num_classes))
                for class_idx in range(num_classes):
                    class_shap_array = raw_shap_values[class_idx]
                    total_abs_shap_per_neighbor_for_class = np.sum(np.abs(class_shap_array), axis=1)
                    for neighbor_local_idx in range(X_neighbors.shape[0]):
                        original_neighbor_idx = neighbor_original_indices[neighbor_local_idx]
                        for feature_idx in range(X_neighbors.shape[1]):
                            shap_val = class_shap_array[neighbor_local_idx, feature_idx]
                            orig_val = X_neighbors[neighbor_local_idx, feature_idx]
                            shap_pct = (np.abs(shap_val) / total_abs_shap_per_neighbor_for_class[neighbor_local_idx]) * 100 \
                                if total_abs_shap_per_neighbor_for_class[neighbor_local_idx] != 0 else 0
                            detailed_data.append({
                                'Central_Location_Index': location_idx,
                                'Neighbor_Original_Index': original_neighbor_idx,
                                'Feature': feature_names[feature_idx],
                                'Target_Class': class_labels[class_idx],
                                'SHAP_Value': shap_val,
                                'SHAP_Direction': 'Positive' if shap_val > 0 else ('Negative' if shap_val < 0 else 'Zero'),
                                'SHAP_Percentage_for_Class': shap_pct,
                                'Original_Feature_Value': orig_val
                            })
            elif raw_shap_values.ndim == 3:
                num_classes = raw_shap_values.shape[2]
                if class_labels is None or len(class_labels) != num_classes:
                    class_labels = list(range(num_classes))
                for class_idx in range(num_classes):
                    class_shap_array = raw_shap_values[:, :, class_idx]
                    total_abs_shap_per_neighbor_for_class = np.sum(np.abs(class_shap_array), axis=1)
                    for neighbor_local_idx in range(X_neighbors.shape[0]):
                        original_neighbor_idx = neighbor_original_indices[neighbor_local_idx]
                        for feature_idx in range(X_neighbors.shape[1]):
                            shap_val = class_shap_array[neighbor_local_idx, feature_idx]
                            orig_val = X_neighbors[neighbor_local_idx, feature_idx]
                            shap_pct = (np.abs(shap_val) / total_abs_shap_per_neighbor_for_class[neighbor_local_idx]) * 100 \
                                if total_abs_shap_per_neighbor_for_class[neighbor_local_idx] != 0 else 0
                            detailed_data.append({
                                'Central_Location_Index': location_idx,
                                'Neighbor_Original_Index': original_neighbor_idx,
                                'Feature': feature_names[feature_idx],
                                'Target_Class': class_labels[class_idx],
                                'SHAP_Value': shap_val,
                                'SHAP_Direction': 'Positive' if shap_val > 0 else ('Negative' if shap_val < 0 else 'Zero'),
                                'SHAP_Percentage_for_Class': shap_pct,
                                'Original_Feature_Value': orig_val
                            })
            else:
                print(f"Warning: Unexpected 2D SHAP output for classification task at location {location_idx}. Treating as single overall influence.", file=sys.stderr)
                total_abs_shap_per_neighbor = np.sum(np.abs(raw_shap_values), axis=1)
                for neighbor_local_idx in range(X_neighbors.shape[0]):
                    original_neighbor_idx = neighbor_original_indices[neighbor_local_idx]
                    for feature_idx in range(X_neighbors.shape[1]):
                        shap_val = raw_shap_values[neighbor_local_idx, feature_idx]
                        orig_val = X_neighbors[neighbor_local_idx, feature_idx]
                        shap_pct = (np.abs(shap_val) / total_abs_shap_per_neighbor[neighbor_local_idx]) * 100 \
                            if total_abs_shap_per_neighbor[neighbor_local_idx] != 0 else 0
                        detailed_data.append({
                            'Central_Location_Index': location_idx,
                            'Neighbor_Original_Index': original_neighbor_idx,
                            'Feature': feature_names[feature_idx],
                            'Target_Class': 'Overall_Influence',
                            'SHAP_Value': shap_val,
                            'SHAP_Direction': 'Positive' if shap_val > 0 else ('Negative' if shap_val < 0 else 'Zero'),
                            'SHAP_Percentage_for_Class': shap_pct,
                            'Original_Feature_Value': orig_val
                        })
        else:
            total_abs_shap_per_neighbor = np.sum(np.abs(raw_shap_values), axis=1)
            for neighbor_local_idx in range(X_neighbors.shape[0]):
                original_neighbor_idx = neighbor_original_indices[neighbor_local_idx]
                for feature_idx in range(X_neighbors.shape[1]):
                    shap_val = raw_shap_values[neighbor_local_idx, feature_idx]
                    orig_val = X_neighbors[neighbor_local_idx, feature_idx]
                    shap_pct = (np.abs(shap_val) / total_abs_shap_per_neighbor[neighbor_local_idx]) * 100 \
                        if total_abs_shap_per_neighbor[neighbor_local_idx] != 0 else 0
                    detailed_data.append({
                        'Central_Location_Index': location_idx,
                        'Neighbor_Original_Index': original_neighbor_idx,
                        'Feature': feature_names[feature_idx],
                        'SHAP_Value': shap_val,
                        'SHAP_Direction': 'Positive' if shap_val > 0 else ('Negative' if shap_val < 0 else 'Zero'),
                        'SHAP_Percentage': shap_pct,
                        'Original_Feature_Value': orig_val
                    })

        return pd.DataFrame(detailed_data)

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
