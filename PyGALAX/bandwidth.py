"""
Bandwidth selection methods for GALAX.
"""

import os
import numpy as np
np.float = float
import pandas as pd
from joblib import Parallel, delayed
from sklearn.preprocessing import OneHotEncoder
from flaml import AutoML
import libpysal
from esda import Moran

from .kernel import Kernel


def check_class_sizes(weights, y_values, min_samples):
    """
    Check if each location has at least two classes and each present class 
    has at least min_samples samples within its bandwidth.
    
    Parameters
    ----------
    weights : array
        Weight matrix for current bandwidth
    y_values : array
        Target values
    min_samples : int
        Minimum required samples per class
        
    Returns
    -------
    bool
        True if class size requirements are met
    """
    n_locations = len(y_values)
    all_valid = True
    problem_locations = []

    for i in range(n_locations):
        # Get neighboring indices where weight > 0
        if isinstance(weights, libpysal.weights.KNN):
            neighbor_indices = weights.neighbors[i]
        else:
            neighbor_indices = np.where(weights[i] > 0)[0]

        if len(neighbor_indices) == 0:
            all_valid = False
            problem_locations.append({
                'location': i,
                'problems': ['No neighbors found within bandwidth'],
                'class_counts': {}
            })
            continue

        neighbor_classes = y_values[neighbor_indices]

        unique_local_classes, class_counts = np.unique(neighbor_classes, return_counts=True)
        class_count_dict = dict(zip(unique_local_classes, class_counts))

        location_valid = True
        problems = []

        # Check requirements:
        # 1. At least 2 classes present
        if len(unique_local_classes) < 2:
            location_valid = False
            problems.append(f"Only {len(unique_local_classes)} classes present")

        # 2. Each present class must have at least min_samples samples
        insufficient_classes = []
        for cls, count in class_count_dict.items():
            if count < min_samples:
                location_valid = False
                insufficient_classes.append((cls, count))

        if insufficient_classes:
            problems.append(f"Classes with insufficient samples: {insufficient_classes}")

        if not location_valid:
            all_valid = False
            problem_locations.append({
                'location': i,
                'problems': problems,
                'class_counts': class_count_dict
            })

    if not all_valid:
        print(f"\nFound {len(problem_locations)} locations with class size issues:")
        for loc in problem_locations[:5]:
            print(f"Location {loc['location']}:")
            print(f"  Problems: {', '.join(loc['problems'])}")
            print(f"  Class counts: {loc['class_counts']}")
        if len(problem_locations) > 5:
            print(f"... and {len(problem_locations) - 5} more locations with issues")

    return all_valid


def search_bw_lw_ISA(X, y, coords, bw_min=None, bw_max=None, step=1, kernel='bisquare', fixed=False, task='regression', min_samples_per_class=5):
    """
    Search for optimal bandwidth using Incremental Spatial Autocorrelation (ISA).
    
    Parameters
    ----------
    X : array
        Independent variables
    y : array
        Dependent variable
    coords : array
        Spatial coordinates
    bw_min : int, optional
        Minimum bandwidth to consider
    bw_max : int, optional
        Maximum bandwidth to consider
    step : int, optional
        Step size for bandwidth search
    kernel : str, optional
        Kernel function to use
    fixed : bool, optional
        Whether to use fixed or adaptive bandwidth
    task : str, optional
        Type of task: 'regression' or 'classification'
    min_samples_per_class : int, optional
        Minimum required samples per class
        
    Returns
    -------
    tuple
        (optimal bandwidth, Moran's I, p-value)
    """
    n_samples = X.shape[0]
    n_vars = X.shape[1]

    if bw_min is None:
        bw_min = max(round(n_samples * 0.05), n_vars + 2, 20)
    if bw_max is None:
        bw_max = max(round(n_samples * 0.95), n_vars + 2)

    print(f"\nStarting ISA bandwidth search:")
    print(f"- Minimum bandwidth: {bw_min}")
    print(f"- Maximum bandwidth: {bw_max}")
    print(f"- Step size: {step}")
    if task == 'classification':
        print(f"- Minimum samples per class: {min_samples_per_class}")
    print("-" * 50)

    coords_array = np.array(coords)
    kd = libpysal.cg.KDTree(coords_array)

    bandwidth_list = []
    moran_I_list = []
    z_score_list = []
    p_value_list = []

    if task == 'classification':
        print("Task: Classification")
        print(f"Number of samples: {len(y)}")
        print(f"Unique classes: {np.unique(y)}")
        y_reshaped = y.reshape(-1, 1)
        encoder = OneHotEncoder(sparse_output=False)
        y_onehot = encoder.fit_transform(y_reshaped)
        n_classes = y_onehot.shape[1]
        print(f"Number of classes: {n_classes}")
        print("-" * 50)

    total_bandwidths = (bw_max - bw_min) // step + 1
    accepted_bandwidths = 0

    for current_bw in range(bw_min, bw_max + 1, step):
        print(f"\nTesting bandwidth {current_bw} ({(current_bw - bw_min) // step + 1}/{total_bandwidths})...")
        if fixed:
            w = np.zeros((len(y), len(y)))
            for i in range(len(y)):
                kernel_obj = Kernel(coords_array[i], coords_array, current_bw, fixed=True, function=kernel)
                w[i] = kernel_obj.kernel
        else:
            w = libpysal.weights.KNN(kd, current_bw)

        if task == 'classification':
            has_enough_samples = check_class_sizes(w.full()[0] if not fixed else w, y, min_samples_per_class)
            if not has_enough_samples:
                print(f"✗ Bandwidth {current_bw} rejected: insufficient samples per class")
                continue

            accepted_bandwidths += 1
            print(f"✓ Bandwidth {current_bw} accepted: class size requirements met")

            morans = []
            zscores = []
            pvalues = []
            for class_idx in range(n_classes):
                class_values = y_onehot[:, class_idx]
                moran = Moran(class_values, w)
                morans.append(moran.I)
                zscores.append(moran.z_norm)
                pvalues.append(moran.p_norm)
            moran_I = np.mean(morans)
            z_score = np.mean(zscores)
            p_value = np.mean(pvalues)
        else:
            moran = Moran(y, w)
            moran_I = moran.I
            z_score = moran.z_norm
            p_value = moran.p_norm
            pass

        bandwidth_list.append(current_bw)
        moran_I_list.append(moran_I)
        z_score_list.append(z_score)
        p_value_list.append(p_value)

    print(f"\nBandwidth search completed:")
    print(f"- Tested {total_bandwidths} bandwidths")
    if task == 'classification':
        print(f"- {accepted_bandwidths} bandwidths accepted")
        print(f"- {total_bandwidths - accepted_bandwidths} bandwidths rejected")

    if not bandwidth_list:
        raise ValueError("No bandwidth found that satisfies the minimum class size requirements")

    # Find optimal bandwidth (highest z-score with p < 0.05)
    significant_indices = [i for i, p in enumerate(p_value_list) if p < 0.05]
    if not significant_indices:
        raise ValueError("No significant bandwidth found")

    max_index = max(significant_indices, key=lambda i: z_score_list[i])
    found_bandwidth = bandwidth_list[max_index]
    found_moran_I = moran_I_list[max_index]
    found_p_value = p_value_list[max_index]

    print(f"\nOptimal bandwidth found: {found_bandwidth}")
    print(f"Moran's I: {found_moran_I:.4f}")
    print(f"p-value: {found_p_value:.4f}")

    return found_bandwidth, found_moran_I, found_p_value


def search_bandwidth(X, y, coords, automl_settings, bw_min=None, bw_max=None, step=1, kernel='bisquare', fixed=False, n_jobs=None, task='regression', min_samples_per_class=5):
    """
    Optimize bandwidth using AutoML performance.
    
    Parameters
    ----------
    X : array
        Independent variables
    y : array
        Dependent variable
    coords : array
        Spatial coordinates
    automl_settings : dict
        AutoML configuration parameters
    bw_min : int, optional
        Minimum bandwidth to consider
    bw_max : int, optional
        Maximum bandwidth to consider
    step : int, optional
        Step size for bandwidth search
    kernel : str, optional
        Kernel function to use
    fixed : bool, optional
        Whether to use fixed or adaptive bandwidth
    n_jobs : int, optional
        Number of parallel jobs
    task : str, optional
        Type of task: 'regression' or 'classification'
    min_samples_per_class : int, optional
        Minimum required samples per class
        
    Returns
    -------
    dict
        Search results including optimal bandwidth
    """
    total_cpus = os.cpu_count()
    n_jobs = n_jobs if n_jobs is not None else int(total_cpus * 0.5)

    n_samples = X.shape[0]
    n_vars = X.shape[1]

    if bw_min is None:
        bw_min = max(round(n_samples * 0.05), n_vars + 2, 20)
    if bw_max is None:
        bw_max = max(round(n_samples * 0.95), n_vars + 2)

    def evaluate_bandwidth(bw):
        local_scores = []

        if fixed:
            w = np.zeros((n_samples, n_samples))
            for i in range(n_samples):
                kernel_obj = Kernel(coords[i], coords, bw, fixed=True, function=kernel)
                w[i] = kernel_obj.kernel
        else:
            kd = libpysal.cg.KDTree(coords)
            w = libpysal.weights.KNN(kd, bw)
            w = w.full()[0]

        # For classification, check class sizes first
        if task == 'classification':
            print(f"\nChecking bandwidth {bw}...")
            has_enough_samples = check_class_sizes(w, y, min_samples_per_class)
            if not has_enough_samples:
                print(f"Bandwidth {bw} rejected: insufficient samples per class")
                return float('-inf')

        for i in range(n_samples):
            weights = w[i]
            mask = weights > 0
            X_local = X[mask]
            y_local = y[mask]
            weights_local = weights[mask]

            if len(X_local) < n_vars + 2:
                continue

            try:
                automl = AutoML()
                automl.fit(X_local, y_local.ravel(), sample_weight=weights_local, **automl_settings)

                if task == 'classification':
                    y_pred = automl.predict(X_local)
                    correct_predictions = (y_local.ravel() == y_pred)
                    weighted_score = np.sum(weights_local * correct_predictions) / np.sum(weights_local)
                else:
                    weighted_score = automl.score(X_local, y_local.ravel())
                local_scores.append(weighted_score)
            except Exception as e:
                print(f"Error at location {i}: {str(e)}")
                continue

        return np.mean(local_scores) if local_scores else float('-inf')

    # Parallel bandwidth evaluation
    bandwidth_scores = Parallel(n_jobs=n_jobs)(
        delayed(evaluate_bandwidth)(bw)
        for bw in range(bw_min, bw_max + 1, step)
    )

    df_scores = pd.DataFrame({
        'bandwidth': range(bw_min, bw_max + 1, step),
        'score': bandwidth_scores
    })

    df_scores = df_scores[df_scores['score'] > float('-inf')]
    if df_scores.empty:
        raise ValueError("No bandwidth found that satisfies the minimum class size requirements")

    optimal_bw = df_scores.loc[df_scores['score'].idxmax(), 'bandwidth']

    return {
        'bandwidth_search_result': df_scores,
        'best_bandwidth': optimal_bw,
        'metric': 'accuracy' if task == 'classification' else 'r2'
    }
