#!/usr/bin/env python3
"""
Read & Write Epochs for Decision Trees (RWDT)

A novel approach that iteratively refines training data representations
by computing corrections toward correct classification paths in decision trees.

This script demonstrates RWDT on datasets from OpenML.

Usage:
    python rwdt.py
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import openml
import warnings

warnings.filterwarnings('ignore')


# =============================================================================
# CONFIGURATION
# =============================================================================

LEARNING_RATES = [0.1, 0.2, 0.3, 0.5, 1.0]
MAX_DEPTHS = [3, 5, 7, 10]
N_ITERATIONS = [1, 2, 3, 5, 10, 15, 20]

N_FOLDS = 5
N_SEEDS = 5
MAX_SAMPLES = 5000

DATASET_NAMES = ["mfeat-karhunen", "mfeat-factors"]


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def encode_categorical_features(X):
    """Encode categorical features to numeric."""
    if isinstance(X, pd.DataFrame):
        X_df = X.copy()
    else:
        X_df = pd.DataFrame(X)

    for col in X_df.columns:
        if X_df[col].dtype.name == 'category':
            X_df[col] = X_df[col].astype(str)
        if X_df[col].dtype == 'object' or not np.issubdtype(X_df[col].dtype, np.number):
            le = LabelEncoder()
            X_df[col] = X_df[col].fillna('__MISSING__').astype(str)
            X_df[col] = le.fit_transform(X_df[col])

    return X_df.values.astype(np.float64)


def compute_correction(tree, x, y_true):
    """
    Compute feature correction toward correct classification.
    
    For a misclassified sample, find the minimum-cost path to a leaf
    that predicts the correct class, and compute the feature adjustments
    needed to follow that path.
    
    Parameters
    ----------
    tree : DecisionTreeClassifier
        Fitted decision tree
    x : array-like
        Single sample features
    y_true : int
        True class label
        
    Returns
    -------
    correction : ndarray
        Feature adjustments to move toward correct classification
    """
    t = tree.tree_
    n_features = len(x)

    # Already correct - no correction needed
    if tree.predict([x])[0] == y_true:
        return np.zeros(n_features)

    y_idx = np.where(tree.classes_ == y_true)[0][0]

    def find_correct_leaves(node=0):
        """Find all leaves that predict the correct class."""
        if t.feature[node] == -2:  # Leaf node
            if np.argmax(t.value[node][0]) == y_idx:
                return [node]
            return []
        return (find_correct_leaves(t.children_left[node]) +
                find_correct_leaves(t.children_right[node]))

    correct_leaves = find_correct_leaves()
    if not correct_leaves:
        return np.zeros(n_features)

    def path_to_leaf(target, node=0, path=None):
        """Find the path from root to a target leaf."""
        if path is None:
            path = []
        if node == target:
            return path
        if t.feature[node] == -2:
            return None
        left = path_to_leaf(target, t.children_left[node], path + [(node, 'L')])
        return left if left else path_to_leaf(target, t.children_right[node], path + [(node, 'R')])

    best_cost, best_correction = float('inf'), np.zeros(n_features)

    for leaf in correct_leaves:
        path = path_to_leaf(leaf)
        if not path:
            continue

        correction, cost = np.zeros(n_features), 0
        for node, direction in path:
            feat = t.feature[node]
            thresh = t.threshold[node]
            val = x[feat]

            if direction == 'L' and val > thresh:
                delta = thresh - val - 0.001
                correction[feat] += delta
                cost += abs(delta)
            elif direction == 'R' and val <= thresh:
                delta = thresh - val + 0.001
                correction[feat] += delta
                cost += abs(delta)

        if cost < best_cost:
            best_cost, best_correction = cost, correction

    return best_correction


# =============================================================================
# RWDT MODEL
# =============================================================================

class RWDT:
    """
    Read & Write Epochs for Decision Trees.
    
    RWDT iteratively refines the training data representation by:
    1. Training a decision tree on the current data (read phase)
    2. For misclassified samples, computing corrections toward correct leaves (write phase)
    3. Adjusting sample features using these corrections
    4. Repeating for multiple epochs
    
    This "read & write epochs" process allows the model to learn better representations
    while maintaining the interpretability of decision trees.
    
    Parameters
    ----------
    max_depth : int, default=5
        Maximum depth of the decision tree
    n_iterations : int, default=3
        Number of read-write epochs
    learning_rate : float, default=0.2
        Step size for feature corrections
        
    Attributes
    ----------
    tree : DecisionTreeClassifier
        The final fitted decision tree
    X_corrected : ndarray
        The transformed training data after all epochs
    """

    def __init__(self, max_depth=5, n_iterations=3, learning_rate=0.2):
        self.max_depth = max_depth
        self.n_iterations = n_iterations
        self.learning_rate = learning_rate
        self.tree = None
        self.X_corrected = None

    def fit(self, X, y, random_state=42):
        """
        Fit the RWDT model.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data
        y : array-like of shape (n_samples,)
            Target values
        random_state : int, default=42
            Random state for reproducibility
            
        Returns
        -------
        self : RWDT
            Fitted estimator
        """
        X_mod = X.copy()

        for it in range(self.n_iterations):
            tree = DecisionTreeClassifier(max_depth=self.max_depth, random_state=random_state)
            tree.fit(X_mod, y)

            preds = tree.predict(X_mod)
            wrong_idx = np.where(preds != y)[0]

            if len(wrong_idx) == 0:
                break

            for i in wrong_idx:
                correction = compute_correction(tree, X_mod[i], y[i])
                if np.linalg.norm(correction) > 1e-10:
                    X_mod[i] += self.learning_rate * correction

        self.tree = DecisionTreeClassifier(max_depth=self.max_depth, random_state=random_state)
        self.tree.fit(X_mod, y)
        self.X_corrected = X_mod

        return self

    def predict(self, X):
        """
        Predict class labels for samples in X.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to predict
            
        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted class labels
        """
        return self.tree.predict(X)


# =============================================================================
# EXPERIMENT
# =============================================================================

def run_experiment(X_train, y_train, X_test, y_test, seed, max_depth, lr, n_iter):
    """Run a single experiment comparing DT, RF, and RWDT."""
    results = {}

    # Baseline Decision Tree
    dt = DecisionTreeClassifier(max_depth=max_depth, random_state=seed)
    dt.fit(X_train, y_train)
    results['DT_acc'] = accuracy_score(y_test, dt.predict(X_test))

    # Random Forest baseline
    rf = RandomForestClassifier(n_estimators=100, max_depth=max_depth, random_state=seed, n_jobs=1)
    rf.fit(X_train, y_train)
    results['RF_acc'] = accuracy_score(y_test, rf.predict(X_test))

    # RWDT
    rwdt = RWDT(max_depth=max_depth, n_iterations=n_iter, learning_rate=lr)
    rwdt.fit(X_train.copy(), y_train.copy(), random_state=seed)
    results['RWDT_acc'] = accuracy_score(y_test, rwdt.predict(X_test))

    return results


def load_dataset(name):
    """Load dataset from OpenML by name."""
    print(f"Loading dataset: {name}")
    
    datasets = openml.datasets.list_datasets(output_format='dataframe')
    matches = datasets[datasets['name'] == name]
    
    if len(matches) == 0:
        raise ValueError(f"Dataset '{name}' not found on OpenML")
    
    dataset_id = int(matches.iloc[0]['did'])  # Convert to Python int
    dataset = openml.datasets.get_dataset(dataset_id)
    
    X, y, _, _ = dataset.get_data(target=dataset.default_target_attribute)
    
    return X, y, dataset.name


def run_on_dataset(dataset_name):
    """Run full experiment on a single dataset."""
    print()
    print("-" * 70)
    
    # Load dataset
    X, y, name = load_dataset(dataset_name)
    
    # Preprocess
    if isinstance(X, pd.DataFrame):
        for col in X.columns:
            if X[col].dtype.name == 'category':
                X[col] = X[col].astype(str)
            elif X[col].dtype == 'object':
                X[col] = X[col].fillna('__MISSING__').astype(str)

    X = encode_categorical_features(X)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    
    le = LabelEncoder()
    y = le.fit_transform(np.array(y).astype(str))

    n_samples, n_features = X.shape
    n_classes = len(np.unique(y))

    print(f"Dataset: {name}")
    print(f"  Samples: {n_samples}")
    print(f"  Features: {n_features}")
    print(f"  Classes: {n_classes}")
    print()

    # Subsample if needed
    if n_samples > MAX_SAMPLES:
        np.random.seed(42)
        idx = np.random.choice(n_samples, MAX_SAMPLES, replace=False)
        X, y = X[idx], y[idx]
        n_samples = MAX_SAMPLES
        print(f"  Subsampled to {MAX_SAMPLES} samples")

    # Run experiments
    print("Running experiments...")
    print(f"  {N_FOLDS}-fold CV × {N_SEEDS} seeds")
    print(f"  Hyperparameters: {len(MAX_DEPTHS)} depths × {len(LEARNING_RATES)} LRs × {len(N_ITERATIONS)} iterations")
    print()

    all_results = []

    for seed in range(N_SEEDS):
        cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)

        for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X, y)):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

            for max_depth in MAX_DEPTHS:
                for lr in LEARNING_RATES:
                    for n_iter in N_ITERATIONS:
                        res = run_experiment(
                            X_train.copy(), y_train.copy(),
                            X_test.copy(), y_test.copy(),
                            seed, max_depth, lr, n_iter
                        )

                        all_results.append({
                            'seed': seed,
                            'fold': fold_idx,
                            'max_depth': max_depth,
                            'lr': lr,
                            'n_iter': n_iter,
                            'DT_acc': res['DT_acc'],
                            'RF_acc': res['RF_acc'],
                            'RWDT_acc': res['RWDT_acc'],
                        })

        print(f"  Completed seed {seed + 1}/{N_SEEDS}")

    # Aggregate results
    results_df = pd.DataFrame(all_results)

    aggregated = []
    for max_depth in MAX_DEPTHS:
        for lr in LEARNING_RATES:
            for n_iter in N_ITERATIONS:
                subset = results_df[
                    (results_df['max_depth'] == max_depth) &
                    (results_df['lr'] == lr) &
                    (results_df['n_iter'] == n_iter)
                ]

                agg_row = {
                    'max_depth': max_depth,
                    'lr': lr,
                    'n_iter': n_iter,
                    'DT_acc': subset['DT_acc'].mean(),
                    'DT_std': subset['DT_acc'].std(),
                    'RF_acc': subset['RF_acc'].mean(),
                    'RF_std': subset['RF_acc'].std(),
                    'RWDT_acc': subset['RWDT_acc'].mean(),
                    'RWDT_std': subset['RWDT_acc'].std(),
                }
                agg_row['RWDT_vs_DT'] = (agg_row['RWDT_acc'] - agg_row['DT_acc']) * 100
                agg_row['RF_vs_DT'] = (agg_row['RF_acc'] - agg_row['DT_acc']) * 100
                aggregated.append(agg_row)

    # Find best configuration
    best = max(aggregated, key=lambda x: x['RWDT_vs_DT'])

    # Save results
    output_df = pd.DataFrame(aggregated)
    output_df.insert(0, 'dataset', name)
    
    return output_df, best, name


def main():
    print("=" * 70)
    print("Read & Write Epochs for Decision Trees (RWDT)")
    print("=" * 70)
    
    all_dfs = []
    
    for dataset_name in DATASET_NAMES:
        try:
            output_df, best, name = run_on_dataset(dataset_name)
            all_dfs.append(output_df)
            
            print()
            print(f"Best RWDT configuration for {name}:")
            print(f"  max_depth={best['max_depth']}, lr={best['lr']}, n_iter={best['n_iter']}")
            print()
            print(f"Performance (mean ± std):")
            print(f"  Decision Tree: {best['DT_acc']*100:.2f}% ± {best['DT_std']*100:.2f}%")
            print(f"  Random Forest: {best['RF_acc']*100:.2f}% ± {best['RF_std']*100:.2f}%")
            print(f"  RWDT:          {best['RWDT_acc']*100:.2f}% ± {best['RWDT_std']*100:.2f}%")
            print()
            print(f"Improvement over Decision Tree:")
            print(f"  RWDT: {best['RWDT_vs_DT']:+.2f}%")
            print(f"  RF:   {best['RF_vs_DT']:+.2f}%")
            
        except Exception as e:
            print(f"Failed to process {dataset_name}: {e}")
    
    if all_dfs:
        combined_df = pd.concat(all_dfs, ignore_index=True)
        combined_df.to_csv('rwdt_results.csv', index=False)
        print()
        print("=" * 70)
        print(f"All results saved to rwdt_results.csv")
        print("=" * 70)


if __name__ == "__main__":
    main()