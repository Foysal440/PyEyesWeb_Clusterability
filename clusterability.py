"""Clusterability analysis module for Hopkins statistic computation.

This module provides tools for computing the Hopkins statistic to assess
the clusterability of multivariate data. The Hopkins statistic measures
the spatial randomness of data points and indicates whether meaningful
clustering structure exists.

The Hopkins statistic computation follows these steps:
1. Sample points from the actual data distribution
2. Generate uniform random points within the data space
3. Compute distances to nearest neighbors for both sample types
4. Calculate the Hopkins statistic from distance ratios

The Hopkins statistic ranges from 0 to 1:
- Values close to 0.5 indicate random data distribution
- Values significantly above 0.5 indicate clusterable data
- Values significantly below 0.5 indicate uniformly distributed data

Typical use cases include:
1. Pre-clustering analysis to determine if clustering is appropriate
2. Quality assessment of feature spaces in machine learning
3. Motion pattern analysis in movement data
4. Spatial distribution analysis in sensor data

References
----------
1. Hopkins, B. & Skellam, J. G. (1954). A new method for determining
   the type of distribution of plant individuals. Annals of Botany.
2. Lawson, R. G. & Jurs, P. C. (1990). New index for clustering tendency
   detection. Journal of Chemical Information and Computer Sciences.
"""

import numpy as np
from sklearn.neighbors import NearestNeighbors

from pyeyesweb.data_models.sliding_window import SlidingWindow
from pyeyesweb.data_models.thread_safe_buffer import ThreadSafeHistoryBuffer
from pyeyesweb.utils.validators import validate_integer, validate_boolean, validate_numeric


class Clusterability:
    """Real time clusterability analyzer using Hopkins statistic.

    This class computes the Hopkins statistic to assess whether data exhibits
    meaningful clustering structure. It maintains a history buffer for tracking
    clusterability over time and provides interpretative thresholds.

    The Hopkins statistic interpretation:
    - > 0.75: Strong clustering tendency
    - 0.6 - 0.75: Moderate clustering tendency
    - 0.5 - 0.6: Weak clustering tendency
    - 0.3 - 0.5: Mostly random distribution
    - < 0.3: Uniform distribution

    Parameters
    ----------
    sensitivity : int, optional
        Size of the Hopkins statistic history buffer. Larger values provide
        more temporal context but increase memory usage. Must be positive
        integer between 1 and 10,000 (default: 100).
    output_interpretation : bool, optional
        If True, outputs clusterability interpretation as categorical labels
        (e.g., "STRONG CLUSTERING", "RANDOM"). Must be boolean (default: True).
    sample_fraction : float, optional
        Fraction of data points to use for Hopkins computation. Must be between
        0.01 and 0.5 (default: 0.1).
    random_state : int or None, optional
        Random seed for reproducible sampling. If None, uses random sampling
        (default: None).
    """

    def __init__(self, sensitivity=100, output_interpretation=True,
                 sample_fraction=0.1, random_state=None):
        sensitivity = validate_integer(sensitivity, 'sensitivity', min_val=1, max_val=10000)
        self.output_interpretation = validate_boolean(output_interpretation, 'output_interpretation')
        self.sample_fraction = validate_numeric(sample_fraction, 'sample_fraction', min_val=0.01, max_val=0.5)

        if random_state is not None:
            if not isinstance(random_state, int):
                raise TypeError("random_state must be integer or None")
            if random_state < 0:
                raise ValueError("random_state must be non-negative")
        self.random_state = random_state

        self.hopkins_history = ThreadSafeHistoryBuffer(maxlen=sensitivity)

    def compute_hopkins_statistic(self, data: np.ndarray) -> float:
        """Compute Hopkins statistic for clusterability assessment.

        Parameters
        ----------
        data : np.ndarray
            Multivariate data array of shape (n_samples, n_features).
            Must contain at least 10 samples.

        Returns
        -------
        float
            Hopkins statistic value between 0 and 1.
        """
        if data.ndim != 2:
            raise ValueError(f"Data must be 2D array, got {data.ndim}D")

        n_samples, n_features = data.shape

        if n_samples < 10:
            raise ValueError(f"Need at least 10 samples, got {n_samples}")

        if n_features < 1:
            raise ValueError("Data must have at least 1 feature")

        sample_size = max(2, int(self.sample_fraction * n_samples))
        sample_size = min(sample_size, n_samples // 2)

        if self.random_state is not None:
            np.random.seed(self.random_state)

        data_indices = np.random.choice(n_samples, size=sample_size, replace=False)
        data_sample = data[data_indices]

        mins = np.min(data, axis=0)
        maxs = np.max(data, axis=0)

        ranges = maxs - mins
        if np.any(ranges <= 0):
            return 0.5

        uniform_sample = np.random.uniform(mins, maxs, size=(sample_size, n_features))

        nbrs = NearestNeighbors(n_neighbors=2, algorithm='auto').fit(data)

        data_distances, _ = nbrs.kneighbors(data_sample)
        u_distances = data_distances[:, 1]

        uniform_distances, _ = nbrs.kneighbors(uniform_sample)
        w_distances = uniform_distances[:, 0]

        numerator = np.sum(w_distances)
        denominator = np.sum(u_distances) + np.sum(w_distances)

        if denominator == 0:
            return 0.5

        hopkins_stat = numerator / denominator
        return float(hopkins_stat)

    def interpret_hopkins_statistic(self, hopkins_stat: float) -> str:
        """Interpret Hopkins statistic value with categorical labels.

        Parameters
        ----------
        hopkins_stat : float
            Hopkins statistic value between 0 and 1.

        Returns
        -------
        str
            Interpretation category.
        """
        if hopkins_stat > 0.75:
            return "STRONG CLUSTERING"
        elif hopkins_stat > 0.6:
            return "MODERATE CLUSTERING"
        elif hopkins_stat > 0.5:
            return "WEAK CLUSTERING"
        elif hopkins_stat > 0.3:
            return "RANDOM DISTRIBUTION"
        else:
            return "UNIFORM DISTRIBUTION"

    def compute_clusterability(self, signals: SlidingWindow) -> dict:
        """Compute clusterability analysis for multivariate signals.

        Parameters
        ----------
        signals : SlidingWindow
            Sliding window buffer containing multivariate signal data.

        Returns
        -------
        dict
            Dictionary containing clusterability metrics.
        """
        if not signals.is_full():
            return {
                "hopkins_statistic": np.nan,
                "interpretation": None,
                "sample_size": 0,
                "feature_dimension": signals._n_columns
            }

        data, _ = signals.to_array()
        n_samples, n_features = data.shape

        if n_samples < 10:
            return {
                "hopkins_statistic": np.nan,
                "interpretation": None,
                "sample_size": n_samples,
                "feature_dimension": n_features
            }

        try:
            hopkins_stat = self.compute_hopkins_statistic(data)
            self.hopkins_history.append(hopkins_stat)

            interpretation = None
            if self.output_interpretation:
                interpretation = self.interpret_hopkins_statistic(hopkins_stat)

            return {
                "hopkins_statistic": hopkins_stat,
                "interpretation": interpretation,
                "sample_size": n_samples,
                "feature_dimension": n_features
            }

        except Exception as e:
            return {
                "hopkins_statistic": 0.5,
                "interpretation": "COMPUTATION ERROR" if self.output_interpretation else None,
                "sample_size": n_samples,
                "feature_dimension": n_features
            }

    def get_temporal_statistics(self) -> dict:
        """Get temporal statistics from Hopkins history.

        Returns
        -------
        dict
            Dictionary containing temporal analysis.
        """
        history = self.hopkins_history.get_all()

        if len(history) < 2:
            return {
                "mean": np.nan,
                "std": np.nan,
                "trend": np.nan,
                "stability": np.nan,
                "history_length": len(history)
            }

        hopkins_array = np.array(history)
        mean_val = np.mean(hopkins_array)
        std_val = np.std(hopkins_array)

        x = np.arange(len(history))
        trend = np.polyfit(x, hopkins_array, 1)[0]

        stability = std_val / mean_val if mean_val > 0 else np.inf

        return {
            "mean": float(mean_val),
            "std": float(std_val),
            "trend": float(trend),
            "stability": float(stability),
            "history_length": len(history)
        }

    def __call__(self, sliding_window: SlidingWindow) -> dict:
        """Compute and optionally display clusterability metrics.

        Parameters
        ----------
        sliding_window : SlidingWindow
            Buffer containing multivariate data to analyze.

        Returns
        -------
        dict
            Dictionary containing clusterability metrics.
        """
        result = self.compute_clusterability(sliding_window)
        hopkins_stat = result["hopkins_statistic"]
        interpretation = result["interpretation"]

        if not np.isnan(hopkins_stat):
            if self.output_interpretation:
                print(f"Hopkins Statistic: {hopkins_stat:.3f}, Interpretation: {interpretation}")
            else:
                print(f"Hopkins Statistic: {hopkins_stat:.3f}")

        return result

    def reset_history(self):
        """Clear the Hopkins statistic history buffer."""
        self.hopkins_history.clear()

    def get_history(self) -> np.ndarray:
        """Get the complete history of Hopkins statistics.

        Returns
        -------
        np.ndarray
            Array of all stored Hopkins statistic values.
        """
        return np.array(self.hopkins_history.get_all())


def assess_clusterability(data: np.ndarray, sample_fraction: float = 0.2,
                         random_state: int = None) -> dict:
    """One-time clusterability assessment for static datasets.

    Parameters
    ----------
    data : np.ndarray
        Multivariate data array of shape (n_samples, n_features).
    sample_fraction : float, optional
        Fraction of data points to use for Hopkins computation.
    random_state : int or None, optional
        Random seed for reproducible sampling.

    Returns
    -------
    dict
        Clusterability assessment results.
    """
    analyzer = Clusterability(
        sensitivity=1,
        output_interpretation=True,
        sample_fraction=sample_fraction,
        random_state=random_state
    )

    class MockSlidingWindow:
        def __init__(self, data):
            self.data = data
            self._n_columns = data.shape[1]

        def is_full(self):
            return True

        def to_array(self):
            return self.data, None

    window = MockSlidingWindow(data)
    return analyzer.compute_clusterability(window)