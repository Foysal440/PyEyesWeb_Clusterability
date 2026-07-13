"""Pytest tests for Clusterability module."""

import numpy as np
import pytest
import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Mock dependencies
class SlidingWindow:
    def __init__(self, window_size: int, n_columns: int):
        self.window_size = window_size
        self.n_columns = n_columns
        self._n_columns = n_columns
        self.buffer = []

    def add(self, data_point):
        if len(data_point) != self.n_columns:
            raise ValueError(f"Expected {self.n_columns} columns, got {len(data_point)}")
        self.buffer.append(data_point)
        if len(self.buffer) > self.window_size:
            self.buffer.pop(0)

    def is_full(self):
        return len(self.buffer) >= self.window_size

    def to_array(self):
        return np.array(self.buffer), None


# Import the module to test
from clusterability import Clusterability


class TestClusterability:
    """Test cases for Clusterability class."""

    def test_initialization(self):
        """Test Clusterability initialization."""
        analyzer = Clusterability(n_neighbors=5)
        assert analyzer.n_neighbors == 5
        assert analyzer.random_state is None

        analyzer = Clusterability(n_neighbors=10, random_state=42)
        assert analyzer.n_neighbors == 10
        assert analyzer.random_state == 42

    def test_invalid_parameters(self):
        """Test invalid parameter handling."""
        with pytest.raises(TypeError):
            Clusterability(n_neighbors=5, random_state="invalid")

        with pytest.raises(ValueError):
            Clusterability(n_neighbors=5, random_state=-1)

    def test_hopkins_computation(self):
        """Test Hopkins statistic computation."""
        analyzer = Clusterability(n_neighbors=5, random_state=42)

        # Test with clustered data
        cluster1 = np.random.normal(0, 0.5, (50, 2))
        cluster2 = np.random.normal(5, 0.5, (50, 2))
        clustered_data = np.vstack([cluster1, cluster2])

        hopkins = analyzer.compute_hopkins_statistic(clustered_data)
        assert hopkins > 0.6  # Should be high for clustered data

        # Test with random data
        random_data = np.random.normal(0, 1, (100, 3))
        hopkins = analyzer.compute_hopkins_statistic(random_data)
        assert 0.3 <= hopkins <= 0.7  # Should be around 0.5 for random data

    def test_small_samples(self):
        """Test with small sample sizes."""
        analyzer = Clusterability(n_neighbors=2, random_state=42)

        # Test with 2 samples
        small_data = np.random.randn(2, 2)
        hopkins = analyzer.compute_hopkins_statistic(small_data)
        assert not np.isnan(hopkins)  # Should compute even with 2 samples

        # Test with 3 samples
        small_data = np.random.randn(3, 2)
        hopkins = analyzer.compute_hopkins_statistic(small_data)
        assert not np.isnan(hopkins)

    def test_edge_cases(self):
        """Test edge cases."""
        analyzer = Clusterability(n_neighbors=5)

        # Test with 1 sample (should return nan)
        single_sample = np.random.randn(1, 2)
        hopkins = analyzer.compute_hopkins_statistic(single_sample)
        assert np.isnan(hopkins)

        # Test with 0 samples (should return nan)
        empty_data = np.random.randn(0, 2)
        hopkins = analyzer.compute_hopkins_statistic(empty_data)
        assert np.isnan(hopkins)

        # Test with invalid data dimensions
        with pytest.raises(ValueError):
            analyzer.compute_hopkins_statistic(np.array([1, 2, 3]))  # 1D array

    def test_sliding_window_integration(self):
        """Test integration with SlidingWindow."""
        analyzer = Clusterability(n_neighbors=5, random_state=42)
        window = SlidingWindow(window_size=50, n_columns=2)

        # Fill window with data
        for i in range(50):
            window.add([np.random.normal(0, 1), np.random.normal(0, 1)])

        result = analyzer.compute_clusterability(window)
        assert 'hopkins_statistic' in result
        assert 'sample_size' in result
        assert 'feature_dimension' in result
        assert not np.isnan(result['hopkins_statistic'])
        assert result['sample_size'] == 50
        assert result['feature_dimension'] == 2

    def test_sliding_window_not_full(self):
        """Test with non-full sliding window."""
        analyzer = Clusterability(n_neighbors=5)
        window = SlidingWindow(window_size=50, n_columns=2)

        # Add only 10 samples to window
        for i in range(10):
            window.add([np.random.normal(0, 1), np.random.normal(0, 1)])

        result = analyzer.compute_clusterability(window)
        assert np.isnan(result['hopkins_statistic'])
        assert result['sample_size'] == 0

    def test_call_method(self):
        """Test the __call__ method."""
        analyzer = Clusterability(n_neighbors=5, random_state=42)
        window = SlidingWindow(window_size=30, n_columns=3)

        # Fill window with data
        for i in range(30):
            window.add([np.random.normal(0, 1), np.random.normal(0, 1), np.random.normal(0, 1)])

        result = analyzer(window)
        assert 'hopkins_statistic' in result
        assert 'sample_size' in result
        assert 'feature_dimension' in result
        assert not np.isnan(result['hopkins_statistic'])

    def test_constant_data(self):
        """Test with constant data (all same values)."""
        analyzer = Clusterability(n_neighbors=5)

        # All data points are the same
        constant_data = np.ones((20, 3))
        hopkins = analyzer.compute_hopkins_statistic(constant_data)
        assert np.isnan(hopkins)  # Should return nan when ranges are zero

    def test_different_neighbors(self):
        """Test with different n_neighbors values."""
        for n_neighbors in [2, 5, 10]:
            analyzer = Clusterability(n_neighbors=n_neighbors, random_state=42)
            data = np.random.normal(0, 1, (50, 3))
            hopkins = analyzer.compute_hopkins_statistic(data)
            assert 0 <= hopkins <= 1  # Should be valid probability


if __name__ == "__main__":
    pytest.main([__file__, "-v"])