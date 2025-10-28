"""Pytest tests for Clusterability module."""

import numpy as np
import pytest
import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Mock dependencies
class SlidingWindow:
    def __init__(self, max_length: int, n_columns: int):
        self.max_length = max_length
        self.n_columns = n_columns
        self._n_columns = n_columns  # Add this line to fix the error
        self.buffer = []

    def append(self, data_point):
        if len(data_point) != self.n_columns:
            raise ValueError(f"Expected {self.n_columns} columns, got {len(data_point)}")
        self.buffer.append(data_point)
        if len(self.buffer) > self.max_length:
            self.buffer.pop(0)

    def is_full(self):
        return len(self.buffer) >= self.max_length

    def to_array(self):
        return np.array(self.buffer), None


class ThreadSafeHistoryBuffer:
    def __init__(self, maxlen: int = 100):
        self.buffer = []
        self.maxlen = maxlen

    def append(self, value):
        self.buffer.append(value)
        if len(self.buffer) > self.maxlen:
            self.buffer.pop(0)

    def get_all(self):
        return self.buffer.copy()

    def clear(self):
        self.buffer.clear()


def validate_integer(value, name, min_val=None, max_val=None):
    if not isinstance(value, int):
        raise TypeError(f"{name} must be integer")
    if min_val is not None and value < min_val:
        raise ValueError(f"{name} must be >= {min_val}")
    if max_val is not None and value > max_val:
        raise ValueError(f"{name} must be <= {max_val}")
    return value

def validate_boolean(value, name):
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be boolean")
    return value

def validate_numeric(value, name, min_val=None, max_val=None):
    if not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric")
    if min_val is not None and value < min_val:
        raise ValueError(f"{name} must be >= {min_val}")
    if max_val is not None and value > max_val:
        raise ValueError(f"{name} must be <= {max_val}")
    return float(value)


# Import the module to test
from clusterability import Clusterability, assess_clusterability


class TestClusterability:
    """Test cases for Clusterability class."""

    def test_initialization(self):
        """Test Clusterability initialization."""
        analyzer = Clusterability()
        assert analyzer.sample_fraction == 0.1
        assert analyzer.output_interpretation == True

        analyzer = Clusterability(sensitivity=50, output_interpretation=False, sample_fraction=0.2)
        assert analyzer.sample_fraction == 0.2
        assert analyzer.output_interpretation == False

    def test_invalid_parameters(self):
        """Test invalid parameter handling."""
        with pytest.raises(ValueError):
            Clusterability(sensitivity=0)

        with pytest.raises(ValueError):
            Clusterability(sample_fraction=1.5)

    def test_hopkins_computation(self):
        """Test Hopkins statistic computation."""
        analyzer = Clusterability(random_state=42)

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

    def test_interpretation(self):
        """Test Hopkins statistic interpretation."""
        analyzer = Clusterability()

        assert analyzer.interpret_hopkins_statistic(0.85) == "STRONG CLUSTERING"
        assert analyzer.interpret_hopkins_statistic(0.68) == "MODERATE CLUSTERING"
        assert analyzer.interpret_hopkins_statistic(0.55) == "WEAK CLUSTERING"
        assert analyzer.interpret_hopkins_statistic(0.45) == "RANDOM DISTRIBUTION"
        assert analyzer.interpret_hopkins_statistic(0.25) == "UNIFORM DISTRIBUTION"

    def test_sliding_window_integration(self):
        """Test integration with SlidingWindow."""
        analyzer = Clusterability(random_state=42)
        window = SlidingWindow(max_length=50, n_columns=2)

        # Fill window with data
        for i in range(50):
            window.append([np.random.normal(0, 1), np.random.normal(0, 1)])

        result = analyzer.compute_clusterability(window)
        assert 'hopkins_statistic' in result
        assert 'interpretation' in result
        assert not np.isnan(result['hopkins_statistic'])

    def test_assess_clusterability_function(self):
        """Test the convenience function."""
        data = np.random.normal(0, 1, (100, 3))
        result = assess_clusterability(data, random_state=42)

        assert 'hopkins_statistic' in result
        assert 'interpretation' in result
        assert 'sample_size' in result
        assert 'feature_dimension' in result
        assert not np.isnan(result['hopkins_statistic'])

    def test_edge_cases(self):
        """Test edge cases."""
        analyzer = Clusterability()

        # Test with insufficient data
        with pytest.raises(ValueError):
            small_data = np.random.randn(5, 2)
            analyzer.compute_hopkins_statistic(small_data)

        # Test with empty window
        empty_window = SlidingWindow(max_length=50, n_columns=2)
        result = analyzer.compute_clusterability(empty_window)
        assert np.isnan(result['hopkins_statistic'])
        assert result['interpretation'] is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])