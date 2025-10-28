"""Demo test with window data for Clusterability module."""

import numpy as np
import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Mock PyEyesWeb dependencies
class SlidingWindow:
    def __init__(self, max_length: int, n_columns: int):
        self.max_length = max_length
        self.n_columns = n_columns
        self._n_columns = n_columns
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

    def get_all(self):  # ADD THIS METHOD
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


from Clusterability import Clusterability, assess_clusterability


def demo_window_analysis():
    """Demo clusterability analysis using sliding window with different data patterns."""

    print("=" * 70)
    print("CLUSTERABILITY ANALYSIS DEMO WITH SLIDING WINDOW")
    print("=" * 70)

    # Create analyzer
    analyzer = Clusterability(
        sensitivity=15,
        output_interpretation=True,
        sample_fraction=0.15,
        random_state=42
    )

    # Test Scenario 1: Random Movement Data
    print("\n" + "=" * 50)
    print("SCENARIO 1: RANDOM MOVEMENT PATTERNS")
    print("=" * 50)

    window1 = SlidingWindow(max_length=100, n_columns=3)  # 3D movement data

    print("Simulating random human movement (walking randomly)...")
    hopkins_values = []
    for i in range(120):
        # Random movement in 3D space (x, y, z coordinates)
        x = np.random.normal(0, 2.0)
        y = np.random.normal(0, 1.5)
        z = np.random.normal(0, 0.8)
        window1.append([x, y, z])

        if window1.is_full() and i % 20 == 0:
            result = analyzer(window1)
            if not np.isnan(result['hopkins_statistic']):
                hopkins_values.append(result['hopkins_statistic'])
                print(f"  Frame {i:3d}: Hopkins = {result['hopkins_statistic']:.3f} -> {result['interpretation']}")

    # Show summary for scenario 1
    if hopkins_values:
        avg_hopkins = np.mean(hopkins_values)
        print(f"\n  SUMMARY - Random Movement:")
        print(f"  Average Hopkins: {avg_hopkins:.3f}")
        print(f"  Pattern: {analyzer.interpret_hopkins_statistic(avg_hopkins)}")

    analyzer.reset_history()

    # Test Scenario 2: Repetitive/Cyclic Movement
    print("\n" + "=" * 50)
    print("SCENARIO 2: REPETITIVE MOVEMENT PATTERNS")
    print("=" * 50)

    window2 = SlidingWindow(max_length=100, n_columns=2)  # 2D cyclic motion

    print("Simulating repetitive movement (walking in circles)...")
    hopkins_values = []
    for i in range(120):
        # Cyclic movement pattern (circular motion)
        angle = i * 0.1
        x = 5 * np.cos(angle) + np.random.normal(0, 0.3)  # Circle with noise
        y = 5 * np.sin(angle) + np.random.normal(0, 0.3)
        window2.append([x, y])

        if window2.is_full() and i % 20 == 0:
            result = analyzer(window2)
            if not np.isnan(result['hopkins_statistic']):
                hopkins_values.append(result['hopkins_statistic'])
                print(f"  Frame {i:3d}: Hopkins = {result['hopkins_statistic']:.3f} -> {result['interpretation']}")

    # Show summary for scenario 2
    if hopkins_values:
        avg_hopkins = np.mean(hopkins_values)
        print(f"\n  SUMMARY - Repetitive Movement:")
        print(f"  Average Hopkins: {avg_hopkins:.3f}")
        print(f"  Pattern: {analyzer.interpret_hopkins_statistic(avg_hopkins)}")

    analyzer.reset_history()

    # Test Scenario 3: Transition from Random to Clustered
    print("\n" + "=" * 50)
    print("SCENARIO 3: TRANSITION - RANDOM TO CLUSTERED MOVEMENT")
    print("=" * 50)

    window3 = SlidingWindow(max_length=80, n_columns=2)

    print("Simulating movement transition (random walking -> focused activity)...")
    hopkins_values = []
    for i in range(150):
        if i < 50:
            # Phase 1: Random movement
            x = np.random.normal(0, 2.0)
            y = np.random.normal(0, 2.0)
            phase = "RANDOM"
        elif i < 100:
            # Phase 2: Transition - starting to cluster
            x = np.random.normal(2, 1.2)
            y = np.random.normal(2, 1.2)
            phase = "TRANSITION"
        else:
            # Phase 3: Clustered movement - focused activity
            x = np.random.normal(5, 0.5)
            y = np.random.normal(5, 0.5)
            phase = "CLUSTERED"

        window3.append([x, y])

        if window3.is_full() and i % 15 == 0:
            result = analyzer(window3)
            if not np.isnan(result['hopkins_statistic']):
                hopkins_values.append(result['hopkins_statistic'])
                print(f"  Frame {i:3d} ({phase}): Hopkins = {result['hopkins_statistic']:.3f} -> {result['interpretation']}")

    # Show summary for scenario 3
    if hopkins_values:
        avg_hopkins = np.mean(hopkins_values)
        trend = "increasing" if len(hopkins_values) > 1 and hopkins_values[-1] > hopkins_values[0] else "stable"
        print(f"\n  SUMMARY - Transition Analysis:")
        print(f"  Average Hopkins: {avg_hopkins:.3f}")
        print(f"  Trend: {trend} (positive = increasing clusterability)")
        print(f"  Final Pattern: {analyzer.interpret_hopkins_statistic(avg_hopkins)}")

    # Test Scenario 4: Multiple Movement Clusters
    print("\n" + "=" * 50)
    print("SCENARIO 4: MULTIPLE MOVEMENT CLUSTERS")
    print("=" * 50)

    window4 = SlidingWindow(max_length=120, n_columns=3)

    print("Simulating multiple activity clusters (different movement types)...")
    hopkins_values = []
    for i in range(180):
        # Alternate between 3 different movement patterns
        pattern = i % 3

        if pattern == 0:
            # Cluster 1: Slow, precise movements
            x = np.random.normal(0, 0.3)
            y = np.random.normal(0, 0.3)
            z = np.random.normal(0, 0.2)
        elif pattern == 1:
            # Cluster 2: Medium, regular movements
            x = np.random.normal(3, 0.8)
            y = np.random.normal(3, 0.8)
            z = np.random.normal(1, 0.4)
        else:
            # Cluster 3: Fast, broad movements
            x = np.random.normal(-2, 1.5)
            y = np.random.normal(-2, 1.5)
            z = np.random.normal(2, 0.8)

        window4.append([x, y, z])

        if window4.is_full() and i % 25 == 0:
            result = analyzer(window4)
            if not np.isnan(result['hopkins_statistic']):
                hopkins_values.append(result['hopkins_statistic'])
                print(f"  Frame {i:3d}: Hopkins = {result['hopkins_statistic']:.3f} -> {result['interpretation']}")

    # Show summary for scenario 4
    if hopkins_values:
        avg_hopkins = np.mean(hopkins_values)
        print(f"\n  SUMMARY - Multiple Clusters:")
        print(f"  Average Hopkins: {avg_hopkins:.3f}")
        print(f"  Pattern: {analyzer.interpret_hopkins_statistic(avg_hopkins)}")

    # Final comprehensive analysis
    print("\n" + "=" * 70)
    print("FINAL ANALYSIS SUMMARY")
    print("=" * 70)

    print("\nINTERPRETATION GUIDE:")
    print("> 0.75: STRONG CLUSTERING    - Clear, well-separated movement patterns")
    print("0.6-0.75: MODERATE CLUSTERING - Some repetitive/structured movements")
    print("0.5-0.6: WEAK CLUSTERING     - Minimal pattern recognition")
    print("0.3-0.5: RANDOM DISTRIBUTION  - Unstructured, random movements")
    print("< 0.3: UNIFORM DISTRIBUTION  - Evenly spread, no distinct patterns")

    print("\nAPPLICATION TO HUMAN MOVEMENT ANALYSIS:")
    print("• High Hopkins (>0.7): Repetitive exercises, gait patterns, skilled movements")
    print("• Medium Hopkins (0.5-0.7): Normal walking, semi-structured activities")
    print("• Low Hopkins (<0.5): Random exploration, unstructured movement, rest")
    print("• Increasing trend: Movement becoming more structured/purposeful")
    print("• Decreasing trend: Movement becoming more random/exploratory")

    print("\n DEMO COMPLETED SUCCESSFULLY! ")


if __name__ == "__main__":
    demo_window_analysis()