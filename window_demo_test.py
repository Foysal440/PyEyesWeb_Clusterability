"""Demo test with window data for Clusterability module."""

import numpy as np
import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Mock PyEyesWeb dependencies
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


from clusterability import Clusterability


def demo_window_analysis():
    """Demo clusterability analysis using sliding window with different data patterns."""

    print("=" * 70)
    print("CLUSTERABILITY ANALYSIS DEMO WITH SLIDING WINDOW")
    print("=" * 70)

    # Create analyzer
    analyzer = Clusterability(n_neighbors=5, random_state=42)

    # Test Scenario 1: Random Movement Data
    print("\n" + "=" * 50)
    print("SCENARIO 1: RANDOM MOVEMENT PATTERNS")
    print("=" * 50)

    window1 = SlidingWindow(window_size=100, n_columns=3)  # 3D movement data

    print("Simulating random human movement (walking randomly)...")
    hopkins_values = []
    for i in range(120):
        # Random movement in 3D space (x, y, z coordinates)
        x = np.random.normal(0, 2.0)
        y = np.random.normal(0, 1.5)
        z = np.random.normal(0, 0.8)
        window1.add([x, y, z])

        if window1.is_full() and i % 20 == 0:
            result = analyzer(window1)
            if not np.isnan(result['hopkins_statistic']):
                hopkins_values.append(result['hopkins_statistic'])
                print(f"  Frame {i:3d}: Hopkins = {result['hopkins_statistic']:.3f}")

    # Show summary for scenario 1
    if hopkins_values:
        avg_hopkins = np.mean(hopkins_values)
        print(f"\n  SUMMARY - Random Movement:")
        print(f"  Average Hopkins: {avg_hopkins:.3f}")
        print(f"  Expected: ~0.5 (random distribution)")

    # Test Scenario 2: Repetitive/Cyclic Movement
    print("\n" + "=" * 50)
    print("SCENARIO 2: REPETITIVE MOVEMENT PATTERNS")
    print("=" * 50)

    window2 = SlidingWindow(window_size=100, n_columns=2)  # 2D cyclic motion

    print("Simulating repetitive movement (walking in circles)...")
    hopkins_values = []
    for i in range(120):
        # Cyclic movement pattern (circular motion)
        angle = i * 0.1
        x = 5 * np.cos(angle) + np.random.normal(0, 0.3)  # Circle with noise
        y = 5 * np.sin(angle) + np.random.normal(0, 0.3)
        window2.add([x, y])

        if window2.is_full() and i % 20 == 0:
            result = analyzer(window2)
            if not np.isnan(result['hopkins_statistic']):
                hopkins_values.append(result['hopkins_statistic'])
                print(f"  Frame {i:3d}: Hopkins = {result['hopkins_statistic']:.3f}")

    # Show summary for scenario 2
    if hopkins_values:
        avg_hopkins = np.mean(hopkins_values)
        print(f"\n  SUMMARY - Repetitive Movement:")
        print(f"  Average Hopkins: {avg_hopkins:.3f}")
        print(f"  Expected: >0.6 (clustered distribution)")

    # Test Scenario 3: Transition from Random to Clustered
    print("\n" + "=" * 50)
    print("SCENARIO 3: TRANSITION - RANDOM TO CLUSTERED MOVEMENT")
    print("=" * 50)

    window3 = SlidingWindow(window_size=80, n_columns=2)

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

        window3.add([x, y])

        if window3.is_full() and i % 15 == 0:
            result = analyzer(window3)
            if not np.isnan(result['hopkins_statistic']):
                hopkins_values.append(result['hopkins_statistic'])
                print(f"  Frame {i:3d} ({phase}): Hopkins = {result['hopkins_statistic']:.3f}")

    # Show summary for scenario 3
    if hopkins_values:
        avg_hopkins = np.mean(hopkins_values)
        trend = "increasing" if len(hopkins_values) > 1 and hopkins_values[-1] > hopkins_values[0] else "stable"
        print(f"\n  SUMMARY - Transition Analysis:")
        print(f"  Average Hopkins: {avg_hopkins:.3f}")
        print(f"  Trend: {trend} (positive = increasing clusterability)")

    # Test Scenario 4: Multiple Movement Clusters
    print("\n" + "=" * 50)
    print("SCENARIO 4: MULTIPLE MOVEMENT CLUSTERS")
    print("=" * 50)

    window4 = SlidingWindow(window_size=120, n_columns=3)

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

        window4.add([x, y, z])

        if window4.is_full() and i % 25 == 0:
            result = analyzer(window4)
            if not np.isnan(result['hopkins_statistic']):
                hopkins_values.append(result['hopkins_statistic'])
                print(f"  Frame {i:3d}: Hopkins = {result['hopkins_statistic']:.3f}")

    # Show summary for scenario 4
    if hopkins_values:
        avg_hopkins = np.mean(hopkins_values)
        print(f"\n  SUMMARY - Multiple Clusters:")
        print(f"  Average Hopkins: {avg_hopkins:.3f}")
        print(f"  Expected: >0.7 (strong clustering)")

    # Test Scenario 5: Small Sample Sizes
    print("\n" + "=" * 50)
    print("SCENARIO 5: SMALL SAMPLE SIZES")
    print("=" * 50)

    window5 = SlidingWindow(window_size=10, n_columns=2)

    print("Testing with small window size (10 samples)...")
    for i in range(15):
        x = np.random.normal(0, 1.0)
        y = np.random.normal(0, 1.0)
        window5.add([x, y])

        if window5.is_full():
            result = analyzer(window5)
            print(f"  Frame {i:3d}: Hopkins = {result['hopkins_statistic']:.3f}, Samples = {result['sample_size']}")
            break

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

    print("\n" + "=" * 70)
    print("DEMO COMPLETED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    demo_window_analysis()