# Clusterability

A lightweight Python module for measuring **clustering tendency in multivariate data** using the **Hopkins statistic**.

The module is designed for both static datasets and real-time signal streams. It supports direct NumPy analysis, sliding-window processing, temporal history tracking, and human-readable interpretation of Hopkins scores.

It is suitable for applications such as:

- movement and behavioural signal analysis
- sensor-stream monitoring
- feature-space validation before clustering
- motion-pattern assessment
- exploratory machine-learning pipelines
- real-time human-computer interaction systems

---

## Overview

Clustering algorithms will always produce groups, even when the underlying data has no meaningful cluster structure. The Hopkins statistic helps answer a more fundamental question before clustering:

> **Does this dataset contain evidence of meaningful clusters?**

The statistic compares nearest-neighbour distances from observed data points with distances from uniformly generated points inside the same feature space.

A value near `0.5` generally indicates spatial randomness. Higher values suggest a stronger clustering tendency, while lower values indicate a more uniform distribution.

---

## Features

- Hopkins statistic computation for multidimensional NumPy arrays
- Support for real-time `SlidingWindow` signal buffers
- Configurable sampling fraction
- Reproducible analysis through `random_state`
- Human-readable clusterability labels
- Thread-safe temporal history storage
- Mean, standard deviation, trend, and stability statistics
- Callable analyzer interface
- One-line convenience function for static datasets
- Pytest test suite
- Demonstration script for simulated movement patterns

---

## Interpretation

This implementation uses the following interpretation thresholds:

| Hopkins value | Interpretation |
|---:|---|
| `> 0.75` | Strong clustering |
| `> 0.60` and `<= 0.75` | Moderate clustering |
| `> 0.50` and `<= 0.60` | Weak clustering |
| `> 0.30` and `<= 0.50` | Random distribution |
| `<= 0.30` | Uniform distribution |

These labels are practical guidance rather than universal statistical laws. Hopkins values should be interpreted together with sample size, feature scaling, domain knowledge, and repeated measurements.

---

## Requirements

- Python 3.9 or later
- NumPy
- scikit-learn
- pytest, for running tests
- PyEyesWeb data models and validators, when using the real-time integration

Install the standard Python dependencies with:

```bash
pip install numpy scikit-learn pytest
```

The module imports the following PyEyesWeb components:

```python
from pyeyesweb.data_models.sliding_window import SlidingWindow
from pyeyesweb.data_models.thread_safe_buffer import ThreadSafeHistoryBuffer
from pyeyesweb.utils.validators import (
    validate_integer,
    validate_boolean,
    validate_numeric,
)
```

Make sure the parent project exposes these modules on the Python path.

---

## Suggested Project Structure

```text
project/
├── Clusterability.py
├── test_clusterability.py
├── demo_clusterability.py
├── requirements.txt
└── README.md
```

A minimal `requirements.txt` may contain:

```text
numpy
scikit-learn
pytest
```

If `pyeyesweb` is a separate package, add it according to the way it is distributed in your environment.

---

## Quick Start

### Static NumPy data

Use `assess_clusterability` for a one-time assessment:

```python
import numpy as np

from Clusterability import assess_clusterability

data = np.vstack([
    np.random.normal(loc=0.0, scale=0.5, size=(100, 2)),
    np.random.normal(loc=5.0, scale=0.5, size=(100, 2)),
])

result = assess_clusterability(
    data,
    sample_fraction=0.2,
    random_state=42,
)

print(result)
```

Example output:

```python
{
    "hopkins_statistic": 0.82,
    "interpretation": "STRONG CLUSTERING",
    "sample_size": 200,
    "feature_dimension": 2,
}
```

---

## Using the Analyzer Class

```python
import numpy as np

from Clusterability import Clusterability

analyzer = Clusterability(
    sensitivity=100,
    output_interpretation=True,
    sample_fraction=0.1,
    random_state=42,
)

data = np.random.normal(size=(250, 4))

hopkins = analyzer.compute_hopkins_statistic(data)
label = analyzer.interpret_hopkins_statistic(hopkins)

print(f"Hopkins statistic: {hopkins:.3f}")
print(f"Interpretation: {label}")
```

---

## Real-Time Sliding-Window Analysis

The analyzer can process a populated PyEyesWeb `SlidingWindow`:

```python
from Clusterability import Clusterability
from pyeyesweb.data_models.sliding_window import SlidingWindow

window = SlidingWindow(
    max_length=100,
    n_columns=3,
)

analyzer = Clusterability(
    sensitivity=50,
    sample_fraction=0.15,
    random_state=42,
)

for point in movement_stream:
    window.append(point)

    if window.is_full():
        result = analyzer.compute_clusterability(window)

        print(
            result["hopkins_statistic"],
            result["interpretation"],
        )
```

The result dictionary contains:

| Key | Description |
|---|---|
| `hopkins_statistic` | Hopkins score between 0 and 1 |
| `interpretation` | Categorical interpretation or `None` |
| `sample_size` | Number of observations used |
| `feature_dimension` | Number of input features |

When the window is not full, the method returns `NaN` for the statistic and `None` for the interpretation.

---

## Callable Interface

A `Clusterability` instance can also be called directly:

```python
result = analyzer(window)
```

This computes the clusterability result and prints the Hopkins score. When interpretation output is enabled, the label is printed as well.

---

## Temporal Analysis

Each successfully computed Hopkins value is stored in a thread-safe history buffer.

```python
statistics = analyzer.get_temporal_statistics()

print(statistics)
```

Returned values include:

```python
{
    "mean": 0.71,
    "std": 0.04,
    "trend": 0.002,
    "stability": 0.056,
    "history_length": 40,
}
```

### Temporal fields

| Field | Meaning |
|---|---|
| `mean` | Average Hopkins value in the history |
| `std` | Standard deviation of the history |
| `trend` | Linear slope across stored values |
| `stability` | Coefficient of variation, calculated as `std / mean` |
| `history_length` | Number of stored measurements |

A positive trend may indicate that the signal is becoming more structured or clusterable. A negative trend may indicate increasing randomness or dispersion.

Access or clear the stored history with:

```python
history = analyzer.get_history()
analyzer.reset_history()
```

---

## Constructor Parameters

```python
Clusterability(
    sensitivity=100,
    output_interpretation=True,
    sample_fraction=0.1,
    random_state=None,
)
```

| Parameter | Type | Default | Description |
|---|---|---:|---|
| `sensitivity` | `int` | `100` | Maximum number of Hopkins values retained in history |
| `output_interpretation` | `bool` | `True` | Enables categorical interpretation labels |
| `sample_fraction` | `float` | `0.1` | Fraction of observations sampled for Hopkins computation |
| `random_state` | `int` or `None` | `None` | Seed used for reproducible sampling |

Validation constraints:

- `sensitivity`: between `1` and `10,000`
- `sample_fraction`: between `0.01` and `0.5`
- `random_state`: non-negative integer or `None`

---

## API Reference

### `compute_hopkins_statistic(data)`

Computes the Hopkins statistic from a two-dimensional NumPy array.

```python
score = analyzer.compute_hopkins_statistic(data)
```

Input shape:

```text
(n_samples, n_features)
```

Requirements:

- at least 10 observations
- at least one feature
- two-dimensional input

If one or more features have zero range, the method returns `0.5`.

---

### `interpret_hopkins_statistic(hopkins_stat)`

Maps a Hopkins value to a categorical interpretation.

```python
label = analyzer.interpret_hopkins_statistic(0.78)
# "STRONG CLUSTERING"
```

---

### `compute_clusterability(signals)`

Computes clusterability from a `SlidingWindow` instance and returns a structured result dictionary.

```python
result = analyzer.compute_clusterability(window)
```

---

### `get_temporal_statistics()`

Summarizes the stored Hopkins history.

```python
summary = analyzer.get_temporal_statistics()
```

When fewer than two history values are available, numerical summary fields are returned as `NaN`.

---

### `get_history()`

Returns all stored Hopkins values as a NumPy array.

```python
history = analyzer.get_history()
```

---

### `reset_history()`

Clears all stored Hopkins values.

```python
analyzer.reset_history()
```

---

### `assess_clusterability(data, sample_fraction=0.2, random_state=None)`

Convenience function for analyzing a static NumPy array without manually creating a sliding window.

```python
result = assess_clusterability(
    data,
    sample_fraction=0.2,
    random_state=42,
)
```

---

## Running the Tests

From the project directory:

```bash
pytest -v
```

Or run the test file directly:

```bash
pytest test_clusterability.py -v
```

The included tests cover:

- default and custom initialization
- invalid parameter handling
- clustered and random datasets
- Hopkins interpretation thresholds
- sliding-window integration
- static convenience-function usage
- insufficient-data behaviour
- empty-window behaviour

---

## Running the Demo

Run:

```bash
python demo_clusterability.py
```

The demonstration simulates four movement scenarios:

1. random movement
2. repetitive or cyclic movement
3. transition from random to concentrated movement
4. multiple movement clusters

For each scenario, the script reports Hopkins values and their corresponding interpretation.

---

## Example: Comparing Random and Clustered Data

```python
import numpy as np

from Clusterability import assess_clusterability

rng = np.random.default_rng(42)

random_data = rng.uniform(
    low=-5,
    high=5,
    size=(300, 2),
)

clustered_data = np.vstack([
    rng.normal(loc=(-3, -3), scale=0.4, size=(100, 2)),
    rng.normal(loc=(0, 3), scale=0.4, size=(100, 2)),
    rng.normal(loc=(3, -1), scale=0.4, size=(100, 2)),
])

random_result = assess_clusterability(
    random_data,
    random_state=42,
)

clustered_result = assess_clusterability(
    clustered_data,
    random_state=42,
)

print("Random data:", random_result)
print("Clustered data:", clustered_result)
```

---

## Important Considerations

### Scale the features

Nearest-neighbour distance is sensitive to feature magnitude. Standardize or normalize features when they use different units.

```python
from sklearn.preprocessing import StandardScaler

scaled_data = StandardScaler().fit_transform(data)
```

### Use enough observations

Although the implementation accepts a minimum of 10 samples, larger datasets usually produce more reliable estimates.

### Repeat stochastic analysis

The Hopkins statistic depends on random sampling. Use a fixed `random_state` for reproducibility or evaluate repeated runs for a more stable estimate.

### Interpret geometry carefully

Curved trajectories, rings, manifolds, or cyclic motion may appear structured without forming conventional compact clusters. Hopkins measures clustering tendency, not the quality of a specific clustering algorithm.

### Avoid treating thresholds as clinical or scientific conclusions

The interpretation labels are operational categories. They should not replace domain validation, statistical testing, or downstream clustering evaluation.

---

## Implementation Notes

The calculation follows this general procedure:

1. sample observations from the real dataset
2. generate uniformly distributed points inside the dataset bounds
3. compute nearest-neighbour distances for both groups
4. compare the summed distances
5. return a normalized score between 0 and 1

The implementation uses `sklearn.neighbors.NearestNeighbors` for efficient nearest-neighbour search.

---

## Limitations

- Axis-aligned uniform sampling may be less representative for irregular feature spaces.
- Results may change with feature scaling and dimensionality.
- High-dimensional distance concentration can reduce interpretability.
- The current implementation resets NumPy's global random seed when `random_state` is provided.
- `compute_clusterability` converts internal computation failures into a fallback result instead of re-raising the exception.
- Hopkins alone does not determine the optimal clustering algorithm or number of clusters.

For research-grade use, consider repeated Hopkins estimates, local random-number generators, dimensionality diagnostics, and comparison with additional clustering-tendency measures.

---

## References

- Hopkins, B., & Skellam, J. G. (1954). *A new method for determining the type of distribution of plant individuals*. Annals of Botany.
- Lawson, R. G., & Jurs, P. C. (1990). *New index for clustering tendency and its application to chemical problems*. Journal of Chemical Information and Computer Sciences.

---

## Contributing

Contributions are welcome.

A typical contribution workflow is:

```bash
git checkout -b feature/your-change
pytest -v
git commit -m "Describe the change"
git push origin feature/your-change
```

Open a pull request with:

- a clear explanation of the change
- tests for new behaviour
- updated documentation where necessary
- confirmation that the existing test suite passes

---

## License

Copyright: University of Genoa,Italy

