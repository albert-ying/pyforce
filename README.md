# PyForce

Python implementation of ggforce-style annotations for matplotlib.

PyForce brings the elegant annotation capabilities of R's [ggforce](https://github.com/thomasp85/ggforce) package to Python's matplotlib. Create publication-quality visualizations with smart point annotations and convex hull groupings.

## Features

**Smart Point Annotations** with `annotate_points()`:
- Three connector styles: `horizontal`, `elbow`, or `straight`
- Horizontal mode matches publication-quality scientific figures
- Lines start from marker edges, not centers
- Smart direction selection (prefers right, switches to left near edges)
- Automatic text alignment based on label position
- Uses adjustText for collision-free label placement
- Configurable repulsion forces for fine-tuning

**Convex Hull Groupings** with `geom_mark_hull()`:
- Smooth, expanded boundaries around point groups
- Cubic spline interpolation for aesthetic curves
- Same connector options as point annotations
- Smart label positioning away from data points

## Installation

```bash
pip install pyforce
```

Or install from source:

```bash
git clone https://github.com/yourusername/pyforce
cd pyforce
pip install -e .
```

## Examples

### Annotating Points (Volcano Plot)

Create publication-ready volcano plots with automatic label positioning:

```python
import matplotlib.pyplot as plt
import numpy as np

from pyforce import annotate_points

# Generate data
np.random.seed(42)
n_points = 200
log_fc = np.random.randn(n_points) * 2
neg_log_p = np.abs(np.random.randn(n_points)) * 1.5 + np.abs(log_fc) * 0.3

# Identify significant points
significant = (np.abs(log_fc) > 1.5) & (neg_log_p > 2)
sig_indices = np.where(significant)[0]
gene_names = [f"Gene_{i}" for i in sig_indices]

# Create plot
fig, ax = plt.subplots(figsize=(12, 9))
colors = np.where(significant, "crimson", "gray")
ax.scatter(log_fc, neg_log_p, c=colors, s=60, alpha=0.6)

# Smart annotations
annotate_points(
    ax,
    log_fc,
    neg_log_p,
    labels=gene_names,
    indices=sig_indices,
    point_size=60,
    label_fontsize=8,
    label_color="darkred",
    connection_color="gray",
    force_points=1.5,   # Repulsion from points
    expand_points=3.0,  # Collision box size
)

plt.show()
```

![Volcano Plot Example](assets/volcano_example.png)

### Hull Annotations (Clustered Data)

Annotate groups of points with smooth convex hulls:

```python
import matplotlib.pyplot as plt
import numpy as np

from pyforce import geom_mark_hull

# Generate clustered data
np.random.seed(42)
cluster1 = np.random.randn(20, 2) * 0.5 + np.array([2, 2])
cluster2 = np.random.randn(25, 2) * 0.6 + np.array([-2, 1])
cluster3 = np.random.randn(18, 2) * 0.4 + np.array([0, -2])

points = np.vstack([cluster1, cluster2, cluster3])
x, y = points[:, 0], points[:, 1]
groups = np.array([0] * 20 + [1] * 25 + [2] * 18)

# Create plot
fig, ax = plt.subplots(figsize=(10, 8))
colors = ["#E63946", "#457B9D", "#2A9D8F"]

for cluster, color in zip([cluster1, cluster2, cluster3], colors):
    ax.scatter(cluster[:, 0], cluster[:, 1], c=color, s=60, alpha=0.7)

# Hull annotations with elbow connectors
geom_mark_hull(
    ax,
    x,
    y,
    groups=groups,
    labels=["Cluster A", "Cluster B", "Cluster C"],
    hull_color=colors,
    hull_fill=colors,
    hull_alpha=0.12,
    label_fontsize=11,
    label_fontweight="bold",
    elbow_angle=45.0,
)

plt.show()
```

![Hull Annotation Example](assets/hull_example.png)

### Paper-Style Annotations

For publication-quality figures with horizontal connectors:

```python
import matplotlib.pyplot as plt
import numpy as np
from pyforce import annotate_points

# Data
names = ["CPLX2", "CPLX1", "OLFM1", "NRXN3.2", "STMN2", "FGF4", 
         "LANCL1", "NPTXR", "CNDP1", "ALDOC"]
bootstrap = np.array([85, 78, 92, 45, 55, 95, 12, 88, 75, 98])
coef = np.array([1.8, 1.2, 0.9, 0.6, 0.2, 0.1, -0.8, -1.2, -1.5, -2.0])

fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(bootstrap, coef, s=100, c="#008080")

annotate_points(
    ax, bootstrap, coef,
    labels=names,
    point_size=100,
    connector_type="horizontal",  # Clean horizontal lines
    prefer_direction="right",     # Labels prefer right side
    only_move_text="x",           # Keep vertical alignment
)

plt.show()
```

![Paper Style Example](assets/paper_style_example.png)

## API Reference

### `annotate_points()`

Annotate points with smart connectors.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `ax` | Axes | required | Matplotlib axes to draw on |
| `x`, `y` | array-like | required | All point coordinates |
| `labels` | list[str] | required | Text labels for points |
| `indices` | array-like | None | Indices of points to annotate |
| `point_size` | float | 40 | Scatter point size for edge calculation |
| `connector_type` | str | 'horizontal' | Connector style: 'horizontal', 'elbow', 'straight' |
| `prefer_direction` | str | 'right' | Preferred label direction: 'right' or 'left' |
| `only_move_text` | str | 'x' | Direction to move labels: 'x', 'y', or 'xy' |
| `connection_linewidth` | float | 0.8 | Connector line width |
| `connection_color` | str | 'gray' | Connector color |
| `elbow_angle` | float | 45.0 | Angle for elbow connectors (degrees) |
| `force_points` | float | 1.0 | Repulsion force from points |
| `force_text` | float | 0.5 | Repulsion between labels |

### `geom_mark_hull()`

Annotate groups with convex hulls and smart connectors.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `ax` | Axes | required | Matplotlib axes to draw on |
| `x`, `y` | array-like | required | Point coordinates |
| `groups` | array-like | None | Group membership |
| `labels` | list[str] | None | Labels for each group |
| `hull_color` | str/list | 'black' | Hull boundary color(s) |
| `hull_fill` | str/list | None | Hull fill color(s) |
| `hull_alpha` | float | 0.2 | Fill transparency |
| `expand_factor` | float | 0.12 | Hull expansion factor |
| `connector_type` | str | 'horizontal' | Style: 'horizontal', 'elbow', 'straight' |
| `connection_linewidth` | float | 1.0 | Connector width |
| `connection_color` | str | 'black' | Connector color |

## Requirements

- Python >= 3.8
- numpy >= 1.20.0
- matplotlib >= 3.5.0
- scipy >= 1.7.0
- adjustText >= 0.8

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

MIT License. See LICENSE file for details.

## Acknowledgments

Inspired by Thomas Lin Pedersen's excellent [ggforce](https://github.com/thomasp85/ggforce) package for R.
