# PyForce

Python implementation of ggforce-style annotations for matplotlib.

PyForce brings the elegant annotation capabilities of R's [ggforce](https://github.com/thomasp85/ggforce) package to Python's matplotlib. Create publication-quality visualizations with smart annotations.

## Features

- **Smart Point Annotations** (`annotate_points`) - Elbow connectors with intelligent collision avoidance
  - Automatic label positioning that avoids points, other labels, and axes
  - Multiple candidate positions scored to find optimal placement
- **Margin Annotations** (`annotate_margin`) - 3-segment connectors with labels at plot edges
- **Edge Annotations** (`annotate_edge`) - Universal function for heatmaps, line plots, any plot
- **Convex Hull Groupings** (`geom_mark_hull`) - Smooth boundaries around point groups
  - Labels guaranteed to be outside all polygons
  - Smart positioning avoids points, other labels, and plot boundaries

## Installation

```bash
pip install pyforce-plot
```

Or install from source:

```bash
git clone https://github.com/albert-ying/pyforce
cd pyforce
pip install -e .
```

## Examples

### Smart Elbow Annotations

```python
from pyforce import annotate_points

annotate_points(
    ax, x, y,
    labels=gene_names,
    indices=sig_indices,
    point_size=60,
    connector_type="elbow",
    force_points=1.5,
)
```

![Volcano Plot - Elbow](assets/volcano_elbow.png)

### Margin Annotations (3-segment connectors)

```python
from pyforce import annotate_margin

annotate_margin(
    ax, x, y,
    labels=gene_names,
    indices=sig_indices,
    side="both",  # 'right', 'left', or 'both'
)
```

![Volcano Plot - Margin](assets/volcano_margin.png)

### Edge Annotations (Universal)

Works for **heatmaps**, **line plots**, or **any plot**:

```python
from pyforce import annotate_edge

# For heatmap rows
annotate_edge(
    ax,
    y_positions=[74, 75, 76],
    labels=["Target_A", "Target_B", "Target_C"],
    x_start=n_cols - 0.5,  # Heatmap edge
    min_spacing=2.0,
)

# For line plot ends
annotate_edge(
    ax,
    y_positions=[y1[-1], y2[-1], y3[-1]],
    labels=["Line 1", "Line 2", "Line 3"],
    x_start=x[-1],  # End of lines
    min_spacing=0.4,
)
```

**Smart dodge**: Straight line when labels are far apart, 3-segment (1/3 + 1/3 + 1/3) when close.

![Heatmap Example](assets/heatmap_example.png)
![Line Plot Example](assets/lineplot_example.png)

### Hull Annotations

```python
from pyforce import geom_mark_hull

geom_mark_hull(
    ax, x, y,
    groups=groups,
    labels=["Cluster A", "Cluster B", "Cluster C"],
    hull_alpha=0.12,
)
```

![Hull Example](assets/hull_example.png)

## API Reference

### `annotate_points()`

Smart elbow annotations with intelligent collision avoidance. Labels are automatically positioned to avoid overlapping with points, other labels, and axes.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `ax` | Axes | required | Matplotlib axes |
| `x`, `y` | array-like | required | Point coordinates |
| `labels` | list[str] | required | Labels for points |
| `indices` | array-like | None | Indices to annotate |
| `connector_type` | str | 'horizontal' | 'elbow', 'straight', 'horizontal' |
| `offset_factor` | float | 3.5 | Label offset multiplier |
| `force_points` | float | 1.5 | Repulsion from points |
| `force_text` | float | 0.8 | Repulsion between labels |

### `annotate_margin()`

Labels aligned at plot margins with 3-segment connectors.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `ax` | Axes | required | Matplotlib axes |
| `x`, `y` | array-like | required | Point coordinates |
| `labels` | list[str] | required | Labels for points |
| `side` | str | 'right' | 'right', 'left', 'both' |

### `annotate_edge()`

Universal edge annotation for any plot type.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `ax` | Axes | required | Matplotlib axes |
| `y_positions` | list[float] | required | Y positions to annotate |
| `labels` | list[str] | required | Labels |
| `x_start` | float | None | X where connectors start |
| `side` | str | 'right' | 'right' or 'left' |
| `min_spacing` | float | auto | Min spacing between labels |

### `geom_mark_hull()`

Convex hull annotations for grouped data. Labels are intelligently positioned outside all polygons, avoiding points and plot boundaries.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `ax` | Axes | required | Matplotlib axes |
| `x`, `y` | array-like | required | Point coordinates |
| `groups` | array-like | None | Group membership |
| `labels` | list[str] | None | Group labels |
| `hull_alpha` | float | 0.2 | Fill transparency |
| `label_buffer_factor` | float | 0.25 | Distance from hull to label |

## Requirements

- Python >= 3.8
- numpy >= 1.20.0
- matplotlib >= 3.5.0
- scipy >= 1.7.0
- adjustText >= 0.8

## License

MIT License

## Acknowledgments

Inspired by [ggforce](https://github.com/thomasp85/ggforce) for R.
