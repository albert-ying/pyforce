"""
PyForce: ggforce-style annotations for matplotlib.

This package provides smart annotation tools for matplotlib plots,
inspired by R's ggforce package.
"""

from pyforce.annotations import (
    annotate_edge,
    annotate_heatmap_rows,
    annotate_margin,
    annotate_points,
    geom_mark_hull,
    geom_text_repel,
)

__version__ = "0.1.0"
__all__ = [
    "annotate_points",
    "annotate_margin",
    "annotate_edge",
    "annotate_heatmap_rows",
    "geom_mark_hull",
    "geom_text_repel",
    "__version__",
]
