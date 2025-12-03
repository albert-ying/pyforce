"""
Smart annotation system with intelligent connector logic for matplotlib.

This module provides functions for annotating points and groups in matplotlib plots
with automatic label positioning and smart connector lines, inspired by R's ggforce.
"""

import warnings
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import PathPatch, Polygon
from matplotlib.path import Path
from scipy.interpolate import splev, splprep
from scipy.spatial import ConvexHull

try:
    from adjustText import adjust_text

    HAS_ADJUST_TEXT = True
except ImportError:
    HAS_ADJUST_TEXT = False
    warnings.warn(
        "adjustText not available. Install with: pip install adjustText", ImportWarning
    )


@dataclass
class ConnectorStyle:
    """Configuration for connector line style."""

    linewidth: float = 1.0
    color: str = "gray"
    elbow_angle: float = 45.0
    min_distance: float = 0.3


def _get_data_scale(ax: plt.Axes) -> Tuple[float, float]:
    """Get the scale of data units relative to display units."""
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    bbox = ax.get_window_extent()

    x_scale = (xlim[1] - xlim[0]) / bbox.width
    y_scale = (ylim[1] - ylim[0]) / bbox.height

    return x_scale, y_scale


def _point_size_to_radius(point_size: float, ax: plt.Axes) -> float:
    """
    Convert matplotlib point size to radius in data units.

    Uses the actual axes transformation for accurate conversion.
    """
    x_scale, y_scale = _get_data_scale(ax)
    avg_scale = (x_scale + y_scale) / 2

    # Point size is area in points^2, radius in points = sqrt(size/pi)
    radius_points = np.sqrt(point_size / np.pi)
    # Convert to data units (72 points per inch)
    radius_data = radius_points * avg_scale * 0.8

    return radius_data


def _create_elbow_connector(
    start_x: float,
    start_y: float,
    end_x: float,
    end_y: float,
    start_radius: float = 0.0,
    elbow_angle: float = 45.0,
) -> Path:
    """
    Create an elbow connector path from start to end point.

    The connector has two segments:
    1. Diagonal segment from start (offset by radius) at the specified angle
    2. Horizontal segment connecting to end point

    Parameters
    ----------
    start_x, start_y : float
        Starting point coordinates (e.g., data point or hull edge)
    end_x, end_y : float
        Ending point coordinates (e.g., label position)
    start_radius : float
        Offset from start point (to start from edge of marker)
    elbow_angle : float
        Angle of the diagonal segment in degrees (default 45)

    Returns
    -------
    Path
        Matplotlib Path object for the connector
    """
    dx = end_x - start_x
    dy = end_y - start_y
    dist = np.sqrt(dx**2 + dy**2)

    if dist < 1e-10:
        return Path(np.array([[start_x, start_y]]), [Path.MOVETO])

    # Calculate actual start point (offset from center by radius)
    unit_dx = dx / dist
    unit_dy = dy / dist
    actual_start_x = start_x + unit_dx * start_radius
    actual_start_y = start_y + unit_dy * start_radius

    # Elbow point: horizontal segment is at end_y height
    elbow_y = end_y

    # Calculate elbow x position based on desired angle
    angle_rad = np.radians(elbow_angle)
    vertical_dist = abs(elbow_y - actual_start_y)

    if vertical_dist > 1e-10:
        # Calculate horizontal distance for desired angle
        desired_horiz = vertical_dist / np.tan(angle_rad)
        max_horiz = abs(end_x - actual_start_x)

        if desired_horiz < max_horiz * 0.85:
            elbow_x = actual_start_x + np.sign(dx) * desired_horiz
        else:
            # Fallback: transition at 65% of the way
            elbow_x = actual_start_x + dx * 0.65
    else:
        # Nearly horizontal: just use straight line
        elbow_x = actual_start_x + dx * 0.5
        elbow_y = actual_start_y + dy * 0.5

    vertices = np.array(
        [[actual_start_x, actual_start_y], [elbow_x, elbow_y], [end_x, end_y]]
    )
    codes = [Path.MOVETO, Path.LINETO, Path.LINETO]

    return Path(vertices, codes)


def _draw_connector(
    ax: plt.Axes,
    start_x: float,
    start_y: float,
    end_x: float,
    end_y: float,
    start_radius: float,
    style: ConnectorStyle,
) -> Optional[PathPatch]:
    """
    Draw an elbow connector between two points.

    Returns None if distance is below minimum threshold.
    """
    dist = np.sqrt((end_x - start_x) ** 2 + (end_y - start_y) ** 2)

    if dist < style.min_distance:
        return None

    path = _create_elbow_connector(
        start_x,
        start_y,
        end_x,
        end_y,
        start_radius=start_radius,
        elbow_angle=style.elbow_angle,
    )

    patch = PathPatch(
        path,
        facecolor="none",
        edgecolor=style.color,
        linewidth=style.linewidth,
        zorder=5,
        capstyle="round",
        joinstyle="round",
    )
    ax.add_patch(patch)

    return patch


def _adjust_text_alignment(text_obj: plt.Text, point_x: float, point_y: float) -> None:
    """Adjust text alignment based on position relative to point."""
    label_x, label_y = text_obj.get_position()
    dx = label_x - point_x

    # Horizontal alignment based on direction
    if abs(dx) > 0.01:
        text_obj.set_ha("left" if dx > 0 else "right")
    else:
        text_obj.set_ha("center")

    text_obj.set_va("center")


def annotate_points(
    ax: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    labels: List[str],
    indices: Optional[np.ndarray] = None,
    point_size: float = 40,
    min_distance_for_connector: float = 0.3,
    label_fontsize: int = 10,
    label_fontweight: str = "normal",
    label_color: str = "black",
    connection_linewidth: float = 1.0,
    connection_color: str = "black",
    elbow_angle: float = 45.0,
    force_points: float = 1.0,
    force_text: float = 0.8,
    expand_points: float = 2.0,
    expand_text: float = 1.5,
    use_adjust_text: bool = True,
) -> List[plt.Artist]:
    """
    Annotate points with smart elbow connectors and collision-free labels.

    Labels are automatically positioned to avoid overlaps with points and each other.
    Elbow connectors (diagonal + horizontal segments) are drawn from marker edges
    to label positions.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to draw on
    x, y : array-like
        All point coordinates in the plot
    labels : list of str
        Text labels for points to annotate
    indices : array-like, optional
        Indices of points to annotate. If None, annotates all points.
    point_size : float, default=40
        Size of scatter points (for calculating edge offset)
    min_distance_for_connector : float, default=0.3
        Minimum distance before drawing a connector
    label_fontsize : int, default=10
        Font size for labels
    label_fontweight : str, default='normal'
        Font weight for labels
    label_color : str, default='black'
        Text color
    connection_linewidth : float, default=1.0
        Line width for connectors
    connection_color : str, default='gray'
        Line color for connectors
    elbow_angle : float, default=45.0
        Angle of the diagonal segment in degrees
    force_points : float, default=1.0
        Repulsion force from points (higher = more repulsion)
    force_text : float, default=0.8
        Repulsion force between texts
    expand_points : float, default=2.0
        Expansion factor for point collision boxes
    expand_text : float, default=1.5
        Expansion factor for text collision boxes
    use_adjust_text : bool, default=True
        Whether to use adjustText for automatic label positioning

    Returns
    -------
    list of matplotlib.artist.Artist
        List of artists created (text objects and connector patches)
    """
    x = np.asarray(x)
    y = np.asarray(y)

    if indices is None:
        indices = np.arange(len(labels)) if len(labels) <= len(x) else np.arange(len(x))
    indices = np.asarray(indices)

    if isinstance(labels, str):
        labels = [labels]

    artists = []
    text_objects = []
    point_positions = []

    # Calculate point radius for edge offset
    point_radius = _point_size_to_radius(point_size, ax)

    # Create text objects at offset positions initially
    for idx, label in zip(indices, labels):
        if idx >= len(x):
            warnings.warn(f"Index {idx} out of bounds for data of length {len(x)}")
            continue

        point_x, point_y = x[idx], y[idx]
        point_positions.append((point_x, point_y))

        # Initial position: offset from point
        text_obj = ax.text(
            point_x,
            point_y,
            label,
            fontsize=label_fontsize,
            fontweight=label_fontweight,
            color=label_color,
            ha="center",
            va="center",
            zorder=10,
        )
        text_objects.append(text_obj)

    # Use adjustText for collision-free positioning
    if use_adjust_text and HAS_ADJUST_TEXT and text_objects:
        adjust_text(
            text_objects,
            x=x,
            y=y,
            arrowprops=None,
            expand_points=(expand_points, expand_points),
            expand_text=(expand_text, expand_text),
            force_points=(force_points, force_points),
            force_text=(force_text, force_text),
            lim=3000,
            only_move={"points": "xy", "text": "xy"},
            avoid_self=True,
        )

    # Draw connectors AFTER text positioning
    connector_style = ConnectorStyle(
        linewidth=connection_linewidth,
        color=connection_color,
        elbow_angle=elbow_angle,
        min_distance=min_distance_for_connector,
    )

    for text_obj, (point_x, point_y) in zip(text_objects, point_positions):
        label_x, label_y = text_obj.get_position()

        # Adjust text alignment based on direction
        _adjust_text_alignment(text_obj, point_x, point_y)

        # Draw connector
        connector = _draw_connector(
            ax,
            point_x,
            point_y,
            label_x,
            label_y,
            start_radius=point_radius,
            style=connector_style,
        )

        if connector:
            artists.append(connector)
        artists.append(text_obj)

    return artists


def _expand_polygon(vertices: np.ndarray, expand_factor: float = 0.1) -> np.ndarray:
    """Expand polygon outward from its centroid."""
    centroid = np.mean(vertices, axis=0)
    vectors = vertices - centroid
    return centroid + vectors * (1 + expand_factor)


def _smooth_polygon(vertices: np.ndarray, smoothness: int = 150) -> np.ndarray:
    """Smooth polygon using cubic spline interpolation."""
    if len(vertices) < 3:
        return vertices

    vertices_closed = np.vstack([vertices, vertices[0]])

    try:
        tck, _ = splprep(
            [vertices_closed[:, 0], vertices_closed[:, 1]], s=0, per=True, k=3
        )
        u_new = np.linspace(0, 1, smoothness)
        smoothed = np.array(splev(u_new, tck)).T
        return smoothed
    except Exception as e:
        warnings.warn(f"Polygon smoothing failed: {e}. Using original vertices.")
        return vertices


def _find_hull_edge_point(
    hull_vertices: np.ndarray, label_x: float, label_y: float
) -> Tuple[float, float]:
    """Find the nearest point on hull boundary to the label."""
    distances = np.sqrt(
        (hull_vertices[:, 0] - label_x) ** 2 + (hull_vertices[:, 1] - label_y) ** 2
    )
    nearest_idx = np.argmin(distances)
    return hull_vertices[nearest_idx, 0], hull_vertices[nearest_idx, 1]


def geom_mark_hull(
    ax: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    groups: Optional[np.ndarray] = None,
    labels: Optional[List[str]] = None,
    descriptions: Optional[List[str]] = None,
    hull_color: Union[str, List[str]] = "black",
    hull_fill: Optional[Union[str, List[str]]] = None,
    hull_alpha: float = 0.2,
    hull_linewidth: float = 2.0,
    expand_factor: float = 0.12,
    smoothness: int = 150,
    label_fontsize: int = 12,
    label_fontweight: str = "bold",
    label_color: str = "black",
    label_buffer_factor: float = 0.25,
    connection_linewidth: float = 1.5,
    connection_color: str = "black",
    elbow_angle: float = 45.0,
    min_distance_for_connector: float = 0.2,
    force_points: float = 0.8,
    force_text: float = 1.0,
    use_adjust_text: bool = True,
) -> List[plt.Artist]:
    """
    Annotate groups of points with convex hulls and smart elbow connectors.

    Each group is enclosed by a smooth convex hull with a label connected
    via an elbow connector. Labels are positioned to avoid overlaps.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to draw on
    x, y : array-like
        Point coordinates
    groups : array-like, optional
        Group membership for each point. If None, treats all as one group.
    labels : list of str, optional
        Labels for each group
    descriptions : list of str, optional
        Additional description text for each group
    hull_color : str or list, default='black'
        Hull boundary color(s)
    hull_fill : str or list, optional
        Hull fill color(s)
    hull_alpha : float, default=0.2
        Fill transparency
    hull_linewidth : float, default=2.0
        Hull boundary line width
    expand_factor : float, default=0.12
        Hull expansion factor (0.12 = 12% larger)
    smoothness : int, default=150
        Points for hull smoothing
    label_fontsize : int, default=12
        Font size for labels
    label_fontweight : str, default='bold'
        Font weight for labels
    label_color : str, default='black'
        Text color for labels
    label_buffer_factor : float, default=0.25
        Distance factor from hull to place label
    connection_linewidth : float, default=1.5
        Connector line width
    connection_color : str, default='black'
        Connector line color
    elbow_angle : float, default=45.0
        Angle of connector diagonal segment
    min_distance_for_connector : float, default=0.2
        Minimum distance before drawing connector
    force_points : float, default=0.8
        Repulsion force from points
    force_text : float, default=1.0
        Repulsion force between texts
    use_adjust_text : bool, default=True
        Whether to use adjustText

    Returns
    -------
    list of matplotlib.artist.Artist
        List of artists created
    """
    x = np.asarray(x)
    y = np.asarray(y)
    points = np.column_stack([x, y])

    if groups is None:
        groups = np.zeros(len(x), dtype=int)
    groups = np.asarray(groups)

    unique_groups = np.unique(groups)
    n_groups = len(unique_groups)

    # Prepare labels and descriptions
    if labels is None:
        labels = [f"Group {i+1}" for i in range(n_groups)]
    elif isinstance(labels, str):
        labels = [labels]

    if descriptions is None:
        descriptions = [None] * n_groups
    elif isinstance(descriptions, str):
        descriptions = [descriptions]

    # Normalize colors to lists
    hull_colors = (
        [hull_color] * n_groups if isinstance(hull_color, str) else list(hull_color)
    )
    if hull_fill is None:
        hull_fills = [None] * n_groups
    elif isinstance(hull_fill, str):
        hull_fills = [hull_fill] * n_groups
    else:
        hull_fills = list(hull_fill)

    artists = []
    text_objects = []
    hull_data = []  # Store hull info for connector drawing after adjustText

    for i, group in enumerate(unique_groups):
        mask = groups == group
        group_points = points[mask]

        if len(group_points) < 3:
            warnings.warn(f"Group {group} has fewer than 3 points, skipping hull")
            continue

        # Compute and smooth hull
        hull = ConvexHull(group_points)
        hull_vertices = group_points[hull.vertices]
        hull_vertices = _expand_polygon(hull_vertices, expand_factor)
        smooth_vertices = _smooth_polygon(hull_vertices, smoothness)

        # Draw hull patch
        color_idx = i % len(hull_colors)
        fill_idx = i % len(hull_fills)

        hull_patch = Polygon(
            smooth_vertices,
            closed=True,
            edgecolor=hull_colors[color_idx],
            facecolor=hull_fills[fill_idx] if hull_fills[fill_idx] else "none",
            alpha=hull_alpha if hull_fills[fill_idx] else 1.0,
            linewidth=hull_linewidth,
            zorder=1,
        )
        ax.add_patch(hull_patch)
        artists.append(hull_patch)

        # Calculate initial label position
        min_xy = np.min(smooth_vertices, axis=0)
        max_xy = np.max(smooth_vertices, axis=0)
        center = (min_xy + max_xy) / 2
        size = max_xy - min_xy
        buffer = label_buffer_factor * max(size)

        # Choose best position (furthest from all points)
        candidates = [
            (center[0], max_xy[1] + buffer),  # top
            (center[0], min_xy[1] - buffer),  # bottom
            (max_xy[0] + buffer, center[1]),  # right
            (min_xy[0] - buffer, center[1]),  # left
        ]

        best_pos = max(
            candidates,
            key=lambda c: np.min(
                np.sqrt((points[:, 0] - c[0]) ** 2 + (points[:, 1] - c[1]) ** 2)
            ),
        )

        # Create label text
        label_text = labels[i % len(labels)]
        if descriptions[i % len(descriptions)]:
            label_text = f"{label_text}\n{descriptions[i % len(descriptions)]}"

        text_obj = ax.text(
            best_pos[0],
            best_pos[1],
            label_text,
            fontsize=label_fontsize,
            fontweight=label_fontweight,
            color=label_color,
            ha="center",
            va="center",
            zorder=10,
        )
        text_objects.append(text_obj)

        # Store hull data for connector drawing
        hull_data.append(
            {
                "vertices": smooth_vertices,
                "text_obj": text_obj,
                "color": hull_colors[color_idx],
            }
        )

    # Adjust text positions to avoid overlaps
    if use_adjust_text and HAS_ADJUST_TEXT and text_objects:
        adjust_text(
            text_objects,
            x=points[:, 0],
            y=points[:, 1],
            arrowprops=None,
            expand_points=(1.5, 1.5),
            expand_text=(1.8, 1.8),
            force_points=(force_points, force_points),
            force_text=(force_text, force_text),
            lim=1500,
            only_move={"points": "xy", "text": "xy"},
        )

    # Draw elbow connectors AFTER text positioning
    connector_style = ConnectorStyle(
        linewidth=connection_linewidth,
        color=connection_color,
        elbow_angle=elbow_angle,
        min_distance=min_distance_for_connector,
    )

    for data in hull_data:
        text_obj = data["text_obj"]
        label_x, label_y = text_obj.get_position()

        # Find nearest hull edge point to label
        hull_x, hull_y = _find_hull_edge_point(data["vertices"], label_x, label_y)

        # Adjust text alignment
        dx = label_x - hull_x
        text_obj.set_ha("left" if dx > 0 else "right")
        text_obj.set_va("center")

        # Draw elbow connector from hull edge to label
        connector = _draw_connector(
            ax,
            hull_x,
            hull_y,
            label_x,
            label_y,
            start_radius=0,  # Start from hull edge directly
            style=connector_style,
        )

        if connector:
            artists.append(connector)
        artists.append(text_obj)

    return artists


# Aliases for compatibility
annotate_points_smart = annotate_points
geom_text_repel = annotate_points
