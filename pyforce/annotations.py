"""
Smart annotation system with intelligent connector logic for matplotlib.

This module provides functions for annotating points and groups in matplotlib plots
with automatic label positioning and smart connector lines, inspired by R's ggforce.
"""

import warnings
from dataclasses import dataclass
from enum import Enum
from typing import List, Literal, Optional, Tuple, Union

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


class ConnectorType(Enum):
    """Types of connector lines."""

    HORIZONTAL = "horizontal"  # Simple horizontal line (like the paper example)
    ELBOW = "elbow"  # Diagonal + horizontal segments
    STRAIGHT = "straight"  # Direct line from point to label


@dataclass
class ConnectorStyle:
    """Configuration for connector line style."""

    linewidth: float = 0.8
    color: str = "gray"
    connector_type: ConnectorType = ConnectorType.HORIZONTAL
    elbow_angle: float = 45.0
    min_distance: float = 0.1
    gap: float = 0.02  # Gap between connector end and text


def _get_data_scale(ax: plt.Axes) -> Tuple[float, float]:
    """Get the scale of data units relative to display units."""
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    bbox = ax.get_window_extent()

    x_scale = (xlim[1] - xlim[0]) / bbox.width
    y_scale = (ylim[1] - ylim[0]) / bbox.height

    return x_scale, y_scale


def _point_size_to_radius(point_size: float, ax: plt.Axes) -> float:
    """Convert matplotlib point size to radius in data units."""
    x_scale, y_scale = _get_data_scale(ax)
    avg_scale = (x_scale + y_scale) / 2

    radius_points = np.sqrt(point_size / np.pi)
    radius_data = radius_points * avg_scale * 0.8

    return radius_data


def _create_smart_elbow_connector(
    point_x: float,
    point_y: float,
    label_x: float,
    label_y: float,
    point_radius: float = 0.0,
    gap: float = 0.02,
) -> Path:
    """
    Create a smart elbow connector that chooses the best orientation.

    Automatically decides between:
    1. Horizontal from dot → diagonal to text (when text is mostly horizontally displaced)
    2. Diagonal from dot → horizontal to text (when text is mostly vertically displaced)

    The choice minimizes awkward angles and creates cleaner annotations.
    """
    dx = label_x - point_x
    dy = label_y - point_y
    abs_dx = abs(dx)
    abs_dy = abs(dy)

    # If very small displacement, just draw a straight line
    if abs_dx < 0.01 and abs_dy < 0.01:
        return Path(np.array([[point_x, point_y]]), [Path.MOVETO])

    direction_x = np.sign(dx) if dx != 0 else 1
    direction_y = np.sign(dy) if dy != 0 else 1

    # Start from edge of point toward the label
    dist = np.sqrt(dx**2 + dy**2)
    start_x = point_x + (dx / dist) * point_radius if dist > 0 else point_x
    start_y = point_y + (dy / dist) * point_radius if dist > 0 else point_y

    # End point near the label (with gap)
    end_x = label_x - direction_x * gap if abs_dx > gap else label_x
    end_y = label_y

    # Decide orientation based on displacement ratio
    # If mostly horizontal displacement → horizontal first, then diagonal
    # If mostly vertical displacement → diagonal first, then horizontal

    if abs_dy < 0.01:
        # Pure horizontal - simple line
        vertices = np.array([[start_x, start_y], [end_x, end_y]])
        codes = [Path.MOVETO, Path.LINETO]
    elif abs_dx < 0.01:
        # Pure vertical - simple line
        vertices = np.array([[start_x, start_y], [end_x, end_y]])
        codes = [Path.MOVETO, Path.LINETO]
    elif abs_dx >= abs_dy:
        # Mostly horizontal displacement → horizontal from dot, diagonal to text
        # Elbow is near the TEXT (at ~75% of horizontal distance)
        elbow_fraction = 0.75
        elbow_x = start_x + (end_x - start_x) * elbow_fraction
        elbow_y = start_y  # Stay at dot's Y level

        vertices = np.array(
            [
                [start_x, start_y],
                [elbow_x, elbow_y],
                [end_x, end_y],
            ]
        )
        codes = [Path.MOVETO, Path.LINETO, Path.LINETO]
    else:
        # Mostly vertical displacement → diagonal from dot, horizontal to text
        # Elbow is near the TEXT (at label's Y level, ~25% from end horizontally)
        elbow_x = end_x - (end_x - start_x) * 0.25
        elbow_y = end_y  # At text's Y level

        vertices = np.array(
            [
                [start_x, start_y],
                [elbow_x, elbow_y],
                [end_x, end_y],
            ]
        )
        codes = [Path.MOVETO, Path.LINETO, Path.LINETO]

    return Path(vertices, codes)


def _create_elbow_connector(
    point_x: float,
    point_y: float,
    label_x: float,
    label_y: float,
    point_radius: float = 0.0,
    elbow_angle: float = 45.0,
    gap: float = 0.02,
) -> Path:
    """
    Create an elbow connector (diagonal + horizontal segments).
    """
    dx = label_x - point_x
    dy = label_y - point_y
    dist = np.sqrt(dx**2 + dy**2)

    if dist < 1e-10:
        return Path(np.array([[point_x, point_y]]), [Path.MOVETO])

    # Start from edge of point
    unit_dx = dx / dist
    unit_dy = dy / dist
    start_x = point_x + unit_dx * point_radius
    start_y = point_y + unit_dy * point_radius

    # Elbow at label's Y height
    elbow_y = label_y

    # Calculate elbow x based on angle
    angle_rad = np.radians(elbow_angle)
    vertical_dist = abs(elbow_y - start_y)

    if vertical_dist > 1e-10:
        desired_horiz = vertical_dist / np.tan(angle_rad)
        max_horiz = abs(label_x - start_x) - gap

        if desired_horiz < max_horiz * 0.85:
            elbow_x = start_x + np.sign(dx) * desired_horiz
        else:
            elbow_x = start_x + (label_x - start_x - np.sign(dx) * gap) * 0.65
    else:
        # Nearly horizontal
        elbow_x = start_x + (label_x - start_x) * 0.5
        elbow_y = start_y + dy * 0.5

    # End point (with gap before label)
    end_x = label_x - np.sign(dx) * gap
    end_y = label_y

    vertices = np.array([[start_x, start_y], [elbow_x, elbow_y], [end_x, end_y]])
    codes = [Path.MOVETO, Path.LINETO, Path.LINETO]

    return Path(vertices, codes)


def _create_straight_connector(
    point_x: float,
    point_y: float,
    label_x: float,
    label_y: float,
    point_radius: float = 0.0,
    gap: float = 0.02,
) -> Path:
    """Create a straight line connector from point edge to label."""
    dx = label_x - point_x
    dy = label_y - point_y
    dist = np.sqrt(dx**2 + dy**2)

    if dist < 1e-10:
        return Path(np.array([[point_x, point_y]]), [Path.MOVETO])

    unit_dx = dx / dist
    unit_dy = dy / dist

    start_x = point_x + unit_dx * point_radius
    start_y = point_y + unit_dy * point_radius
    end_x = label_x - unit_dx * gap
    end_y = label_y - unit_dy * gap

    vertices = np.array([[start_x, start_y], [end_x, end_y]])
    codes = [Path.MOVETO, Path.LINETO]

    return Path(vertices, codes)


def _draw_connector(
    ax: plt.Axes,
    point_x: float,
    point_y: float,
    label_x: float,
    label_y: float,
    point_radius: float,
    style: ConnectorStyle,
) -> Optional[PathPatch]:
    """Draw a connector between point and label."""
    dist = np.sqrt((label_x - point_x) ** 2 + (label_y - point_y) ** 2)

    if dist < style.min_distance:
        return None

    # Select connector type
    if style.connector_type == ConnectorType.HORIZONTAL:
        path = _create_smart_elbow_connector(
            point_x, point_y, label_x, label_y, point_radius=point_radius, gap=style.gap
        )
    elif style.connector_type == ConnectorType.ELBOW:
        path = _create_elbow_connector(
            point_x,
            point_y,
            label_x,
            label_y,
            point_radius=point_radius,
            elbow_angle=style.elbow_angle,
            gap=style.gap,
        )
    else:  # STRAIGHT
        path = _create_straight_connector(
            point_x, point_y, label_x, label_y, point_radius=point_radius, gap=style.gap
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


def _compute_smart_label_positions(
    ax: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    indices: np.ndarray,
    labels: List[str],
    point_radius: float,
    label_fontsize: int,
    prefer_direction: str = "right",
    offset_factor: float = 2.5,
) -> List[Tuple[float, float, str]]:
    """
    Compute smart initial positions for labels.

    Returns list of (x, y, ha) tuples where ha is horizontal alignment.
    Prefers placing labels to the right, but switches to left near edges
    or when it would reduce overlap.
    """
    xlim = ax.get_xlim()
    x_range = xlim[1] - xlim[0]

    # Base offset for labels
    base_offset = point_radius * offset_factor

    positions = []

    for idx, label in zip(indices, labels):
        if idx >= len(x):
            continue

        px, py = x[idx], y[idx]

        # Determine direction based on position in plot
        # Default to right, but use left if point is in right 30% of plot
        # or if there are many points to the right

        right_edge_threshold = xlim[1] - 0.3 * x_range
        left_edge_threshold = xlim[0] + 0.3 * x_range

        # Count nearby points on each side
        nearby_mask = np.abs(y - py) < (point_radius * 5)
        points_to_right = np.sum(
            (x[nearby_mask] > px) & (x[nearby_mask] < px + base_offset * 3)
        )
        points_to_left = np.sum(
            (x[nearby_mask] < px) & (x[nearby_mask] > px - base_offset * 3)
        )

        # Decision logic for direction
        if px > right_edge_threshold:
            direction = "left"
        elif px < left_edge_threshold:
            direction = "right"
        elif points_to_right > points_to_left + 2:
            direction = "left"
        else:
            direction = prefer_direction

        if direction == "right":
            label_x = px + base_offset
            ha = "left"
        else:
            label_x = px - base_offset
            ha = "right"

        label_y = py  # Keep at same Y level for horizontal connectors

        positions.append((label_x, label_y, ha))

    return positions


def annotate_points(
    ax: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    labels: List[str],
    indices: Optional[np.ndarray] = None,
    point_size: float = 40,
    min_distance_for_connector: float = 0.1,
    label_fontsize: int = 10,
    label_fontweight: str = "normal",
    label_color: str = "black",
    connection_linewidth: float = 0.8,
    connection_color: str = "gray",
    connector_type: Literal["horizontal", "elbow", "straight"] = "horizontal",
    elbow_angle: float = 45.0,
    connector_gap: float = 0.02,
    prefer_direction: str = "right",
    offset_factor: float = 2.5,
    force_points: float = 1.0,
    force_text: float = 0.5,
    expand_points: float = 2.0,
    expand_text: float = 1.2,
    use_adjust_text: bool = True,
    only_move_text: str = "x",
) -> List[plt.Artist]:
    """
    Annotate points with smart connectors and collision-free labels.

    Supports three connector styles:
    - "horizontal": Simple horizontal lines (like scientific papers)
    - "elbow": Diagonal + horizontal segments
    - "straight": Direct lines from point to label

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to draw on
    x, y : array-like
        All point coordinates in the plot
    labels : list of str
        Text labels for points to annotate
    indices : array-like, optional
        Indices of points to annotate. If None, annotates all.
    point_size : float, default=40
        Size of scatter points (for edge offset calculation)
    min_distance_for_connector : float, default=0.1
        Minimum distance before drawing a connector
    label_fontsize : int, default=10
        Font size for labels
    label_fontweight : str, default='normal'
        Font weight for labels
    label_color : str, default='black'
        Text color
    connection_linewidth : float, default=0.8
        Line width for connectors
    connection_color : str, default='gray'
        Line color for connectors
    connector_type : str, default='horizontal'
        Type of connector: 'horizontal', 'elbow', or 'straight'
    elbow_angle : float, default=45.0
        Angle for elbow connectors (degrees)
    connector_gap : float, default=0.02
        Gap between connector end and text
    prefer_direction : str, default='right'
        Preferred direction for label placement ('right' or 'left')
    offset_factor : float, default=2.5
        Multiplier for label offset from point
    force_points : float, default=1.0
        Repulsion force from points
    force_text : float, default=0.5
        Repulsion force between labels
    expand_points : float, default=2.0
        Expansion factor for point collision boxes
    expand_text : float, default=1.2
        Expansion factor for text collision boxes
    use_adjust_text : bool, default=True
        Whether to use adjustText for positioning
    only_move_text : str, default='x'
        Direction to move text: 'x', 'y', or 'xy'

    Returns
    -------
    list of matplotlib.artist.Artist
        List of created artists
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

    # Calculate point radius
    point_radius = _point_size_to_radius(point_size, ax)

    # Get smart initial positions
    initial_positions = _compute_smart_label_positions(
        ax,
        x,
        y,
        indices,
        labels,
        point_radius,
        label_fontsize,
        prefer_direction=prefer_direction,
        offset_factor=offset_factor,
    )

    # Create text objects at computed positions
    for (idx, label), (lx, ly, ha) in zip(zip(indices, labels), initial_positions):
        if idx >= len(x):
            warnings.warn(f"Index {idx} out of bounds for data of length {len(x)}")
            continue

        point_x, point_y = x[idx], y[idx]
        point_positions.append((point_x, point_y))

        text_obj = ax.text(
            lx,
            ly,
            label,
            fontsize=label_fontsize,
            fontweight=label_fontweight,
            color=label_color,
            ha=ha,
            va="center",
            zorder=10,
        )
        text_objects.append(text_obj)

    # Use adjustText to refine positions
    if use_adjust_text and HAS_ADJUST_TEXT and text_objects:
        # Configure movement direction
        if only_move_text == "x":
            only_move = {"points": "x", "text": "x"}
        elif only_move_text == "y":
            only_move = {"points": "y", "text": "y"}
        else:
            only_move = {"points": "xy", "text": "xy"}

        adjust_text(
            text_objects,
            x=x,
            y=y,
            arrowprops=None,
            expand_points=(expand_points, expand_points),
            expand_text=(expand_text, expand_text),
            force_points=(force_points, force_points),
            force_text=(force_text, force_text),
            lim=2000,
            only_move=only_move,
            avoid_self=True,
        )

    # Determine connector type
    connector_type_enum = {
        "horizontal": ConnectorType.HORIZONTAL,
        "elbow": ConnectorType.ELBOW,
        "straight": ConnectorType.STRAIGHT,
    }.get(connector_type, ConnectorType.HORIZONTAL)

    connector_style = ConnectorStyle(
        linewidth=connection_linewidth,
        color=connection_color,
        connector_type=connector_type_enum,
        elbow_angle=elbow_angle,
        min_distance=min_distance_for_connector,
        gap=connector_gap,
    )

    # Draw connectors
    for text_obj, (point_x, point_y) in zip(text_objects, point_positions):
        label_x, label_y = text_obj.get_position()

        # Update alignment based on final position
        dx = label_x - point_x
        text_obj.set_ha("left" if dx > 0 else "right")

        connector = _draw_connector(
            ax,
            point_x,
            point_y,
            label_x,
            label_y,
            point_radius=point_radius,
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
    connection_linewidth: float = 1.0,
    connection_color: str = "black",
    connector_type: Literal["horizontal", "elbow", "straight"] = "horizontal",
    elbow_angle: float = 45.0,
    connector_gap: float = 0.02,
    min_distance_for_connector: float = 0.1,
    force_points: float = 0.8,
    force_text: float = 1.0,
    use_adjust_text: bool = True,
) -> List[plt.Artist]:
    """
    Annotate groups of points with convex hulls and smart connectors.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to draw on
    x, y : array-like
        Point coordinates
    groups : array-like, optional
        Group membership for each point
    labels : list of str, optional
        Labels for each group
    descriptions : list of str, optional
        Additional description text
    hull_color : str or list, default='black'
        Hull boundary color(s)
    hull_fill : str or list, optional
        Hull fill color(s)
    hull_alpha : float, default=0.2
        Fill transparency
    hull_linewidth : float, default=2.0
        Hull line width
    expand_factor : float, default=0.12
        Hull expansion factor
    smoothness : int, default=150
        Points for hull smoothing
    label_fontsize : int, default=12
        Font size
    label_fontweight : str, default='bold'
        Font weight
    label_color : str, default='black'
        Text color
    label_buffer_factor : float, default=0.25
        Distance from hull to label
    connection_linewidth : float, default=1.0
        Connector line width
    connection_color : str, default='black'
        Connector color
    connector_type : str, default='horizontal'
        Connector style: 'horizontal', 'elbow', or 'straight'
    elbow_angle : float, default=45.0
        Angle for elbow connectors
    connector_gap : float, default=0.02
        Gap between connector and text
    min_distance_for_connector : float, default=0.1
        Minimum distance for connector
    force_points : float, default=0.8
        Repulsion from points
    force_text : float, default=1.0
        Repulsion between labels
    use_adjust_text : bool, default=True
        Use adjustText

    Returns
    -------
    list of matplotlib.artist.Artist
        Created artists
    """
    x = np.asarray(x)
    y = np.asarray(y)
    points = np.column_stack([x, y])

    if groups is None:
        groups = np.zeros(len(x), dtype=int)
    groups = np.asarray(groups)

    unique_groups = np.unique(groups)
    n_groups = len(unique_groups)

    if labels is None:
        labels = [f"Group {i+1}" for i in range(n_groups)]
    elif isinstance(labels, str):
        labels = [labels]

    if descriptions is None:
        descriptions = [None] * n_groups
    elif isinstance(descriptions, str):
        descriptions = [descriptions]

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
    hull_data = []

    for i, group in enumerate(unique_groups):
        mask = groups == group
        group_points = points[mask]

        if len(group_points) < 3:
            warnings.warn(f"Group {group} has fewer than 3 points, skipping hull")
            continue

        hull = ConvexHull(group_points)
        hull_vertices = group_points[hull.vertices]
        hull_vertices = _expand_polygon(hull_vertices, expand_factor)
        smooth_vertices = _smooth_polygon(hull_vertices, smoothness)

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

        # Label position
        min_xy = np.min(smooth_vertices, axis=0)
        max_xy = np.max(smooth_vertices, axis=0)
        center = (min_xy + max_xy) / 2
        size = max_xy - min_xy
        buffer = label_buffer_factor * max(size)

        candidates = [
            (center[0], max_xy[1] + buffer),
            (center[0], min_xy[1] - buffer),
            (max_xy[0] + buffer, center[1]),
            (min_xy[0] - buffer, center[1]),
        ]

        best_pos = max(
            candidates,
            key=lambda c: np.min(
                np.sqrt((points[:, 0] - c[0]) ** 2 + (points[:, 1] - c[1]) ** 2)
            ),
        )

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

        hull_data.append(
            {
                "vertices": smooth_vertices,
                "text_obj": text_obj,
                "color": hull_colors[color_idx],
            }
        )

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

    # Connector type
    connector_type_enum = {
        "horizontal": ConnectorType.HORIZONTAL,
        "elbow": ConnectorType.ELBOW,
        "straight": ConnectorType.STRAIGHT,
    }.get(connector_type, ConnectorType.HORIZONTAL)

    connector_style = ConnectorStyle(
        linewidth=connection_linewidth,
        color=connection_color,
        connector_type=connector_type_enum,
        elbow_angle=elbow_angle,
        min_distance=min_distance_for_connector,
        gap=connector_gap,
    )

    for data in hull_data:
        text_obj = data["text_obj"]
        label_x, label_y = text_obj.get_position()

        hull_x, hull_y = _find_hull_edge_point(data["vertices"], label_x, label_y)

        dx = label_x - hull_x
        text_obj.set_ha("left" if dx > 0 else "right")
        text_obj.set_va("center")

        connector = _draw_connector(
            ax,
            hull_x,
            hull_y,
            label_x,
            label_y,
            point_radius=0,
            style=connector_style,
        )

        if connector:
            artists.append(connector)
        artists.append(text_obj)

    return artists


def _create_margin_connector(
    point_x: float,
    point_y: float,
    label_x: float,
    label_y: float,
    point_radius: float,
    margin_position: str,
    elbow_x: float,
    label_align_x: float,
) -> Path:
    """
    Create a three-segment connector for margin-aligned labels.

    All connectors have aligned elbow points to prevent overlap:
    1. First segment: horizontal from dot to elbow_x (all end at same X)
    2. Second segment: diagonal from elbow to label_align_x
    3. Third segment: short horizontal from label_align_x to label

    Parameters
    ----------
    elbow_x : float
        X position where all first segments end (vertical alignment)
    label_align_x : float
        X position where all third segments start (vertical alignment)
    """
    if margin_position in ("right", "left"):
        direction = 1 if margin_position == "right" else -1

        # Start from edge of dot
        start_x = point_x + direction * point_radius
        start_y = point_y

        # First segment ends at elbow_x (same for all connectors)
        seg1_end_x = elbow_x
        seg1_end_y = point_y

        # Third segment starts at label_align_x (same for all connectors)
        seg3_start_x = label_align_x
        seg3_start_y = label_y

        # End at label
        end_x = label_x
        end_y = label_y

        vertices = np.array(
            [
                [start_x, start_y],
                [seg1_end_x, seg1_end_y],
                [seg3_start_x, seg3_start_y],
                [end_x, end_y],
            ]
        )
    else:
        # Vertical orientation (top/bottom)
        direction = 1 if margin_position == "top" else -1

        start_x = point_x
        start_y = point_y + direction * point_radius

        # For vertical, elbow_x becomes elbow_y, label_align_x becomes label_align_y
        seg1_end_x = point_x
        seg1_end_y = elbow_x  # Using elbow_x as elbow_y for vertical

        seg3_start_x = label_x
        seg3_start_y = label_align_x  # Using label_align_x as label_align_y

        end_x = label_x
        end_y = label_y

        vertices = np.array(
            [
                [start_x, start_y],
                [seg1_end_x, seg1_end_y],
                [seg3_start_x, seg3_start_y],
                [end_x, end_y],
            ]
        )

    codes = [Path.MOVETO, Path.LINETO, Path.LINETO, Path.LINETO]
    return Path(vertices, codes)


def annotate_margin(
    ax: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    labels: List[str],
    indices: Optional[np.ndarray] = None,
    side: Literal["right", "left", "top", "bottom", "both"] = "right",
    point_size: float = 40,
    margin_x: Optional[float] = None,
    margin_y: Optional[float] = None,
    label_fontsize: int = 9,
    label_fontweight: str = "normal",
    label_color: str = "black",
    connection_linewidth: float = 0.6,
    connection_color: str = "gray",
    first_segment_length: Optional[float] = None,
    label_spacing: Optional[float] = None,
    sort_by: Literal["y", "x", "value", "none"] = "y",
) -> List[plt.Artist]:
    """
    Annotate points with labels aligned at a margin position.

    Labels are placed at a fixed x (for left/right) or y (for top/bottom) position,
    with three-segment connectors:
    1. First segment from dot (horizontal/vertical)
    2. Diagonal segment
    3. Final segment to aligned labels

    This is ideal for dense plots where labels would overlap in the middle.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to draw on
    x, y : array-like
        All point coordinates
    labels : list of str
        Text labels for points to annotate
    indices : array-like, optional
        Indices of points to annotate. If None, annotates all.
    side : str, default='right'
        Where to place labels: 'right', 'left', 'top', 'bottom', or 'both'
        For 'both', labels are split between left and right based on x position
    point_size : float, default=40
        Size of scatter points
    margin_x : float, optional
        X position for label margin (for left/right). Auto-calculated if None.
    margin_y : float, optional
        Y position for label margin (for top/bottom). Auto-calculated if None.
    label_fontsize : int, default=9
        Font size for labels
    label_fontweight : str, default='normal'
        Font weight
    label_color : str, default='black'
        Text color
    connection_linewidth : float, default=0.6
        Connector line width
    connection_color : str, default='gray'
        Connector color
    first_segment_length : float, optional
        Length of first horizontal/vertical segment. Auto-calculated if None.
    label_spacing : float, optional
        Vertical/horizontal spacing between labels. Auto-calculated if None.
    sort_by : str, default='y'
        How to sort labels: 'y', 'x', 'value', or 'none'

    Returns
    -------
    list of matplotlib.artist.Artist
        List of created artists
    """
    x = np.asarray(x)
    y = np.asarray(y)

    if indices is None:
        indices = np.arange(len(labels)) if len(labels) <= len(x) else np.arange(len(x))
    indices = np.asarray(indices)

    if isinstance(labels, str):
        labels = [labels]

    # Get axis limits
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    x_range = xlim[1] - xlim[0]
    y_range = ylim[1] - ylim[0]

    # Calculate point radius
    point_radius = _point_size_to_radius(point_size, ax)

    # Auto-calculate parameters
    if first_segment_length is None:
        first_segment_length = x_range * 0.015

    artists = []

    # Collect point data
    point_data = []
    for idx, label in zip(indices, labels):
        if idx >= len(x):
            continue
        point_data.append(
            {
                "idx": idx,
                "x": x[idx],
                "y": y[idx],
                "label": label,
            }
        )

    if not point_data:
        return artists

    # Split points for 'both' mode
    if side == "both":
        x_mid = (xlim[0] + xlim[1]) / 2
        left_points = [p for p in point_data if p["x"] < x_mid]
        right_points = [p for p in point_data if p["x"] >= x_mid]

        # Recursively annotate each side
        if left_points:
            left_labels = [p["label"] for p in left_points]
            left_indices = [p["idx"] for p in left_points]
            artists.extend(
                annotate_margin(
                    ax,
                    x,
                    y,
                    left_labels,
                    indices=left_indices,
                    side="left",
                    point_size=point_size,
                    margin_x=margin_x,
                    label_fontsize=label_fontsize,
                    label_fontweight=label_fontweight,
                    label_color=label_color,
                    connection_linewidth=connection_linewidth,
                    connection_color=connection_color,
                    first_segment_length=first_segment_length,
                    label_spacing=label_spacing,
                    sort_by=sort_by,
                )
            )
        if right_points:
            right_labels = [p["label"] for p in right_points]
            right_indices = [p["idx"] for p in right_points]
            artists.extend(
                annotate_margin(
                    ax,
                    x,
                    y,
                    right_labels,
                    indices=right_indices,
                    side="right",
                    point_size=point_size,
                    margin_x=margin_x,
                    label_fontsize=label_fontsize,
                    label_fontweight=label_fontweight,
                    label_color=label_color,
                    connection_linewidth=connection_linewidth,
                    connection_color=connection_color,
                    first_segment_length=first_segment_length,
                    label_spacing=label_spacing,
                    sort_by=sort_by,
                )
            )
        return artists

    # CRITICAL: Sort by y position (highest first) to prevent line crossing
    if side in ("right", "left"):
        point_data.sort(key=lambda p: p["y"], reverse=True)
    elif side in ("top", "bottom"):
        point_data.sort(key=lambda p: p["x"])

    n_labels = len(point_data)

    # Calculate margin position and label positions
    if side in ("right", "left"):
        direction = 1 if side == "right" else -1

        # Find extremes of annotated dots
        if side == "right":
            max_dot_x = max(p["x"] for p in point_data)
        else:
            min_dot_x = min(p["x"] for p in point_data)

        # Margin position - CLOSER to dots (not at plot edge)
        if margin_x is None:
            if side == "right":
                margin_x = max_dot_x + x_range * 0.12  # Close to rightmost dot
            else:
                margin_x = min_dot_x - x_range * 0.12  # Close to leftmost dot

        # Aligned X positions for non-overlapping lines:
        if side == "right":
            elbow_x = max_dot_x + x_range * 0.02  # Just past farthest dot
            label_align_x = margin_x - x_range * 0.015  # Short final segment
        else:
            elbow_x = min_dot_x - x_range * 0.02
            label_align_x = margin_x + x_range * 0.015

        # Minimum spacing between labels
        min_spacing = label_spacing if label_spacing else y_range * 0.03

        # Calculate label y positions - CENTER labels vertically in plot
        # and ensure they maintain the same rank order as dots
        total_label_height = (n_labels - 1) * min_spacing
        y_center = (ylim[0] + ylim[1]) / 2
        y_start = y_center + total_label_height / 2

        label_y_positions = [y_start - i * min_spacing for i in range(n_labels)]

        ha = "left" if side == "right" else "right"

        for i, p in enumerate(point_data):
            label_y = label_y_positions[i]
            label_x = margin_x

            # Create 3-segment connector with aligned positions
            path = _create_margin_connector(
                p["x"],
                p["y"],
                label_x,
                label_y,
                point_radius,
                side,
                elbow_x,
                label_align_x,
            )

            patch = PathPatch(
                path,
                facecolor="none",
                edgecolor=connection_color,
                linewidth=connection_linewidth,
                zorder=5,
                capstyle="round",
                joinstyle="round",
            )
            ax.add_patch(patch)
            artists.append(patch)

            # Create label
            text_obj = ax.text(
                label_x,
                label_y,
                p["label"],
                fontsize=label_fontsize,
                fontweight=label_fontweight,
                color=label_color,
                ha=ha,
                va="center",
                zorder=10,
            )
            artists.append(text_obj)

    else:  # top or bottom
        # Vertical margin
        if margin_y is None:
            if side == "top":
                margin_y = ylim[1] + y_range * 0.02
            else:
                margin_y = ylim[0] - y_range * 0.02

        # Calculate label spacing
        if label_spacing is None:
            x_min = min(p["x"] for p in point_data)
            x_max = max(p["x"] for p in point_data)
            x_span = max(x_max - x_min, x_range * 0.5)
            label_spacing = x_span / max(n_labels - 1, 1) if n_labels > 1 else 0

        # Sort by x for top/bottom
        point_data.sort(key=lambda p: p["x"])

        x_center = sum(p["x"] for p in point_data) / n_labels
        x_start = x_center - (n_labels - 1) * label_spacing / 2

        va = "bottom" if side == "top" else "top"
        rotation = 90 if side == "top" else -90

        for i, p in enumerate(point_data):
            label_x = x_start + i * label_spacing
            label_y = margin_y

            # Create connector
            path = _create_margin_connector(
                p["x"],
                p["y"],
                label_x,
                label_y,
                point_radius,
                side,
                first_segment_length,
            )

            patch = PathPatch(
                path,
                facecolor="none",
                edgecolor=connection_color,
                linewidth=connection_linewidth,
                zorder=5,
                capstyle="round",
                joinstyle="round",
            )
            ax.add_patch(patch)
            artists.append(patch)

            # Create rotated label
            text_obj = ax.text(
                label_x,
                label_y,
                p["label"],
                fontsize=label_fontsize,
                fontweight=label_fontweight,
                color=label_color,
                ha="center",
                va=va,
                rotation=rotation,
                rotation_mode="anchor",
                zorder=10,
            )
            artists.append(text_obj)

    return artists


def annotate_edge(
    ax: plt.Axes,
    y_positions: List[float],
    labels: List[str],
    x_start: Optional[float] = None,
    label_x: Optional[float] = None,
    side: Literal["right", "left"] = "right",
    label_fontsize: int = 9,
    label_color: str = "black",
    line_color: str = "gray",
    linewidth: float = 0.6,
    min_spacing: Optional[float] = None,
) -> List[plt.Artist]:
    """
    Smart edge annotation for any plot (heatmap, line plot, etc.).

    Automatically detects when dodge is needed:
    - Far apart: simple horizontal line
    - Close together: 3-segment connector (1/3 + 1/3 + 1/3 ratio)

    Works for heatmaps (annotate rows), line plots (annotate at end), etc.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to annotate
    y_positions : list of float
        Y positions to annotate (row indices for heatmap, y-values for line plot)
    labels : list of str
        Labels for each position
    x_start : float, optional
        X position where connectors start. If None, uses plot edge.
    label_x : float, optional
        X position where labels are placed. If None, defaults to x_start + 0.2
        (or x_start - 0.2 for left side). Falls back to plot edge if x_start is also None.
    side : str, default='right'
        'right' or 'left'
    label_fontsize : int, default=9
        Font size for labels
    label_color : str, default='black'
        Color for labels
    line_color : str, default='gray'
        Color for connector lines
    linewidth : float, default=0.6
        Width of connector lines
    min_spacing : float, optional
        Minimum spacing between labels. If None, auto-calculated.

    Returns
    -------
    list of matplotlib.artist.Artist
        Created artists (patches and text)

    Example
    -------
    >>> # For heatmap
    >>> annotate_edge(ax, [10, 11, 12], ['A', 'B', 'C'], x_start=n_cols-0.5)
    >>> # For line plot
    >>> annotate_edge(ax, [y1[-1], y2[-1]], ['Line 1', 'Line 2'])
    >>> # With custom label position
    >>> annotate_edge(ax, [10, 11], ['A', 'B'], x_start=n_cols-0.5, label_x=n_cols+0.5)
    """
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    y_range = ylim[1] - ylim[0]
    artists = []

    # Auto-calculate min_spacing if not provided
    if min_spacing is None:
        # Convert font size (points) to data coordinates
        # 1 point = 1/72 inch
        fig = ax.get_figure()
        fig_height_inches = fig.get_figheight()
        # Get axes height as fraction of figure
        bbox = ax.get_position()
        axes_height_inches = fig_height_inches * bbox.height
        # Data units per inch (use abs for inverted axes like heatmaps)
        data_per_inch = abs(y_range) / axes_height_inches
        # Text height in data units (with 1.4x padding for comfortable spacing)
        text_height_data = (label_fontsize / 72) * data_per_inch * 1.4
        min_spacing = text_height_data

    # Sort by y position
    sorted_data = sorted(zip(y_positions, labels), key=lambda x: x[0])
    y_sorted = [x[0] for x in sorted_data]
    labels_sorted = [x[1] for x in sorted_data]

    if side == "right":
        edge_x = x_start if x_start is not None else xlim[1]
        if label_x is None:
            label_x = (x_start + 0.2) if x_start is not None else (xlim[1] + 0.01)
        ha = "left"
        direction = 1
    else:
        edge_x = x_start if x_start is not None else xlim[0]
        if label_x is None:
            label_x = (x_start - 0.2) if x_start is not None else (xlim[0] - 0.01)
        ha = "right"
        direction = -1

    # Total horizontal distance for connector
    total_width = abs(label_x - edge_x)

    # Smart label positioning: only dodge when labels would overlap
    label_y_positions = []
    for i, y in enumerate(y_sorted):
        if i == 0:
            label_y_positions.append(float(y))
        else:
            prev_label_y = label_y_positions[-1]
            if y - prev_label_y < min_spacing:
                # Need to dodge
                label_y_positions.append(prev_label_y + min_spacing)
            else:
                # No dodge needed
                label_y_positions.append(float(y))

    for y, label, label_y in zip(y_sorted, labels_sorted, label_y_positions):
        needs_dodge = abs(label_y - y) > 0.01

        if needs_dodge:
            # 3-segment connector with EQUAL 1/3 ratio
            seg_width = total_width / 3
            elbow_x = edge_x + direction * seg_width
            label_align_x = label_x - direction * seg_width

            vertices = np.array(
                [
                    [edge_x, y],
                    [elbow_x, y],
                    [label_align_x, label_y],
                    [label_x, label_y],
                ]
            )
            codes = [Path.MOVETO, Path.LINETO, Path.LINETO, Path.LINETO]
        else:
            # Simple horizontal line
            vertices = np.array(
                [
                    [edge_x, y],
                    [label_x, y],
                ]
            )
            codes = [Path.MOVETO, Path.LINETO]

        path = Path(vertices, codes)
        patch = PathPatch(
            path,
            facecolor="none",
            edgecolor=line_color,
            linewidth=linewidth,
            capstyle="round",
            joinstyle="round",
        )
        ax.add_patch(patch)
        artists.append(patch)

        text_obj = ax.text(
            label_x + direction * 0.01,
            label_y,
            label,
            fontsize=label_fontsize,
            color=label_color,
            ha=ha,
            va="center",
        )
        artists.append(text_obj)

    return artists


def annotate_heatmap_rows(
    ax: plt.Axes,
    rows_to_annotate: List[int],
    row_labels: List[str],
    n_cols: int,
    side: Literal["right", "left"] = "right",
    **kwargs,
) -> List[plt.Artist]:
    """
    Convenience wrapper for annotate_edge() for heatmaps.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes containing the heatmap
    rows_to_annotate : list of int
        Row indices to annotate
    row_labels : list of str
        Labels for each row
    n_cols : int
        Number of columns in heatmap
    side : str, default='right'
        'right' or 'left'
    **kwargs
        Additional arguments passed to annotate_edge()

    Returns
    -------
    list of matplotlib.artist.Artist
        Created artists
    """
    x_start = n_cols - 0.5 if side == "right" else -0.5
    return annotate_edge(
        ax,
        y_positions=[float(r) for r in rows_to_annotate],
        labels=row_labels,
        x_start=x_start,
        side=side,
        **kwargs,
    )


# Aliases for compatibility
annotate_points_smart = annotate_points
geom_text_repel = annotate_points
