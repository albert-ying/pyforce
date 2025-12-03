"""
Heatmap Row Annotation Example

Demonstrates how to annotate specific rows in a heatmap with many rows.
Uses compact 3-segment connectors: short horizontal → short diagonal → short horizontal
This is useful when rows to annotate are close together.
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import PathPatch
from matplotlib.path import Path


def annotate_heatmap_rows(
    ax,
    rows_to_annotate,
    row_labels,
    n_cols,
    side="right",
    label_fontsize=9,
    label_color="black",
    line_color="gray",
    linewidth=0.6,
    first_segment_length=0.3,
    final_segment_length=0.1,
    min_label_spacing=1.2,
):
    """
    Annotate specific rows in a heatmap with compact 3-segment connectors.

    Connector structure:
    1. Short horizontal from heatmap edge
    2. Short diagonal to separate labels
    3. Short horizontal to label

    Labels are sorted by row position (top to bottom) to prevent crossing.

    Parameters
    ----------
    ax : matplotlib axes
        The axes containing the heatmap
    rows_to_annotate : list
        Row indices to annotate
    row_labels : list
        Labels for each row
    n_cols : int
        Number of columns in heatmap
    side : str
        'right' or 'left'
    first_segment_length : float
        Length of first horizontal segment
    final_segment_length : float
        Length of final horizontal segment (should be short)
    min_label_spacing : float
        Minimum vertical spacing between labels
    """
    xlim = ax.get_xlim()
    artists = []

    # Sort by row position (top to bottom) - CRITICAL to prevent crossing
    sorted_data = sorted(zip(rows_to_annotate, row_labels), key=lambda x: x[0])
    rows_sorted = [x[0] for x in sorted_data]
    labels_sorted = [x[1] for x in sorted_data]

    # Calculate label positions - stay at row y, only adjust if overlap
    label_y_positions = []
    for i, row in enumerate(rows_sorted):
        if i == 0:
            label_y_positions.append(float(row))
        else:
            prev_y = label_y_positions[-1]
            # Labels go down (increasing row index), so new y should be >= prev + spacing
            min_allowed_y = prev_y + min_label_spacing
            label_y_positions.append(max(float(row), min_allowed_y))

    if side == "right":
        start_x = n_cols - 0.5
        label_x = xlim[1] - 0.1
        ha = "left"
        direction = 1
    else:
        start_x = -0.5
        label_x = xlim[0] + 0.1
        ha = "right"
        direction = -1

    for row, label, label_y in zip(rows_sorted, labels_sorted, label_y_positions):
        # 3-segment connector: horizontal → diagonal → horizontal
        seg1_end_x = start_x + direction * first_segment_length
        seg3_start_x = label_x - direction * final_segment_length

        vertices = np.array(
            [
                [start_x, row],  # Start at heatmap edge
                [seg1_end_x, row],  # End of first horizontal
                [seg3_start_x, label_y],  # After diagonal, before final horizontal
                [label_x, label_y],  # At label
            ]
        )
        codes = [Path.MOVETO, Path.LINETO, Path.LINETO, Path.LINETO]

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

        # Add label
        text_obj = ax.text(
            label_x + direction * 0.05,
            label_y,
            label,
            fontsize=label_fontsize,
            color=label_color,
            ha=ha,
            va="center",
        )
        artists.append(text_obj)

    return artists


def create_heatmap_right_annotation():
    """Heatmap with compact annotations on the right side."""
    np.random.seed(42)

    n_rows = 80
    n_cols = 6

    # Create clustered data
    data = np.random.randn(n_rows, n_cols)
    data[:20, :3] += 2
    data[20:40, 3:] += 2
    data[60:70, :] += 1.5

    fig, ax = plt.subplots(figsize=(10, 12))

    im = ax.imshow(data, aspect="auto", cmap="RdBu_r", vmin=-3, vmax=3)

    # Rows to annotate - some close together, some spread out
    rows_to_annotate = [5, 12, 18, 25, 33, 38, 45, 52, 63, 67, 72]
    row_labels = [f"Gene_{r}" for r in rows_to_annotate]

    # Extend plot for labels (compact - not too much space)
    ax.set_xlim(-0.5, n_cols + 2.5)

    annotate_heatmap_rows(
        ax,
        rows_to_annotate,
        row_labels,
        n_cols,
        side="right",
        label_fontsize=9,
        label_color="darkblue",
        line_color="gray",
        linewidth=0.6,
        first_segment_length=0.4,
        final_segment_length=0.15,
        min_label_spacing=1.5,
    )

    ax.set_xlabel("Samples", fontsize=11)
    ax.set_ylabel("Features (rows)", fontsize=11)
    ax.set_title("Heatmap - Right Annotations", fontsize=13, fontweight="bold")
    ax.set_xticks(range(n_cols))
    ax.set_xticklabels([f"S{i+1}" for i in range(n_cols)])

    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    fig.colorbar(im, ax=ax, fraction=0.02, pad=0.08)

    plt.savefig("heatmap_right_example.png", dpi=150, bbox_inches="tight")
    print("Saved heatmap_right_example.png")


def create_heatmap_left_annotation():
    """Heatmap with compact annotations on the left side."""
    np.random.seed(42)

    n_rows = 80
    n_cols = 6

    data = np.random.randn(n_rows, n_cols)
    data[:20, :3] += 2
    data[20:40, 3:] += 2
    data[60:70, :] += 1.5

    fig, ax = plt.subplots(figsize=(10, 12))

    im = ax.imshow(data, aspect="auto", cmap="RdBu_r", vmin=-3, vmax=3)

    rows_to_annotate = [5, 12, 18, 25, 33, 38, 45, 52, 63, 67, 72]
    row_labels = [f"Gene_{r}" for r in rows_to_annotate]

    # Extend plot for labels on left
    ax.set_xlim(-3.0, n_cols - 0.5)

    annotate_heatmap_rows(
        ax,
        rows_to_annotate,
        row_labels,
        n_cols,
        side="left",
        label_fontsize=9,
        label_color="darkgreen",
        line_color="gray",
        linewidth=0.6,
        first_segment_length=0.4,
        final_segment_length=0.15,
        min_label_spacing=1.5,
    )

    ax.set_xlabel("Samples", fontsize=11)
    ax.set_ylabel("Features (rows)", fontsize=11)
    ax.yaxis.set_label_coords(-0.2, 0.5)
    ax.set_title("Heatmap - Left Annotations", fontsize=13, fontweight="bold")
    ax.set_xticks(range(n_cols))
    ax.set_xticklabels([f"S{i+1}" for i in range(n_cols)])

    ax.spines["left"].set_visible(False)
    ax.spines["top"].set_visible(False)

    fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02)

    plt.savefig("heatmap_left_example.png", dpi=150, bbox_inches="tight")
    print("Saved heatmap_left_example.png")


def create_heatmap_dense_region():
    """
    Heatmap with MANY rows, annotating a dense region.
    This demonstrates the compact connector style for tightly clustered rows.
    """
    np.random.seed(123)

    n_rows = 200  # Many rows
    n_cols = 8

    data = np.random.randn(n_rows, n_cols)
    # Create a notable region
    data[90:110, 2:6] += 3

    fig, ax = plt.subplots(figsize=(12, 14))

    im = ax.imshow(data, aspect="auto", cmap="viridis")

    # Annotate rows in a very dense region (rows close together)
    rows_to_annotate = [92, 94, 96, 98, 100, 102, 104, 106]
    row_labels = [
        "Marker_A", "Marker_B", "Target_1", "Target_2",
        "Gene_X", "Gene_Y", "Factor_1", "Factor_2"
    ]

    # Extend plot for labels - keep compact
    ax.set_xlim(-0.5, n_cols + 4)

    annotate_heatmap_rows(
        ax,
        rows_to_annotate,
        row_labels,
        n_cols,
        side="right",
        label_fontsize=9,
        label_color="black",
        line_color="dimgray",
        linewidth=0.8,
        first_segment_length=0.4,
        final_segment_length=0.15,
        min_label_spacing=2.5,  # More spacing for dense region
    )

    # Highlight the annotated region
    from matplotlib.patches import Rectangle

    rect = Rectangle(
        (-0.5, 90 - 0.5),
        n_cols,
        20,
        linewidth=2,
        edgecolor="red",
        facecolor="none",
        linestyle="--",
    )
    ax.add_patch(rect)

    ax.set_xlabel("Samples", fontsize=11)
    ax.set_ylabel("Features (200 rows)", fontsize=11)
    ax.set_title("Large Heatmap - Dense Region Annotations", fontsize=13, fontweight="bold")
    ax.set_xticks(range(n_cols))
    ax.set_xticklabels([f"S{i+1}" for i in range(n_cols)])

    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    fig.colorbar(im, ax=ax, fraction=0.02, pad=0.08)

    plt.savefig("heatmap_dense_example.png", dpi=150, bbox_inches="tight")
    print("Saved heatmap_dense_example.png")


def main():
    create_heatmap_right_annotation()
    create_heatmap_left_annotation()
    create_heatmap_dense_region()


if __name__ == "__main__":
    main()
