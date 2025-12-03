"""
Heatmap Row Annotation Example

Demonstrates compact 3-segment connectors for annotating specific rows
in a heatmap with many rows. The connectors are short and close to the heatmap.
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import PathPatch, Rectangle
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
    first_segment_length=0.2,
    final_segment_length=0.08,
    min_label_spacing=1.0,
):
    """
    Annotate specific rows in a heatmap with compact 3-segment connectors.

    The connectors are short and stay close to the heatmap:
    1. Short horizontal from heatmap edge
    2. Short diagonal to separate labels
    3. Very short horizontal to label

    Labels are sorted by row position to prevent line crossing.
    """
    xlim = ax.get_xlim()
    artists = []

    # Sort by row position (top to bottom) - prevents crossing
    sorted_data = sorted(zip(rows_to_annotate, row_labels), key=lambda x: x[0])
    rows_sorted = [x[0] for x in sorted_data]
    labels_sorted = [x[1] for x in sorted_data]

    # Calculate label positions - stay at row y, adjust only if overlap
    label_y_positions = []
    for i, row in enumerate(rows_sorted):
        if i == 0:
            label_y_positions.append(float(row))
        else:
            prev_y = label_y_positions[-1]
            min_allowed_y = prev_y + min_label_spacing
            label_y_positions.append(max(float(row), min_allowed_y))

    if side == "right":
        start_x = n_cols - 0.5
        label_x = xlim[1] - 0.05
        ha = "left"
        direction = 1
    else:
        start_x = -0.5
        label_x = xlim[0] + 0.05
        ha = "right"
        direction = -1

    for row, label, label_y in zip(rows_sorted, labels_sorted, label_y_positions):
        # Compact 3-segment connector
        seg1_end_x = start_x + direction * first_segment_length
        seg3_start_x = label_x - direction * final_segment_length

        vertices = np.array([
            [start_x, row],
            [seg1_end_x, row],
            [seg3_start_x, label_y],
            [label_x, label_y],
        ])
        codes = [Path.MOVETO, Path.LINETO, Path.LINETO, Path.LINETO]

        path = Path(vertices, codes)
        patch = PathPatch(
            path, facecolor="none", edgecolor=line_color,
            linewidth=linewidth, capstyle="round", joinstyle="round",
        )
        ax.add_patch(patch)
        artists.append(patch)

        text_obj = ax.text(
            label_x + direction * 0.03, label_y, label,
            fontsize=label_fontsize, color=label_color, ha=ha, va="center",
        )
        artists.append(text_obj)

    return artists


def main():
    """Heatmap with dense row annotations - 3 continuous rows."""
    np.random.seed(123)

    n_rows = 150
    n_cols = 8

    data = np.random.randn(n_rows, n_cols)
    # Create a notable region
    data[70:85, 2:6] += 3

    fig, ax = plt.subplots(figsize=(10, 12))

    im = ax.imshow(data, aspect="auto", cmap="viridis")

    # Annotate 3 CONTINUOUS rows very close together
    rows_to_annotate = [74, 75, 76]
    row_labels = ["Target_A", "Target_B", "Target_C"]

    # Minimal extension - keep labels close to heatmap
    ax.set_xlim(-0.5, n_cols + 1.8)

    annotate_heatmap_rows(
        ax,
        rows_to_annotate,
        row_labels,
        n_cols,
        side="right",
        label_fontsize=10,
        label_color="black",
        line_color="dimgray",
        linewidth=0.8,
        first_segment_length=0.15,
        final_segment_length=0.05,
        min_label_spacing=1.2,
    )

    # Highlight the annotated region
    rect = Rectangle(
        (-0.5, 70 - 0.5), n_cols, 15,
        linewidth=2, edgecolor="red", facecolor="none", linestyle="--",
    )
    ax.add_patch(rect)

    ax.set_xlabel("Samples", fontsize=11)
    ax.set_ylabel("Features (150 rows)", fontsize=11)
    ax.set_title("Heatmap - Dense Row Annotations", fontsize=13, fontweight="bold")
    ax.set_xticks(range(n_cols))
    ax.set_xticklabels([f"S{i+1}" for i in range(n_cols)])

    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    fig.colorbar(im, ax=ax, fraction=0.02, pad=0.15)

    plt.savefig("heatmap_example.png", dpi=150, bbox_inches="tight")
    print("Saved heatmap_example.png")


if __name__ == "__main__":
    main()
