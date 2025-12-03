"""
Heatmap Row Annotation Example

Clear 3-segment connectors:
1. Horizontal from heatmap edge
2. Diagonal (slanted)
3. Horizontal to label

Labels close to heatmap but with enough spacing to not overlap.
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
    min_label_spacing=2.5,  # Increased spacing to prevent overlap
):
    """
    Annotate heatmap rows with clear 3-segment connectors.

    1. Horizontal: from heatmap edge to elbow_x
    2. Diagonal: from elbow to label position
    3. Horizontal: short segment to label
    """
    xlim = ax.get_xlim()
    artists = []

    # Sort by row position - prevents crossing
    sorted_data = sorted(zip(rows_to_annotate, row_labels), key=lambda x: x[0])
    rows_sorted = [x[0] for x in sorted_data]
    labels_sorted = [x[1] for x in sorted_data]
    n_labels = len(rows_sorted)

    # Center labels vertically with enough spacing
    total_height = (n_labels - 1) * min_label_spacing
    y_center = sum(rows_sorted) / n_labels
    y_start = y_center - total_height / 2
    label_y_positions = [y_start + i * min_label_spacing for i in range(n_labels)]

    if side == "right":
        heatmap_edge = n_cols - 0.5
        # Positions for visible 3-segment connector
        elbow_x = heatmap_edge + 0.3  # End of first horizontal
        label_align_x = xlim[1] - 0.25  # Start of third horizontal
        label_x = xlim[1] - 0.05  # Label position
        ha = "left"
    else:
        heatmap_edge = -0.5
        elbow_x = heatmap_edge - 0.3
        label_align_x = xlim[0] + 0.25
        label_x = xlim[0] + 0.05
        ha = "right"

    for row, label, label_y in zip(rows_sorted, labels_sorted, label_y_positions):
        # Clear 3-segment connector
        vertices = np.array(
            [
                [heatmap_edge, row],  # Start at heatmap edge
                [elbow_x, row],  # End of first HORIZONTAL
                [label_align_x, label_y],  # End of DIAGONAL / start of third horizontal
                [label_x, label_y],  # End at label (HORIZONTAL)
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

        text_obj = ax.text(
            label_x + (0.05 if side == "right" else -0.05),
            label_y,
            label,
            fontsize=label_fontsize,
            color=label_color,
            ha=ha,
            va="center",
        )
        artists.append(text_obj)

    return artists


def main():
    """Heatmap with dense row annotations - 3 continuous rows."""
    np.random.seed(123)

    n_rows = 150
    n_cols = 8

    data = np.random.randn(n_rows, n_cols)
    data[70:85, 2:6] += 3

    fig, ax = plt.subplots(figsize=(10, 12))

    im = ax.imshow(data, aspect="auto", cmap="viridis")

    # Annotate 3 CONTINUOUS rows
    rows_to_annotate = [74, 75, 76]
    row_labels = ["Target_A", "Target_B", "Target_C"]

    # Extension for labels - close to heatmap
    ax.set_xlim(-0.5, n_cols + 2.0)

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
        min_label_spacing=2.5,  # Enough spacing to prevent overlap
    )

    # Highlight region
    rect = Rectangle(
        (-0.5, 70 - 0.5),
        n_cols,
        15,
        linewidth=2,
        edgecolor="red",
        facecolor="none",
        linestyle="--",
    )
    ax.add_patch(rect)

    ax.set_xlabel("Samples", fontsize=11)
    ax.set_ylabel("Features (150 rows)", fontsize=11)
    ax.set_title("Heatmap - Dense Row Annotations", fontsize=13, fontweight="bold")
    ax.set_xticks(range(n_cols))
    ax.set_xticklabels([f"S{i+1}" for i in range(n_cols)])

    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    fig.colorbar(im, ax=ax, fraction=0.02, pad=0.08)

    plt.savefig("heatmap_example.png", dpi=150, bbox_inches="tight")
    print("Saved heatmap_example.png")


if __name__ == "__main__":
    main()
