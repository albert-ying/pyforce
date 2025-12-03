"""
Heatmap Row Annotation Example

Uses the package function annotate_heatmap_rows() to annotate
specific rows with 3-segment connectors close to the heatmap.
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from pyforce import annotate_heatmap_rows


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

    # Small extension - labels close to heatmap
    ax.set_xlim(-0.5, n_cols + 1.3)

    # Use package function
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
        min_spacing=2.5,
    )

    # Highlight region
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

    fig.colorbar(im, ax=ax, fraction=0.02, pad=0.12)

    plt.savefig("heatmap_example.png", dpi=150, bbox_inches="tight")
    print("Saved heatmap_example.png")


if __name__ == "__main__":
    main()
